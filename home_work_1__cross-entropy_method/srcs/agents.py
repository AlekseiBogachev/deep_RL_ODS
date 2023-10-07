import numpy as np
import time



class Agent():

    def __init__(self, n_actions, seed=None, vis_delay=0.1):
        self.n_actions = n_actions
        self.seed = seed
        self.vis_delay = vis_delay

    
    def fit(self, trajectories, rewards):
        raise NotImplementedError

    
    def get_action(self, state):
        raise NotImplementedError
    
    
    def get_trajectory(self, env, visualise=False):

        trajectory = dict(
            iterations = [],
            states = [],
            actions = [],
            rewards = [],
        )

        state = env.reset(seed=self.seed)
        done = False
        iter = 0
        while not done:
            trajectory['states'].append(state)

            action = self.get_action(state)
            trajectory['actions'].append(action)

            state, reward, done, info = env.step(action)
            trajectory['rewards'].append(reward)
            
            if visualise:
                print(f'Iter: {iter}. acton: {action} -> state: {state}, '
                    f'reward: {reward}, done: {done}, info: {info}.')

                env.render()
                time.sleep(self.vis_delay)

            iter += 1
            trajectory['iterations'].append(iter)

        return trajectory
    

    def evaluate_policy(self, env, n_trajectories):
        raise NotImplementedError
    

    def improve_policy(self):
        pass


class RandomAgent(Agent):

    def get_action(self, state):
        return np.random.randint(self.n_actions)



class CrossEntropyAgent(Agent):

    def __init__(self, 
                 n_actions, 
                 n_states, 
                 n_trajectories, 
                 q, 
                 seed=None, 
                 vis_delay=0.1
                ):
        
        self.n_actions = n_actions
        self.n_states = n_states
        self.n_trajectories = n_trajectories
        self.q = q
        self.seed = seed
        self.vis_delay = vis_delay

        self.policy_ = np.ones([self.n_states, self.n_actions]) / self.n_actions


    def get_action(self, state):

        action = np.random.choice(
            a=np.arange(stop=self.n_actions, dtype='int'),
            p=self.policy_[state]
        )

        return action
    

    def evaluate_policy(self, env):

        trajectories = [
            self.get_trajectory(env) for _ in range(self.n_trajectories)
        ]

        total_rewards = [
            sum(trajectory['rewards']) for trajectory in trajectories
        ]

        return total_rewards, trajectories
    

    def improve_policy(self, total_rewards, trajectories):
        quantile = np.quantile(total_rewards, self.q)

        elite_trajectories = [
            trajectory 
            for total_reward, trajectory in zip(total_rewards, trajectories) 
            if total_reward >= quantile
        ]

        new_policy = np.zeros([self.n_states, self.n_actions])

        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_policy[state, action] += 1

        for state in range(self.n_states):
            if np.sum(new_policy[state]) > 0:
                new_policy[state] /= np.sum(new_policy[state])
            else:
                new_policy[state] = self.policy_[state].copy()

        self.policy_ = new_policy


    def fit(self, env, n_iterations=100, verbose=False):

        for iter in range(n_iterations):
            total_rewards, trajectories = self.evaluate_policy(env)

            if verbose:
                print(
                    f'Iteration {iter}: mean total reward: '
                    f'{sum(total_rewards) / len(total_rewards)}'
                )

            self.improve_policy(total_rewards, trajectories)



class SmoothCrossEntropyAgent(CrossEntropyAgent):

    def __init__(self, 
                 n_actions, 
                 n_states, 
                 n_trajectories, 
                 q, 
                 laplace_smoothing_lambda=0,
                 policy_smoothing_lambda=1,
                 seed=None, 
                 vis_delay=0.1
                ):
        
        self.n_actions = n_actions
        self.n_states = n_states
        self.n_trajectories = n_trajectories
        self.q = q
        self.laplace_smoothing_lambda = laplace_smoothing_lambda
        self.policy_smoothing_lambda = policy_smoothing_lambda 
        self.seed = seed
        self.vis_delay = vis_delay

        self.policy_ = np.ones([self.n_states, self.n_actions]) / self.n_actions


    def improve_policy(self, total_rewards, trajectories):
        quantile = np.quantile(total_rewards, self.q)

        elite_trajectories = [
            trajectory 
            for total_reward, trajectory in zip(total_rewards, trajectories) 
            if total_reward >= quantile
        ]

        new_policy = np.zeros([self.n_states, self.n_actions])

        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_policy[state, action] += 1

        for state in range(self.n_states):
            if np.sum(new_policy[state]) > 0:
                new_policy[state] = (new_policy[state] + self.laplace_smoothing_lambda) / (np.sum(new_policy[state])+ self.laplace_smoothing_lambda * self.n_actions)
                new_policy[state] = new_policy[state] + (1 - self.policy_smoothing_lambda) * self.policy_[state]
                new_policy[state] = new_policy[state] / np.sum(new_policy[state])
            else:
                new_policy[state] = self.policy_[state].copy()

        self.policy_ = new_policy