import gym
import numpy as np

from agents import SmoothCrossEntropyAgent

MAX_STEPS = 100
SEED = None



def print_env_info(env):

    print(f'Число состояний среды: {env.observation_space.n}')
    print(f'Число действий агента: {env.action_space.n}')
    print(f'Максимальное число шагов: {env._max_episode_steps}')
    


if __name__ == '__main__':

    env = gym.make('Taxi-v3', max_episode_steps=MAX_STEPS)
    print_env_info(env)

    agent = SmoothCrossEntropyAgent(
        n_actions=env.action_space.n, 
        n_states=env.observation_space.n, 
        n_trajectories=10000, 
        q=0.9,
        laplace_smoothing_lambda=0.1,
        policy_smoothing_lambda=0.001,
        seed=SEED
    )

    agent.fit(
        env=env, 
        n_iterations=10, 
        verbose=True
    )

    state = env.reset(seed=SEED)
    print(f'Начальное состояние: {state}')

    trajectory = agent.get_trajectory(env, visualise=True)

    print('Суммарная награда', sum(trajectory['rewards']))

    env.close()