from yams_env import YamsEnvNoe as YamsEnv
from random import choice
import numpy as np
from tqdm import tqdm

def first_visit_MC_prediction(env:YamsEnv, gamma:float=0.9, max_iter:int=100, seed:int=None):
    Returns = {}
    if seed is not None:
        np.random.seed(seed)
    Q = {}
    for s in env.states:
        all_actions = env.list_actions(s)
        for a in all_actions:
            Returns[(s,a)] = []
            Q[(s,a)] = np.random.random() # Random policy
    for i in tqdm(range(max_iter)):
        episode = env.generate_episode(Q)
        for i, (s, a, r, next_s) in enumerate(reversed(episode)):
            G = r + sum(map(lambda x: x[2], episode[:i]))
            Returns[(s, a)].append(G)
            Q[(s, a)] = float(np.mean(Returns[(s,a)]))
    return Q


if __name__ == '__main__':
    from figures import Multiple, Chance, Number, Brelan, Suite
    env = YamsEnv(3, 3, [Number(0), Number(1), Number(2), Multiple(3, 7), Multiple(3, 7), Multiple(3, 56)])
    #env = YamsEnv(4, 5, [Multiple(4, 20), Brelan(), Suite(1, 4, 20), Suite(2, 5, 20), Number(0), Number(1), Number(2), Number(3), Number(4)])
    for j in range(10):
        best_Q = first_visit_MC_prediction(env, max_iter=1000, seed=j)
        randomQ_rewards = []
        bestQ_rewards = []
        np.random.seed(j)
        Q = {}
        for s in env.states:
            all_actions = env.list_actions(s)
            for a in all_actions:
                Q[(s,a)] = np.random.random() # Random policy
        k = 100
        # Test with a random policy
        for i in tqdm(range(k), desc='Generate episodes for random policy'):
            episode = env.generate_episode(Q, seed=i)
            randomQ_rewards.append(env.tot_reward)
            
        for i in tqdm(range(k), desc='Generate episodes for learned policy'):
            episode = env.generate_episode(best_Q,seed=i)
            bestQ_rewards.append(env.tot_reward)
        
        print(f'Pour les mêmes lancer de dès:')
        print(f'Reward moyen politique random: {np.mean(randomQ_rewards)}')
        print(f'Reward moyen best politique: {np.mean(bestQ_rewards)}')
        