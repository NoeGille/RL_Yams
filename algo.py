from yams_env import YamsEnvAlternative as YamsEnv
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

def SARSA(env:YamsEnv, gamma:float=0.9, alpha=0.4, max_iter:int=100, epsilon:float=0.1, seed:int=None):
    if seed is not None:
        np.random.seed(seed)
    Q = {}
    for s in env.states:
        all_actions = env.list_actions(s)
        for a in all_actions:
            if env.is_terminal_state(s):
                Q[(s, a)] = 0
            else: 
                Q[(s,a)] = np.random.random()
    for i in tqdm(range(max_iter)):
        env.reset()
        s = env.scored
        a, _ = env.choose_epsilon_action(Q, epsilon)
        while not env.is_done():
            env.next_state(a)
            next_s = env.scored
            next_a, r = env.choose_epsilon_action(Q, epsilon)
            Q[(s, a)] = Q[(s, a)] + alpha * (r + gamma*Q[(next_s, next_a)] - Q[(s, a)])
            s = next_s
            a = next_a
    return Q


if __name__ == '__main__':
    from figures import Multiple, Chance, Number, Brelan, Suite
    env = YamsEnv(3, 3, [Number(0), Number(1), Number(2), Multiple(3, 7)])
    #env = YamsEnv(4, 5, [Multiple(4, 20), Brelan(), Suite(1, 4, 20), Suite(2, 5, 20), Number(0), Number(1), Number(2), Number(3), Number(4)])
    env = YamsEnv(5, 6, [Number(0), Number(1), Number(2), Number(3), Number(4), Number(5), Brelan(), Suite(1,5, 25), Suite(2, 6, 25), Multiple(4, 30), Multiple(5, 50)])
    for j in range(10):
        best_Q = first_visit_MC_prediction(env, max_iter=10000, seed=j)
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
        