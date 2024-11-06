from random import choice

import numpy as np
from tqdm import tqdm

from yams_env import YamsEnvAlternative, YamsEnvNoe


def first_visit_MC_prediction(env, gamma:float=0.9, epsilon:float=0.2, max_iter:int=100, seed:int=None):
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
        episode = env.generate_episode(Q, epsilon=epsilon)
        for i, (s, a, r, next_s) in enumerate(reversed(episode)):
            G = r + sum(map(lambda x: x[2], episode[:i]))
            Returns[(s, a)].append(G)
            Q[(s, a)] = float(np.mean(Returns[(s,a)]))
    return Q

def SARSA(env, gamma:float=1.0, alpha=1.0, max_iter:int=100, epsilon:float=.5, seed:int=None):
    if seed is not None:
        np.random.seed(seed)
    Q = {}
    for s in tqdm(env.states, desc='Initialisation'):
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

def Qlearning(env:YamsEnvNoe, gamma:float=1.0, alpha=1.0, max_iter:int=100, epsilon:float=.5, seed:int=None):
    if seed is not None:
        np.random.seed(seed)
    Q = {}
    for s in tqdm(env.states, desc='Initialisation'):
        all_actions = env.list_actions(s)
        for a in all_actions:
            if env.is_terminal_state(s):
                Q[(s, a)] = 0
            else: 
                Q[(s,a)] = np.random.random()
    for i in tqdm(range(max_iter)):
        env.reset()
        while not env.is_done():
            s = env.scored
            a, r = env.choose_epsilon_action(Q, epsilon)
            env.next_state(a)
            next_s = env.scored
            Q[(s, a)] = Q[(s, a)] + alpha * (r + gamma*max([Q[(next_s, next_a)] for next_a in env.list_actions(next_s)]) - Q[(s, a)])
            s = next_s
    return Q

if __name__ == '__main__':
    from figures import Brelan, Chance, Multiple, Number, Suite

    env1 = YamsEnvNoe(3, 3, [Number(0), Number(1), Number(2), Multiple(3, 7), Multiple(3, 56)])
    #env2 = YamsEnvAlternative(3, 3, [Number(0), Number(1), Number(2), Multiple(3, 7), Multiple(3, 56)])
    #env = YamsEnv(4, 5, [Multiple(4, 20), Brelan(), Suite(1, 4, 20), Suite(2, 5, 20), Number(0), Number(1), Number(2), Number(3), Number(4)])
    #env = YamsEnv(3, 3, [Number(0), Number(1), Number(2), Multiple(3, 7)])
    #env = YamsEnv(4, 5, [Multiple(4, 20), Brelan(), Suite(1, 4, 20), Suite(2, 5, 20), Number(0), Number(1), Number(2), Number(3), Number(4)])
    env1 = YamsEnvNoe(5, 6, [Number(0), Number(1), Number(2), Number(3), Number(4), Number(5), Brelan(), Multiple(4, 30), Multiple(5, 50)])
    #env2 = YamsEnvAlternative(5, 6, [Number(0), Number(1), Number(2), Number(3), Number(4), Number(5), Brelan(), Multiple(4, 30), Multiple(5, 50)])
    
    
    
    for j in range(10):

        #best_Qenv1 = first_visit_MC_prediction(env1, max_iter=10000, seed=0, epsilon=j/100)
        #best_Qenv2 = first_visit_MC_prediction(env2, max_iter=10000, seed=j+4, epsilon=j/100)
        best_Qenv1 = Qlearning(env1, max_iter=10000, seed=0, epsilon=0.5)
        randomQ_rewards = []
        bestQenv1_rewards = []
        bestQenv2_rewards = []
        np.random.seed(j+4)
        Q = {}
        for s in env1.states:
            all_actions = env1.list_actions(s)
            for a in all_actions:
                Q[(s,a)] = np.random.random() # Random policy
        k = 1000
        print(f'Rank{j}: starts generating episodes')
        
        # Test with a random policy
        for i in tqdm(range(k), desc='Random policy'):
            episode = env1.generate_episode(Q, seed=i, epsilon=1)
            randomQ_rewards.append(env1.tot_reward)
        for i in tqdm(range(k), desc='Best policy Noe'):
            episode = env1.generate_episode(best_Qenv1,seed=i)
            bestQenv1_rewards.append(env1.tot_reward)
        #for i in tqdm(range(k), desc='Best policy Berar'):
        #    episode = env2.generate_episode(best_Qenv2, seed=i)
        #    bestQenv2_rewards.append(env2.tot_reward)
        
        
        print(f'Pour les mêmes lancer de dès:')
        print(f'Reward moyen politique random: {np.mean(randomQ_rewards)}')
        print(f'Reward moyen best politique (Noe): {np.mean(bestQenv1_rewards)}')
        #print(f'Reward moyen best politique (Berar): {np.mean(bestQenv2_rewards)}')
        