from yams_env import YamsEnvNoe, YamsEnvAlternative
from random import choice
import numpy as np
from tqdm import tqdm
from mpi4py import MPI

COMM=MPI.COMM_WORLD
RANK=COMM.Get_rank()
SIZE=COMM.Get_size()
CTYPE = 'float64'

def first_visit_MC_prediction(env, gamma:float=0.9, max_iter:int=100, seed:int=None):
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



def SARSA(env, gamma:float=0.9, alpha=0.4, max_iter:int=100, epsilon:float=0.1, seed:int=None):
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
    for i in range(max_iter):
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
    env1 = YamsEnvNoe(3, 3, [Number(0), Number(1), Number(2), Multiple(3, 7), Multiple(3, 56)])
    env2 = YamsEnvAlternative(3, 3, [Number(0), Number(1), Number(2), Multiple(3, 7), Multiple(3, 56)])
    #env = YamsEnv(4, 5, [Multiple(4, 20), Brelan(), Suite(1, 4, 20), Suite(2, 5, 20), Number(0), Number(1), Number(2), Number(3), Number(4)])
    print(f'Rank{RANK}: starts training')
    best_Qenv1 = SARSA(env1, max_iter=1000, seed=RANK+4)
    best_Qenv2 = SARSA(env2, max_iter=1000, seed=RANK+4)
    randomQ_rewards = []
    bestQenv1_rewards = []
    bestQenv2_rewards = []
    np.random.seed(RANK+4)
    Q = {}
    for s in env1.states:
        all_actions = env1.list_actions(s)
        for a in all_actions:
            Q[(s,a)] = np.random.random() # Random policy
    k = 100
    print(f'Rank{RANK}: starts generating episodes')
    
    # Test with a random policy
    for i in range(k):
        episode = env1.generate_episode(Q, seed=i)
        randomQ_rewards.append(env1.tot_reward)
    for i in range(k):
        episode = env2.generate_episode(best_Qenv2, seed=i)
        bestQenv2_rewards.append(env2.tot_reward)
    for i in range(k):
        episode = env1.generate_episode(best_Qenv1,seed=i)
        bestQenv1_rewards.append(env1.tot_reward)
    
    print(f'Pour les mêmes lancer de dès:')
    print(f'Reward moyen politique random: {np.mean(randomQ_rewards)}')
    print(f'Reward moyen best politique(Noe): {np.mean(bestQenv1_rewards)}')
    print(f'Reward moyen best politique(Berar): {np.mean(bestQenv2_rewards)}')
        