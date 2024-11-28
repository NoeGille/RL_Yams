from tqdm import tqdm

from figures import Chance, Multiple, Number
from yams_env import YamsEnvTotal
from yams_env import YamsEnvBinary as YamsEnv
from yams_env import default_RL_policy, random_policy
import numpy as np
from random import choice
from abc import ABC, abstractmethod

class RLAlgorithm:
    '''Abastract class for Reinforcement Learning algorithms.
    Contains implementation of epsilon-greedy policy.
    epsilon: float, probability of choosing a random action
    gamma: float, discount factor'''
    def __init__(self, epsilon, gamma):
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = None
    
    @abstractmethod
    def fit(self, env:YamsEnvTotal, max_iter:int=100, seed:int=None, test_episodes:list=[]):
        pass

    def choose_action(self, state, actions):
        if np.random.random() < self.epsilon:
            return choice(actions)
        else:
            return max(actions, key=lambda a: self.Q[(state, a[0])])

    def test(self, env:YamsEnvTotal, n=1, seed=None):
        epsilon = self.epsilon
        self.epsilon = 0
        rewards = []
        if seed is not None:
            np.random.seed(seed)
        for i in range(n):
            env.reset()
            while not env.is_done():
                #a, r = env.choose_action_from_Q(self.Q) # turn policy is not greedy during test
                a, r = env.choose_action()
                env.next_state(a, r)
            rewards.append(env.tot_reward)
        self.epsilon = epsilon
        return rewards
    

class FirstVisitMCPrediction(RLAlgorithm):
    '''First visit Monte Carlo prediction algorithm
    epilson: float, probability of choosing a random action
    gamma: float, discount factor'''

    def fit(self, env:YamsEnvTotal, max_iter:int=100, seed:int=None, test_episodes:list=[]):
        Returns = {}
        if seed is not None:
            np.random.seed(seed)
        Q = {}
        for s in tqdm(env.states):
            all_actions = env.list_actions(s)
            for a in all_actions:
                #print((s, a))
                Returns[(s,a)] = []
                Q[(s,a)] = 0
        self.Q = Q
        test = []
        for i in tqdm(range(max_iter)):
            if i in test_episodes:
                rewards = self.test(env, 1000)
                test.append(rewards)
            episode = []
            env.reset()
            while not env.is_done():
                s = env.scored
                a, r = env.choose_action()
                env.next_state(a, r)
                next_s = env.scored
                episode.append((s, a, r, next_s))
            G = 0
            for i, (s, a, r, next_s) in enumerate(reversed(episode)):
                G = r + self.gamma * G
                Returns[(s, a)].append(G)
                Q[(s, a)] = float(np.mean(Returns[(s,a)]))
            self.Q = Q
        return test

    

class SARSA(RLAlgorithm):
    '''SARSA algorithm
    epilson: float, probability of choosing a random action
    gamma: float, discount factor
    alpha: float, learning rate'''

    def __init__(self, epsilon, gamma, alpha):
        super().__init__(epsilon, gamma)
        self.alpha = alpha

    def fit(self, env:YamsEnvTotal, max_iter:int=100, seed:int=None, test_episodes:list=[]):
        if seed is not None:
            np.random.seed(seed)
        Q = {}
        for s in env.states:
            all_actions = env.list_actions(s)
            for a in all_actions:
                Q[(s,a)] = 0
        self.Q = Q
        test = []
        for i in tqdm(range(max_iter)):
            if i in test_episodes:
                rewards = self.test(env, 1000)
                test.append(rewards)
            env.reset()
            s = env.scored
            a, r = env.choose_action()
            while not env.is_done():
                
                env.next_state(a, r)
                next_s = env.scored
                next_a, next_r = env.choose_action()
                Q[(s, a)] = Q[(s, a)] + self.alpha * (r + self.gamma * Q[(next_s, next_a)] - Q[(s, a)])
                s, a, r = next_s, next_a, next_r
            self.Q = Q
        return test


class QLearning(RLAlgorithm):
    '''Q-learning algorithm
    epilson: float, probability of choosing a random action
    gamma: float, discount factor
    alpha: float, learning rate'''

    def __init__(self, epsilon, gamma, alpha):
        super().__init__(epsilon, gamma)
        self.alpha = alpha

    def fit(self, env:YamsEnvTotal, max_iter:int=100, seed:int=None, test_episodes:list=[]):
        if seed is not None:
            np.random.seed(seed)
        Q = {}
        for s in env.states:
            all_actions = env.list_actions(s)
            for a in all_actions:
                Q[(s,a)] = 0
        self.Q = Q
        test = []
        for i in tqdm(range(max_iter)):
            if i in test_episodes:
                rewards = self.test(env, 1000)
                test.append(rewards)
            env.reset()
            s = env.scored
            a, r = env.choose_action()
            while not env.is_done():
                env.next_state(a, r)
                next_s = env.scored
                next_a, next_r = env.choose_action()
                Q[(s, a)] = Q[(s, a)] + self.alpha * (r + self.gamma * max(Q[(next_s, a)] for a in env.list_actions(next_s)) - Q[(s, a)])
                s, a, r = next_s, next_a, next_r
            self.Q = Q
        return test

class first_visit_MC():
    def choose_action(self, state, actions):
        best_action = (0, -1)
        vmax = -1
        for action, reward in actions:
            new_state = state.copy()
            new_state[action] = True
            new_state = tuple(new_state)
            v = reward + self.state_value[new_state]/self.state_count[new_state]
            if v > vmax:
                best_action = (action, reward)
                vmax = v
        return best_action
    
    def learn(self, n_episodes, game:YamsEnv, gamma=0.9):
        self.state_value = {state:100 for state in game.states}
        self.state_count = {state:1 for state in game.states}
        self.game = game
        self.scores = []
        for _ in tqdm(range(n_episodes), desc='Learning'):
            self.game.reset()
            r_game, history = self.game.play_game()
            explored = {}
            tot_reward = 0
            for state, reward in history[::-1]:
                tot_reward = tot_reward * gamma + reward
                explored[state] = tot_reward
            for state, v in explored.items():
                self.state_value[state] += v
                self.state_count[state] += 1
            self.scores.append(r_game)
        for state in self.state_value:
            self.state_value[state] /= self.state_count[state]
        self.state_value[tuple([True]*len(self.game.figures))] = 0
        for i in range(len(self.game.figures)):
            # Compute average value of action i
            v = 0
            for state in self.state_value:
                if state[i]:
                    v -= self.state_value[state]
                else:
                    v += self.state_value[state]
        return self.scores
    
    def test(self, game:YamsEnv):
        self.game = game
        self.game.reset()
        r_game, history = self.game.play_game()
        return r_game


if __name__== "__main__":
    from figures import *
    import matplotlib.pyplot as plt
    rlalgo1 = FirstVisitMCPrediction(epsilon=1.0, gamma=1.0)
    rlalgo2 = SARSA(epsilon=1.0, gamma=1.0, alpha=0.2)
    rlalgo3 = QLearning(epsilon=1.0, gamma=1.0, alpha=0.2)

    env = YamsEnvTotal(3, 3, [Multiple(3, 15), Multiple(2, 5), Number(0), Number(1), Number(2), Chance()], rlalgo1.choose_action)
    #env = YamsEnv(4, 5, [Multiple(4, 20), Brelan(), Suite(1, 4, 20), Suite(2, 5, 20), Number(0), Number(1), Number(2), Number(3), Number(4)], rl.choose_action)
    #env = YamsEnv(3, 3, [Number(0), Number(1), Number(2), Multiple(3, 7), Multiple(3, 56)])
    #env = YamsEnvTotal(5, 6, [Number(0), Number(1), Number(2), Number(3), Number(4), Number(5), Brelan(), Multiple(4, 30), Multiple(5, 50), Chance()], rl.choose_action) #saved_turnQs='turnQsF.npy')
    #env = YamsEnv(5, 6, [Multiple(5, 50), Multiple(4, 20), Brelan(), Suite(1, 5, 30), Suite(2, 6, 30), Number(0), Number(1), Number(2), Number(3), Number(4), Number(5), Chance(), Full()])
    test_episodes = [0, 1, 10, 100, 1000, 2000, 5000, 8000, 10000, 11999, 30000, 49999, 99999]
    max_iter = 100000
    test = rlalgo1.fit( env, max_iter, 1, test_episodes=test_episodes)
    plt.figure(figsize=(5, 10))
    plt.plot(test_episodes, np.mean(test, axis=1), label='First visit MC')
    #plt.fill_between(test_episodes, np.mean(test, axis=1)-np.std(test, axis=1), np.mean(test, axis=1)+np.std(test, axis=1), alpha=0.5)
    scores = rlalgo1.test(env, 10000)
    print(f'Moyenne des rewards', np.mean(scores))
    
    env.RL_policy = rlalgo2.choose_action
    test = rlalgo2.fit( env, max_iter, 1, test_episodes=test_episodes)
    plt.plot(test_episodes, np.mean(test, axis=1), label='SARSA')
    #plt.fill_between(test_episodes, np.mean(test, axis=1)-np.std(test, axis=1), np.mean(test, axis=1)+np.std(test, axis=1), alpha=0.5)
    scores = rlalgo2.test(env, 10000)

    env.RL_policy = rlalgo3.choose_action
    test = rlalgo3.fit( env, max_iter, 1, test_episodes=test_episodes)
    plt.plot(test_episodes, np.mean(test, axis=1), label='Q-learning')
    #plt.fill_between(test_episodes, np.mean(test, axis=1)-np.std(test, axis=1), np.mean(test, axis=1)+np.std(test, axis=1), alpha=0.5)
    scores = rlalgo3.test(env, 10000)



    #plt.plot(scores)
    print(f'Moyenne des rewards', np.mean(scores))
    #plt.plot(np.polyval(np.polyfit(range(len(scores)), scores, 1), range(len(scores))))
    #plt.show()

    env = YamsEnvTotal(3, 3, [Multiple(3, 15), Multiple(2, 5), Number(0), Number(1), Number(2), Chance()], default_RL_policy)
    
    def test(env:YamsEnvTotal, n=1):
        rewards = []
        for i in range(n):
            env.reset()
            while not env.is_done():
                s = env.scored
                a, r = env.choose_action()
                env.next_state(a, r)
            rewards.append(env.tot_reward)
        return rewards

    scores = test(env, 10000)
    plt.hlines(np.mean(scores), 0, max_iter, color='red', linestyles='dashed', label='Greedy policy')
    #plt.fill_between(range(10000), np.mean(scores)-np.std(scores), np.mean(scores)+np.std(scores), alpha=0.5, color='red')

    print(f'Moyenne des rewards', np.mean(scores))

    env = YamsEnvTotal(3, 3, [Multiple(3, 15), Multiple(2, 5), Number(0), Number(1), Number(2), Chance()], random_policy)
    scores = test(env, 10000)
    print(f'Moyenne des rewards', np.mean(scores))
    plt.hlines(np.mean(scores), 0, max_iter, color='green', linestyles='dashed', label='Random policy')
    #plt.fill_between(range(10000), np.mean(scores)-np.std(scores), np.mean(scores)+np.std(scores), alpha=0.5, color='green')
    plt.legend()
    plt.show()
    