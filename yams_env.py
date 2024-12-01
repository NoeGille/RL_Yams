import itertools
from random import choice

import numpy as np
from tqdm import tqdm

from figures import Figure
from turn_env import TurnEnvironment

def default_RL_policy(state, actions):
    """Default policy, greedy"""
    return max(actions, key=lambda x: x[1])

def random_policy(state, actions):
    '''random policy'''
    return choice(actions)

class YamsEnvBinary:
    """Class to represent a Yams Game.
    score_sheet: ndarray of size (nb_dice, coombinaison)"""

    def __init__(self, nb_dice: int, nb_face: int, figures: dict[Figure], RL_policy: callable = default_RL_policy, saved_turnQs=None):
        self.nb_face = nb_face
        self.nb_dice = nb_dice
        self.MyTurn = TurnEnvironment(self.nb_dice,self.nb_face)
        self.figures = figures
        self.states = self.get_states()
        if saved_turnQs is not None:
            self.turnQs = np.load(saved_turnQs,allow_pickle=True).item()
        else:
            self.turnQs = self.get_turnQs() # List of all possible TurnEnv Q value for each YamsEnv state
        self.RL_policy = RL_policy
        self.scored = tuple([False for _ in range(len(figures))])
        self.tot_reward = 0

    
    def list_actions(self, state):
        '''Return a list of all possible actions from state s regardles
        of dices.'''
        if self.is_terminal_state(state):
            return [0]
        actions = []
        for i in range(len(state)):
            if not state[i]:
                actions.append(i)
        return actions
    
    def get_states(self):
        """Return a list of all possibles states"""
        states = []
        for it in itertools.product(range(2), repeat=len(self.figures)):
            states.append(tuple(it))
        return states
    
    def next_state(self, action, reward):
        """Update the state of the game"""
        assert not self.scored[action]
        self.scored = list(self.scored)
        self.scored[action] = True
        self.scored = tuple(self.scored)
        self.tot_reward += reward
    
    def get_turnQs(self):
        '''Return a list of all possible TurnEnv Q value for each YamsEnv state'''
        turnQs = {}
        for scored in tqdm(self.states):
            self.scored = scored
            reward_table = np.zeros((len(self.MyTurn.S),len(self.figures)))
            for i, s in enumerate(self.MyTurn.S):
                Aa = self.get_actions(s)
                for a, r in Aa :
                    reward_table[i,a]= r

            v_3 = reward_table.max(axis=1)
            v_2,Q_2 = self.MyTurn.One_step_backward(v_3)
            v_1,Q_1 = self.MyTurn.One_step_backward(v_2)
            turnQs[scored] = (Q_1, Q_2)
        return turnQs
    
    def get_actions(self, dices):
        """Return a list of all possible actions from state s."""
        actions = []
        for i, figure in enumerate(self.figures):
            if figure.is_valid(dices) and not self.scored[i]:
                future_reward = figure.compute_value(dices)
                actions.append((i, future_reward))
        if not actions: # Empty list
            for i in range(len(self.figures)):
                if not self.scored[i]:
                    actions.append((i, 0))
        if not actions: # State is terminal
            actions.append((0, 0))
        return actions
        
    def is_terminal_state(self, state):
        """Return True if the game is over."""
        return all(state)
    
    def is_done(self):
        """Return True if the game is over."""
        return all(self.scored)
    
    def choose_action(self):
        Q_1, Q_2 = self.turnQs[tuple(self.scored)]
        #######################################
        # First Roll
        s0 = self.MyTurn.get_state_from_action(np.zeros((self.nb_face),dtype='int'))       
        a0,_ = self.MyTurn.choose_best_action(s0,Q_1)
        ############################################"""
        # Second Roll
        s1 = self.MyTurn.get_state_from_action(a0)
        a1,_ = self.MyTurn.choose_best_action(s1,Q_2)
        #######
        #Third Roll
        s2 = self.MyTurn.get_state_from_action(a0)

        Aa = self.get_actions(s2)        

        action, reward = self.RL_policy(self.scored, Aa)

        return action, int(reward)

    def play_game(self):
        history = []
        for i in range(len(self.figures)):
            action, reward = self.choose_action()
            history.append((tuple(self.scored), reward))
            assert not self.scored[action]
            self.scored[action] = True
            self.tot_reward += reward
        return self.tot_reward, history
    
    def reset(self):
        self.scored = tuple([False for _ in range(len(self.figures))])
        self.tot_reward = 0


class YamsEnvTotal:
    """Class to represent a Yams Game.
    score_sheet: ndarray of size (nb_dice, coombinaison)"""

    def __init__(self, nb_dice: int, nb_face: int, figures: dict[Figure], RL_policy: callable = default_RL_policy, saved_turnQs=None):
        self.nb_face = nb_face
        self.nb_dice = nb_dice
        self.MyTurn = TurnEnvironment(self.nb_dice,self.nb_face)
        self.figures = figures
        self.states = self.get_states()
        self.turnQs = {}
        '''if saved_turnQs is not None:
            self.turnQs = np.load(saved_turnQs,allow_pickle=True).item()
        else:
            self.turnQs = self.get_turnQs() # List of all possible TurnEnv Q value for each YamsEnv state
            np.save('turnQsreduce.npy', self.turnQs)'''
        self.RL_policy = RL_policy
        self.scored = tuple([False for _ in range(len(figures))] + [0])
        self.tot_reward = 0

    
    def list_actions(self, state):
        '''Return a list of all possible actions from state s regardles
        of dices.'''
        if self.is_terminal_state(state):
            return [(0, 0)]
        actions = []
        for i in range(len(state) - 1):
            if not state[i]:
                for r in self.figures[i].get_possible_values(self.nb_dice, self.nb_face):
                    actions.append((i, r))
        return actions
    
    def get_states(self):
        """Return a list of all possibles states"""
        states = []
        for it in itertools.product(range(2), repeat=len(self.figures)):
            max_score = 0
            for i, fig in enumerate(self.figures):
                if it[i]:
                    max_score += fig.get_max_values(self.nb_dice, self.nb_face)
            for score in range(max_score+1):
                binary_state = list(it)
                states.append(tuple(binary_state + [score]))  
        return states
    
    def next_state(self, action, reward):
        """Update the state of the game"""
        assert not self.scored[action[0]]
        self.scored = list(self.scored)
        self.scored[action[0]] = True
        self.scored[-1] += reward
        self.scored = tuple(self.scored)
        self.tot_reward += reward
    
    def get_turnQs(self):
        '''Return a list of all possible TurnEnv Q value for each YamsEnv state'''
        turnQs = {}
        for scored in tqdm(self.states):
            self.scored = scored
            reward_table = np.zeros((len(self.MyTurn.S),len(self.figures)))
            for i, s in enumerate(self.MyTurn.S):
                Aa = self.get_actions(s)
                for a, r in Aa :
                    reward_table[i,a[0]]= r

            v_3 = reward_table.max(axis=1)
            v_2,Q_2 = self.MyTurn.One_step_backward(v_3)
            v_1,Q_1 = self.MyTurn.One_step_backward(v_2)
            turnQs[scored] = (Q_1, Q_2)
        return turnQs
    
    
    
    def get_actions(self, dices):
        """Return a list of all possible actions from state s."""
        actions = []
        for i, figure in enumerate(self.figures):
            if figure.is_valid(dices) and not self.scored[i]:
                future_reward = figure.compute_value(dices)
                actions.append(((i, future_reward), future_reward))
        if not actions: # Empty list
            for i in range(len(self.figures)):
                if not self.scored[i]:
                    actions.append(((i, 0), 0))
        if not actions: # State is terminal
            actions.append(((0, 0), 0))
        return actions
        
    def is_terminal_state(self, state):
        """Return True if the game is over."""
        return all(state[:-1])
    
    def is_done(self):
        """Return True if the game is over."""
        return all(self.scored[:-1])
    
    def update_turnQs(self):
        reward_table = np.zeros((len(self.MyTurn.S),len(self.figures)))
        for i, s in enumerate(self.MyTurn.S):
            Aa = self.get_actions(s)
            for a, r in Aa :
                reward_table[i,a[0]]= r
        v_3 = reward_table.max(axis=1)
        v_2,Q_2 = self.MyTurn.One_step_backward(v_3)
        v_1,Q_1 = self.MyTurn.One_step_backward(v_2)
        self.turnQs[tuple(self.scored)] = (Q_1, Q_2)
        
    def choose_action(self):
        if not(tuple(self.scored) in self.turnQs):
            self.update_turnQs()
        Q_1, Q_2 = self.turnQs[tuple(self.scored)]    
        #######################################
        # First Roll
        s0 = self.MyTurn.get_state_from_action(np.zeros((self.nb_face),dtype='int'))       
        a0,_ = self.MyTurn.choose_best_action(s0,Q_1)
        ############################################"""
        # Second Roll
        s1 = self.MyTurn.get_state_from_action(a0)
        a1,_ = self.MyTurn.choose_best_action(s1,Q_2)
        #######
        #Third Roll
        s2 = self.MyTurn.get_state_from_action(a0)
        
        Aa = self.get_actions(s2)        

        action, reward = self.RL_policy(self.scored, Aa)

        return action, int(reward)
        
    def choose_action_from_Q(self, Q):
        reward_table = np.zeros((len(self.MyTurn.S),len(self.figures)))
        for i, s in enumerate(self.MyTurn.S):
            Aa = self.get_actions(s)
            for a, r in Aa :
                reward_table[i,a[0]]=Q[(self.scored, a)]
        v_3 = reward_table.max(axis=1)
        v_2,Q_2 = self.MyTurn.One_step_backward(v_3)
        v_1,Q_1 = self.MyTurn.One_step_backward(v_2)

        #######################################
        # First Roll
        s0 = self.MyTurn.get_state_from_action(np.zeros((self.nb_face),dtype='int'))       
        a0,_ = self.MyTurn.choose_best_action(s0,Q_1)
        ############################################"""
        # Second Roll
        s1 = self.MyTurn.get_state_from_action(a0)
        a1,_ = self.MyTurn.choose_best_action(s1,Q_2)
        #######
        #Third Roll
        s2 = self.MyTurn.get_state_from_action(a0)

        Aa = self.get_actions(s2)        

        action, reward = self.RL_policy(self.scored, Aa)

        return action, int(reward)
    
    
    def reset(self):
        self.scored = tuple([False for _ in range(len(self.figures))] + [0])
        self.tot_reward = 0