import itertools
from random import choice

import numpy as np
from tqdm import tqdm

from figures import Figure
from turn_env import TurnEnvironment


def default_RL_policy(state, actions):
    """Default policy, greedy"""
    return max(actions, key=lambda x: x[1])

class YamsEnvPierre:
    """Class to represent a Yams Game.
    score_sheet: ndarray of size (nb_dice, coombinaison)"""

    def __init__(self, nb_dice: int, nb_face: int, figures: dict[Figure], RL_policy: callable = default_RL_policy):
        self.nb_face = nb_face
        self.nb_dice = nb_dice
        self.MyTurn = TurnEnvironment(self.nb_dice,self.nb_face)
        self.figures = figures
        self.states = self.get_states()
        #self.turnQs = self.get_turnQs() # List of all possible TurnEnv Q value for each YamsEnv state
        #np.save('turnQsF.npy', self.turnQs)
        self.turnQs = np.load('turnQs6.npy',allow_pickle=True).item()
        self.RL_policy = RL_policy
        self.scored = [False for _ in range(len(figures))]
        self.tot_reward = 0

    def get_states(self):
        """Return a list of all possibles states"""
        states = []
        for it in itertools.product(range(2), repeat=len(self.figures)):
            states.append(tuple(it))
        return states
    
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
        return actions
    
    def choose_best_action(self, s):
        """Return the best action and its associated reward given a state s."""
        actions = self.get_actions(s)
        if not actions:
            return None, 0
        best_action = max(actions, key=lambda x: x[1])
        return best_action
    
    def choose_action(self):
        '''v_3 = np.zeros(len(self.MyTurn.S))
        for i, s in enumerate(self.MyTurn.S):
            Aa = self.get_actions(s)
            v_3[i] = self.RL_policy(self.scored, Aa)[1]
        
        v_2,Q_2 = self.MyTurn.One_step_backward(v_3)
        v_1,Q_1 = self.MyTurn.One_step_backward(v_2)'''
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
        #print(s2)
        
        Aa = self.get_actions(s2)        
        #print(Aa)
        
        action, reward = self.RL_policy(self.scored, Aa)
        #print(reward)

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
        self.scored = [False for _ in range(len(self.figures))]
        self.tot_reward = 0


class YamsEnvNoe:
    """Class to represent a Yams Game.
    states are represent by tuple where each cell correspond to a unique associated Figure.
    If its value is False then is in not scored else if True it is scored.
    action is an int corresponding to the index of the cell to cross in a state."""

    def __init__(self, nb_dice: int, nb_face: int, figures: dict[Figure]):
        self.nb_face = nb_face
        self.nb_dice = nb_dice 
        self.MyTurn = TurnEnvironment(self.nb_dice,self.nb_face)
        self.figures = figures # List of figure object
        self.states = self.get_states() # List of all possible states
        #self.turnQs = self.get_turnQs() # List of all possible TurnEnv Q value for each YamsEnv state
        #np.save('turnQs6.npy', self.turnQs)
        self.turnQs = np.load('turnQs6.npy',allow_pickle=True).item()
        self.scored = tuple([False for _ in range(len(figures))]) # Current state in the game
        self.tot_reward = 0

    def get_states(self):
        """Return a list of all possibles states"""
        states = []
        for it in itertools.product(range(2), repeat=len(self.figures)):
            states.append(tuple(it))
        return states
    
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

    def list_actions(self, s):
        '''Return a list of all possible actions from state s regardles
        of dices.'''
        if self.is_terminal_state(s):
            return [None]
        actions = []
        for i in range(len(s)):
            if not s[i]:
                actions.append(i)
        return actions
    
    
    def get_actions(self, dices):
        """Return a list of all possible actions and associated reward from state s depending on 
        given dices."""
        actions = []
        for i, figure in enumerate(self.figures):
            if figure.is_valid(dices) and not self.scored[i]:
                future_reward = figure.compute_value(dices)
                actions.append((i, future_reward))
        if not actions: # Empty list
            for i in range(len(self.figures)):
                if not self.scored[i]:
                    actions.append((i, 0))
        return actions
    
    
    def generate_episode(self, Q, epsilon=0.0, seed=None):
        '''Play the game once. Return a list of all tuple starting state, 
        actions, associated reward and ending state. Starts the game on empty
        score sheet state(init state).'''
        self.reset()
        if seed is not None:
            np.random.seed(seed)
        episode = []
        while not self.is_done():
            s = self.scored
            a, r = self.choose_epsilon_action(Q, epsilon) # Returns best action considering the policy corresponding to Q
            self.tot_reward += r
            self.next_state(a) # Go to next state following action a
            next_s = self.scored
            episode.append(tuple([s, a, r, next_s]))
        return episode
            
    
    def next_state(self, a):
        '''Change the current state to the next state according to given action a'''
        assert not self.scored[a] # The cell is not already scored 
        list_scored = list(self.scored)
        list_scored[a] = True
        self.scored = tuple(list_scored) 
        
    def is_done(self):
        '''Is the current state a terminal state ?'''
        return np.all(self.scored)
    
    def is_terminal_state(self,s):
        return np.all(s)
    
    def choose_epsilon_action(self, Q, epsilon):
        """Return the best action (with a probability of 1-epsilon else a random action) 
        and its associated reward given a state s."""
        dices = self.choose_turn_action()
        actions = self.get_actions(dices)
        if not actions:
            return None, 0
        if np.random.random() > epsilon:
            best_action = max(actions, key=lambda x: Q[(self.scored, x[0])])
            return best_action
        return choice(actions)
        
        
    def choose_best_action(self, Q):
        """Return the best action and its associated reward given a state s."""
        return self.choose_epsilon_action(Q, 0)
    
    def choose_turn_action(self):
        '''Play a Turn following the best policy. Returns s (dices) '''
        Q_1, Q_2 = self.turnQs[self.scored]
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
      
        return s2
    
    def reset(self):
        '''Reset the game.
        Current state becomes an empty sheet score'''
        self.scored = tuple([False for _ in range(len(self.figures))])
        self.tot_reward = 0


EMPTY = -1
class YamsEnvAlternative:
    """Class to represent a Yams Game.
    states are represent by tuple where each cell correspond to a unique associated Figure.
    If its value is F-1 then is in not scored else it is scored.
    action is a tuple(int, value) corresponding to the index of the cell to cross in a state and the reward associated."""
    
    def __init__(self, nb_dice: int, nb_face: int, figures: dict[Figure]):
        self.nb_face = nb_face
        self.nb_dice = nb_dice
        self.MyTurn = TurnEnvironment(self.nb_dice,self.nb_face)
        self.figures = figures # List of figure object
        self.states = self.get_states() # List of all possible states
        self.turnQs = self.get_turnQs() # List of all possible TurnEnv Q value for each YamsEnv state
        #self.turnQs = np.load('turnQs6.npy',allow_pickle=True).item()
        self.scored = tuple([EMPTY for _ in range(len(figures))]) # Current state in the game
        self.tot_reward = 0

    def get_states(self):
        """Return a list of all possibles states"""
        states = []
        values = list(map(lambda x: x.get_possible_values(self.nb_dice, self.nb_face), self.figures))
        for i in range(len(values)):
            values[i].append(EMPTY)
        for it in itertools.product(*values):
            states.append(tuple(it))
        return states
    
    def get_turnQs(self):
        '''Return a list of all possible TurnEnv Q value for each YamsEnv state'''
        turnQs = {}
        for it in itertools.product(range(2), repeat=len(self.figures)):
            binary_scored = np.array(it)
            self.scored = tuple(binary_scored - 1)
            binary_scored = tuple(binary_scored)
            reward_table = np.zeros((len(self.MyTurn.S),len(self.figures)))
            for i, s in enumerate(self.MyTurn.S):
                Aa = self.get_actions(s)
                for a, r in Aa :
                    reward_table[i,a[0]]= r

            v_3 = reward_table.max(axis=1)
            v_2,Q_2 = self.MyTurn.One_step_backward(v_3)
            v_1,Q_1 = self.MyTurn.One_step_backward(v_2)
            turnQs[binary_scored] = (Q_1, Q_2)
        return turnQs
    
    def list_actions(self, s):
        '''Return a list of all possible actions from state s regardles
        of dices.'''
        if self.is_terminal_state(s):
            return [None]
        actions = []
        for i in range(len(s)):
            if s[i] == EMPTY:
                for value in self.figures[i].get_possible_values(self.nb_dice, self.nb_face):
                    actions.append((i, int(value)))
        return actions
    
    
    def get_actions(self, dices):
        """Return a list of all possible actions and associated reward from state s depending on 
        given dices."""
        actions = []
        for i, figure in enumerate(self.figures):
            if figure.is_valid(dices) and self.scored[i] == EMPTY:
                future_reward = figure.compute_value(dices)
                actions.append(((i, int(future_reward)), future_reward))
        if not actions: # Empty list
            for i in range(len(self.figures)):
                if self.scored[i] == EMPTY:
                    actions.append(((i, 0), 0))
        return actions
    
    
    def generate_episode(self, Q, epsilon=0.0, seed=None):
        '''Play the game once. Return a list of all tuple starting state, 
        actions, associated reward and ending state. Starts the game on empty
        score sheet state(init state).'''
        self.reset()
        if seed is not None:
            np.random.seed(seed)
        episode = []
        while not self.is_done():
            s = self.scored
            a,r = self.choose_epsilon_action(Q, epsilon=epsilon) # Returns best action considering the policy corresponding to Q
            self.tot_reward += r
            self.next_state(a) # Go to next state following action a
            next_s = self.scored
            episode.append(tuple([s, a, r, next_s]))
        return episode
            
    
    def next_state(self, a):
        '''Change the current state to the next state according to given action a'''
        assert self.scored[a[0]] == EMPTY # The cell is not already scored 
        list_scored = list(self.scored)
        list_scored[a[0]] = a[1]
        self.scored = tuple(list_scored) 
        
    def is_done(self):
        '''Is the current state a terminal state ?'''
        return np.all(np.array(self.scored) != EMPTY)
    
    def is_terminal_state(self,s):
        return np.all(np.array(s) != EMPTY)
    
    def choose_epsilon_action(self, Q, epsilon):
        """Return the best action (with a probability of 1-epsilon else a random action) 
        and its associated reward given a state s."""
        dices = self.choose_turn_action()
        actions = self.get_actions(dices)
        if not actions:
            return None, 0
        if np.random.random() > epsilon:
            best_action = max(actions, key=lambda x: Q[(self.scored, x[0])])
            return best_action
        return choice(actions)
        
        
    def choose_best_action(self, Q):
        """Return the best action and its associated reward given a state s."""
        return self.choose_epsilon_action(Q, 0)
    
    def choose_turn_action(self):
        '''Play a Turn following the best policy. Returns s (dices) '''
        
        binary_scored = tuple(np.where(np.array(self.scored) == EMPTY, 0, 1))
        Q_1, Q_2 = self.turnQs[binary_scored]
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
      
        return s2
    
    def reset(self):
        '''Reset the game.
        Current state becomes an empty sheet score'''
        self.scored = tuple([EMPTY for _ in range(len(self.figures))]) # Current state in the game
        self.tot_reward = 0


if __name__ == '__main__':
    
    def init_random_policy(env:YamsEnvNoe):
        Q = {}
        for s in env.states:
            avaible_actions = env.list_actions(s)
            for a in avaible_actions:
                Q[(s, a)] = 1 / len(avaible_actions)
        return Q
    from figures import Chance, Multiple, Number
    env = YamsEnvAlternative(3, 3, [Multiple(3, 15), Multiple(2, 5), Number(0), Number(1), Number(2)])
    Q = init_random_policy(env)
    episode = env.generate_episode(Q)
    for s, a, r, next_s in episode:
        print(s, a, r, next_s)
    