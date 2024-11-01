import numpy as np
import itertools
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
        self.figures = figures
        self.states = self.get_states()
        self.RL_policy = RL_policy
        self.scored = [False for _ in range(len(figures))]
        self.tot_reward = 0

    def get_states(self):
        """Return a list of all possibles states"""
        states = []
        for it in itertools.product(range(2), repeat=len(self.figures)):
            states.append(it)
        return states
    
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
        MyTurn = TurnEnvironment(self.nb_dice,self.nb_face)
        
        reward_table = np.zeros((len(MyTurn.S),len(self.figures)))
        for i, s in enumerate(MyTurn.S):
            Aa = self.get_actions(s)
            for a, r in Aa :
                reward_table[i,a]= r

        v_3 = reward_table.max(axis=1)
        v_2,Q_2 = MyTurn.One_step_backward(v_3)
        v_1,Q_1 = MyTurn.One_step_backward(v_2)
        #######################################
        # First Roll
        s0 = MyTurn.get_state_from_action(np.zeros((self.nb_face),dtype='int'))       
        a0,_ = MyTurn.choose_best_action(s0,Q_1)
        ############################################"""
        # Second Roll
        s1 = MyTurn.get_state_from_action(a0)
        a1,_ = MyTurn.choose_best_action(s1,Q_2)
        #######
        #Third Roll
        s2 = MyTurn.get_state_from_action(a0)
        #print(s2)
        
        Aa = self.get_actions(s2)        
        #print(Aa)
        
        action, reward = self.RL_policy(self.scored, Aa)
        #print(reward)

        return action, reward

    def play_game(self):
        for i in range(len(self.figures)):
            action, reward = self.choose_action()
            assert not self.scored[action]
            self.scored[action] = True
            self.tot_reward += reward
        return self.tot_reward
    
    def reset(self):
        self.scored = [False for _ in range(len(self.figures))]
        self.tot_reward = 0


class YamsEnvNoe:
    """Class to represent a Yams Game.
    states are represent by tuple where each cell correspond to a unique associated Figure.
    If its value is False then is in not scored else if True it is scored.
    action is an int corresponding to the index of the cell to cross in a state."""

    def __init__(self, nb_dice: int, nb_face: int, figures: dict[Figure], RL_policy: callable = default_RL_policy):
        self.nb_face = nb_face
        self.nb_dice = nb_dice 
        self.figures = figures # List of figure object
        self.states = self.get_states() # List of all possible states
        self.RL_policy = RL_policy
        self.scored = tuple([False for _ in range(len(figures))]) # Current state in the game
        self.tot_reward = 0

    def get_states(self):
        """Return a list of all possibles states"""
        states = []
        for it in itertools.product(range(2), repeat=len(self.figures)):
            states.append(tuple(it))
        return states
    
    def list_actions(self, s):
        '''Return a list of all possible actions from state s regardles
        of dices.'''
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
    
    
    def generate_episode(self, Q, seed=None):
        '''Play the game once. Return a list of all tuple starting state, 
        actions, associated reward and ending state. Starts the game on empty
        score sheet state(init state).'''
        self.reset()
        if seed is not None:
            np.random.seed(seed)
        episode = []
        while not self.is_done():
            s = self.scored
            a,r = self.choose_best_action(Q) # Returns best action considering the policy corresponding to Q
            self.tot_reward += r
            self.next_state(a) # Go to next state following action a
            next_s = self.scored
            episode.append(tuple([s, a, r, next_s]))
        return episode
            
    
    def next_state(self, a):
        '''Action a'''
        assert not self.scored[a] # The cell is not already scored 
        list_scored = list(self.scored)
        list_scored[a] = True
        self.scored = tuple(list_scored) 
        
    def is_done(self):
        '''Is the current state a terminal state ?'''
        return np.all(self.scored)
        
    def choose_best_action(self, Q):
        """Return the best action and its associated reward given a state s."""
        dices = self.choose_turn_action()
        actions = self.get_actions(dices)
        if not actions:
            return None, 0
        best_action = max(actions, key=lambda x: Q[(self.scored, x[0])])
        return best_action
    
    def choose_turn_action(self):
        '''Play a Turn following the best policy. Returns s (dices) '''
        MyTurn = TurnEnvironment(self.nb_dice,self.nb_face)
        
        reward_table = np.zeros((len(MyTurn.S),len(self.figures)))
        for i, s in enumerate(MyTurn.S):
            Aa = self.get_actions(s)
            for a, r in Aa :
                reward_table[i,a]= r

        v_3 = reward_table.max(axis=1)
        v_2,Q_2 = MyTurn.One_step_backward(v_3)
        v_1,Q_1 = MyTurn.One_step_backward(v_2)
        #######################################
        # First Roll
        s0 = MyTurn.get_state_from_action(np.zeros((self.nb_face),dtype='int'))       
        a0,_ = MyTurn.choose_best_action(s0,Q_1)
        ############################################"""
        # Second Roll
        s1 = MyTurn.get_state_from_action(a0)
        a1,_ = MyTurn.choose_best_action(s1,Q_2)
        #######
        #Third Roll
        s2 = MyTurn.get_state_from_action(a0)
      
        return s2
    
    def reset(self):
        '''Reset the game.
        Current state becomes an empty sheet score'''
        self.scored = tuple([False for _ in range(len(self.figures))])
        self.tot_reward = 0


if __name__ == '__main__':
    
    def init_random_policy(env:YamsEnvNoe):
        Q = {}
        for s in env.states:
            avaible_actions = env.list_actions(s)
            for a in avaible_actions:
                Q[(s, a)] = 1 / len(avaible_actions)
        return Q
    from figures import Multiple, Chance, Number
    env = YamsEnvNoe(3, 3, [Multiple(3, 15), Multiple(2, 5), Number(0), Number(1), Number(2), Chance()])
    Q = init_random_policy(env)
    episode = env.generate_episode(Q)
    for s, a, r, next_s in episode:
        print(s, a, r, next_s)
    