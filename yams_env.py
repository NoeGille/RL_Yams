import numpy as np
import itertools
from figures import Figure, Dice
from turn_env import TurnEnvironment

class YamsEnv:
    """Class to represent a Yams Game.
    score_sheet: ndarray of size (nb_dice, coombinaison)"""

    def __init__(self, nb_dice: int, nb_face: int, figures: dict[Figure]):
        self.nb_face = nb_face
        self.nb_dice = nb_dice
        self.figures = figures
        self.dices = [Dice(nb_face) for _ in range(nb_dice)]
        self.states = self.get_states()
        

    def get_states(self):
        """Return a list of all possibles states"""
        states = []
        for it in itertools.product(range(2), repeat=len(self.figures)):
            states.append(it)
        
    def next_state(self, s, a):
        """Return the next state and its associated reward given a state s and an action a."""
        reward = self.figures[a].compute_value(self.dices)
        #new_s = tuple(s[i] if i != a else 1 for i in range(len(self.figures)))
        new_s = list(s)
        new_s[a] = 1
        new_s = tuple(new_s)
        return new_s, reward
    
    def get_actions(self, dices):
        """Return a list of all possible actions from state s."""
        actions = []
        for i, figure in enumerate(self.figures):
            if figure.is_valid(dices) and not self.s[i]:
                future_reward = figure.compute_value(dices)
                actions.append((i, future_reward))
        if not actions: # Empty list
            for i in range(len(self.figures)):
                if not self.s[i]:
                    actions.append((i, 0))
        return actions
    
    def choose_action(self):
        MyTurn = TurnEnvironment(self.nb_dice,self.nb_face)
        
        reward_table = np.zeros((len(MyTurn.S),5))
        for i, s in enumerate(MyTurn.S):
            Aa = self.get_actions(s)
            for j, a in enumerate(Aa) :
                reward_table[i,j]= Aa[a][1]
        
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
        print(s2)
        
        Aa = self.get_actions(s2)        
        print(Aa)
        
        # this a greedy policy! to adapt
        if len(list(Aa)) == 0:
            return [],0
            
        i=0
        action = list(Aa)[i]
        for a in Aa:
            if Aa[a] > Aa[action] :
                action = list(Aa)[i]
            i=i+1
        
        return action, Aa[action]


if __name__ == '__main__':
    env = YamsEnv(5, 6, {'Brelan': 25, 'Full': 25, 'Number': 25})
    