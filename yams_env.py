import numpy as np
import itertools
from figures import Figure
from turn_env import TurnEnvironment

def default_RL_policy(state, actions):
    """Default policy, greedy"""
    return max(actions, key=lambda x: x[1])

class YamsEnv:
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


if __name__ == '__main__':
    from figures import Multiple, Chance, Number
    env = YamsEnv(3, 3, [Multiple(3, 15), Multiple(2, 5), Number(0), Number(1), Number(2), Chance()])
    print(env.play_game())
    env.reset()
    print(env.play_game())
    