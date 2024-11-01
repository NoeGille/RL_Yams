from figures import Multiple, Chance, Number
from yams_env import YamsEnvPierre as YamsEnv
from tqdm import tqdm

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
    
    def learn(self, n_episodes, game:YamsEnv, l=0.9):
        self.state_value = {state:0 for state in game.states}
        self.state_count = {state:1 for state in game.states}
        self.game = game
        self.scores = []
        for _ in tqdm(range(n_episodes)):
            self.game.reset()
            self.history = []
            r_game = self.game.play_game()
            explored = {}
            tot_reward = 0
            for state, reward in self.history[::-1]:
                tot_reward *= l
                tot_reward += reward
                explored[state] = tot_reward
            for state, v in explored.items():
                self.state_value[state] += v
                self.state_count[state] += 1
            self.scores.append(r_game)
        return self.scores
    
if __name__== "__main__":
    from figures import *
    rl = first_visit_MC()
    env = YamsEnv(3, 3, [Multiple(3, 15), Multiple(2, 5), Number(0), Number(1), Number(2), Chance()], rl.choose_action)
    env2 = YamsEnv(4, 5, [Multiple(4, 20), Brelan(), Suite(1, 4, 20), Suite(2, 5, 20), Number(0), Number(1), Number(2), Number(3), Number(4)])
    env3 = YamsEnv(5, 6, [Multiple(5, 30), Multiple(4, 20), Brelan(), Suite(1, 5, 20), Suite(2, 6, 20), Number(0), Number(1), Number(2), Number(3), Number(4), Number(5), Chance()])
    scores = rl.learn(100, env2)
    import matplotlib.pyplot as plt
    plt.plot(scores)
    plt.show()