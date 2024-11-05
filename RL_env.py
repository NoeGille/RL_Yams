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
        self.state_value = {state:100 for state in game.states}
        self.state_count = {state:1 for state in game.states}
        self.game = game
        self.scores = []
        for _ in tqdm(range(n_episodes)):
            self.game.reset()
            r_game, history = self.game.play_game()
            explored = {}
            tot_reward = 0
            for state, reward in history[::-1]:
                tot_reward *= l
                tot_reward += reward
                explored[state] = tot_reward
            for state, v in explored.items():
                self.state_value[state] += v
                self.state_count[state] += 1
            self.scores.append(r_game)
        for state in self.state_value:
            self.state_value[state] /= self.state_count[state]
        print(self.state_value)
        self.state_value[tuple([True]*len(self.game.figures))] = 0
        for i in range(len(self.game.figures)):
            # Compute average value of action i
            v = 0
            for state in self.state_value:
                if state[i]:
                    v -= self.state_value[state]
                else:
                    v += self.state_value[state]
            print(v/2**(len(self.game.figures)-1))
        return self.scores
    
if __name__== "__main__":
    from figures import *
    rl = first_visit_MC()
    #env = YamsEnv(3, 3, [Multiple(3, 15), Multiple(2, 5), Number(0), Number(1), Number(2), Chance()], rl.choose_action)
    env2 = YamsEnv(4, 5, [Multiple(4, 20), Brelan(), Suite(1, 4, 20), Suite(2, 5, 20), Number(0), Number(1), Number(2), Number(3), Number(4)], rl.choose_action)
    #env3 = YamsEnv(5, 6, [Multiple(5, 50), Multiple(4, 20), Brelan(), Suite(1, 5, 30), Suite(2, 6, 30), Number(0), Number(1), Number(2), Number(3), Number(4), Number(5), Chance(), Full()])
    scores = rl.learn(2000, env2, l=1)
    import matplotlib.pyplot as plt
    plt.plot(scores)
    plt.plot(np.polyval(np.polyfit(range(len(scores)), scores, 1), range(len(scores))))
    plt.show()