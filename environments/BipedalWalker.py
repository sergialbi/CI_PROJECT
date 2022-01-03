import gym


class BipedalWalker:

    def __init__(self, render=False):
        self.env = gym.make('BipedalWalker-v3')
        self.render = render


    def start(self):
        return self.env.reset()

    
    def step(self, action):
        next_state, reward, terminal, _ = self.env.step(action)
        return next_state, reward, terminal


    def get_state_shape(self):
        return self.env.observation_space.shape

    
    def get_action_space(self):
        return self.env.action_space


    def end(self):
        self.env.close()