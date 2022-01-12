import gym
from collections import deque
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
from constants.constants_general import CARTPOLE

class CartPole:

    def __init__(self, reward_scale, num_stacked=None, frame_resize=None, render=False):
        self.env = gym.make(CARTPOLE)
        self.reward_scale = reward_scale
        
        self.num_stacked = num_stacked
        if self.num_stacked is not None:
            self.frame_stack = deque([], maxlen=self.num_stacked)
        self.frame_resize = frame_resize
        
        self.render = render
    

    def __get_current_frame(self):
        frame = self.env.render(mode="rgb_array")
        return self.__preprocess_frame(frame)


    def __preprocess_frame(self, frame):        
        frame = rgb2gray(frame)
        frame = resize(frame, self.frame_resize)
        return frame/255.0


    def start(self):
        if self.num_stacked is None:
            return self.env.reset()
        else:
            self.env.reset()
            frame = self.__get_current_frame()
            for _ in range(self.frame_stack.maxlen):
                self.frame_stack.append(frame)
            return np.stack(self.frame_stack, axis=2)

    
    def step(self, action):
        next_state, reward, terminal, _ = self.env.step(action)
        if self.num_stacked is None:
            return next_state, reward*self.reward_scale, terminal
        else:
            next_frame = self.__get_current_frame()
            self.frame_stack.append(next_frame)
            return np.stack(self.frame_stack, axis=2), reward*self.reward_scale, terminal


    def get_state_shape(self):
        return self.env.observation_space.shape

    
    def get_action_space(self):
        return self.env.action_space


    def end(self):
        self.env.close()