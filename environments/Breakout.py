import gym
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
import numpy as np


class Breakout:

    def __init__(self, num_stacked, frame_resize, render=False):
        self.env = gym.make('Breakout-v0')
        self.num_stacked = num_stacked
        self.frame_resize = frame_resize
        self.render = render
        self.frame_stack = deque([], maxlen=self.num_stacked)


    def __preprocess_frame(self, frame):
        frame = rgb2gray(frame)
        frame = resize(frame, self.frame_resize)
        return frame/255.0


    def start(self):
        frame = self.env.reset()
        frame = self.__preprocess_frame(frame)
        for _ in range(self.frame_stack.maxlen):
            self.frame_stack.append(frame)
        return np.stack(self.frame_stack, axis=2)
        

    def step(self, action):
        next_frame, reward, terminal, _ = self.env.step(action)
        next_frame = self.__preprocess_frame(next_frame)
        self.frame_stack.append(next_frame)
        return reward, np.stack(self.frame_stack, axis=2), terminal


    def get_state_shape(self):
        return (*self.frame_resize, self.num_stacked)

    
    def get_action_space(self):
        return self.env.action_space


    def end(self):
        self.env.close()