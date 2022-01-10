import gym
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
import random
import numpy as np
from constants.constants_general import BREAKOUT

class Breakout:

    def __init__(self, num_stacked, frame_resize, noop_actions=10, reward_scale=0.01, render=False):
        self.env = gym.make(BREAKOUT)
        self.render = render
        
        self.reward_scale = reward_scale
        
        self.num_stacked = num_stacked
        self.frame_resize = frame_resize
        self.frame_stack = deque([], maxlen=self.num_stacked)
        
        self.terminal = True
        self.lives = 0

        # Number of No Operations to perform (i.e. let the game transition without moving the bar)
        self.noop_actions = noop_actions

        # Special buttons
        self.noop_action = 0
        self.fire_action = 1


    def __preprocess_frame(self, frame):
        #frame = frame[30:, 6:154]
        frame = rgb2gray(frame)
        frame = resize(frame, self.frame_resize)
        return frame/255.0

    
    def __reset(self):
        if self.terminal:
            # terminal will be true when all lives are exhausted
            self.env.reset()
        # Otherwise, simply a live is used, and the game continues
        # Push the FIRE button to start the game
        frame, _, self.terminal, _ = self.env.step(self.fire_action)
        self.lives = self.env.unwrapped.ale.lives()
        return frame


    def __perform_noop_actions(self):
        num_noop_actions = random.randint(1, self.noop_actions)
        
        if self.render: 
            self.env.render()

        for _ in range(num_noop_actions):
            frame, _, self.terminal, _ = self.env.step(self.noop_action)
            lives = self.env.unwrapped.ale.lives()
            if self.terminal or lives < self.lives:
                frame = self.__reset()

        return frame


    def start(self):
        frame = self.__reset()

        # Perform some transitions at the start of the game (i.e. each time the lives are exhausted) to generate 
        # variability in the episodes
        frame = self.__perform_noop_actions()

        frame = self.__preprocess_frame(frame)
        for _ in range(self.frame_stack.maxlen):
            self.frame_stack.append(frame)
            
        return np.stack(self.frame_stack, axis=2)
        

    def step(self, action):
        # Actions will be either 0 or 1. Add 2 to become RIGHT and LEFT
        action = action + 2

        if self.render: 
            self.env.render()

        next_frame, reward, self.terminal, _ = self.env.step(action)

        lives = self.env.unwrapped.ale.lives()
        if not self.terminal and lives < self.lives:
            next_frame = self.__reset()
        
        next_frame = self.__preprocess_frame(next_frame)
        self.frame_stack.append(next_frame)
        return np.stack(self.frame_stack, axis=2), reward*self.reward_scale, self.terminal


    def random_action(self):
        return self.env.action_space.sample()


    def get_state_shape(self):
        return (*self.frame_resize, self.num_stacked)

    
    def get_action_space(self):
        return self.env.action_space


    def end(self):
        self.env.close()