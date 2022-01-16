from multiprocessing import Process, Pipe
import numpy as np


class Environment(Process):

    def __init__(self, env_index, pipe_end, env_function, **env_params):
        super().__init__()
        self.env = env_function(**env_params)
        self.env_index = env_index
        self.pipe_end = pipe_end

    def run(self):
        super().run()
        state = self.env.start()
        self.pipe_end.send(state)

        while True:
            action = self.pipe_end.recv()
            state, reward, done = self.env.step(action)

            if done:
                state = self.env.start()
            
            self.pipe_end.send([state, reward, done])


class MultiEnvironmentManager:

    def __init__(self, env_function, num_envs, **env_params):
        self.num_envs = num_envs
        self.pipes_main = []
        self.pipes_subprocess = []
        self.envs = []

        for i in range(num_envs):
            pipe_main, pipe_subprocess = Pipe()
            env = Environment(i, pipe_subprocess, env_function, **env_params)
            env.start()
            
            self.pipes_main.append(pipe_main)
            self.pipes_subprocess.append(pipe_subprocess)
            self.envs.append(env)

            self.configure_spaces(env_function, **env_params) 

    def configure_spaces(self, env_function, **env_params):
        temp_env = env_function(**env_params)
        self.state_shape = temp_env.get_state_shape()
        self.action_space = temp_env.get_action_space()
        self.max_steps = temp_env.env._max_episode_steps
        temp_env.end()

    def end(self):
        for env in self.envs:
            env.terminate()
            env.join()

    def start(self):
        states = np.zeros((self.num_envs, *self.state_shape))
        for i in range(len(self.pipes_main)):
            states[i] = self.pipes_main[i].recv()
        return states

    def step(self, actions):
        for pipe_main, action in zip(self.pipes_main, actions):
            pipe_main.send(action)

        next_states = np.zeros((self.num_envs, *self.state_shape))
        rewards = np.zeros((self.num_envs))
        terminals = np.zeros((self.num_envs), dtype = bool)

        for i in range(len(self.pipes_main)):
            [next_state, reward, terminal] = self.pipes_main[i].recv()
            next_states[i] = next_state
            rewards[i] = reward
            terminals[i] = terminal

        return next_states, rewards, terminals

    def get_state_shape(self):
        return self.state_shape

    def get_action_space(self):
        return self.action_space

    def get_max_steps(self):
        return self.max_steps