from multiprocessing import Process, Pipe
import numpy as np


def environment_worker_function(env_function, pipe_end, **env_params):
    environment = env_function(**env_params)

    state = environment.start()

    while True:
        (msg, data) = pipe_end.recv()

        if msg == "start":
            pipe_end.send(("start", state))

        elif msg == "step":
            action = data
            reward, next_state, terminal = environment.step(action)
            if terminal:
                next_state = environment.start()
            pipe_end.send(("step", (reward, next_state, terminal)))

        elif msg == "end":
            pipe_end.close()
            environment.end()
            break

        elif msg == "get_state_shape":
            state_shape = environment.get_state_shape()
            pipe_end.send(("state_shape", state_shape))
        
        elif msg == "get_action_space":
            action_space = environment.get_action_space()
            pipe_end.send(("action_space", action_space))

        else:
            raise ValueError(msg)


class MultiEnvironmentManager:

    def __init__(self, env_function, num_envs, **env_params):
        self.num_envs = num_envs
        self.pipes_main, self.pipes_subprocess = zip(*[Pipe() for _ in range(num_envs)])
        self.envs = [Process(target = environment_worker_function, args = (env_function, self.pipes_subprocess[i]), 
            kwargs = env_params) for i in range(self.num_envs)]

        self.__initialize_subprocesses()
        self.__configure_state_space()
        self.__configure_action_space()


    def __initialize_subprocesses(self):
        for env in self.envs:
            env.daemon = True
            env.start()


    def __configure_state_space(self):
        self.pipes_main[0].send(("get_state_shape", None))
        (_, self.state_shape) = self.pipes_main[0].recv()


    def __configure_action_space(self):
        self.pipes_main[0].send(("get_action_space", None))
        (_, self.action_space) = self.pipes_main[0].recv()


    def start(self):
        states = np.zeros((self.num_envs, *self.state_shape))
        for pipe_main in self.pipes_main:
            pipe_main.send(("start", None))

        for i in range(self.num_envs):
            (_, state) = self.pipes_main[i].recv()
            states[i] = state

        return states


    def end(self):
        for pipe_main in self.pipes_main:
            pipe_main.send(("end", None))

        for env_process in self.envs:
            env_process.join()


    def step(self, actions):
        for pipe_main, action in zip(self.pipes_main, actions):
            pipe_main.send(("step", action))

        rewards = np.zeros((self.num_envs))
        terminals = np.zeros((self.num_envs), dtype = bool)
        next_states = np.zeros((self.num_envs, *self.state_shape))

        for i in range(len(self.pipes_main)):
            (_, data) = self.pipes_main[i].recv()
            (reward, next_state, terminal) = data
            rewards[i] = reward
            terminals[i] = terminal
            next_states[i] = next_state

        return rewards, next_states, terminals


    def get_state_shape(self):
        return self.state_shape


    def get_action_space(self):
        return self.action_space