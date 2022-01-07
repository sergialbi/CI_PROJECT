import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from constants.constants_general import *
from constants.constants_dqn import *


# ----------- MODEL AND MEMORY ----------- #
class DQN(nn.Module):
    def __init__(self, height, width, n_outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        linear_input_size = convw * convh * 32
        self.n_outputs = n_outputs
        self.head = nn.Linear(linear_input_size, n_outputs)

    def get_num_outputs(self):
        return self.n_outputs

    def forward(self, x):
        """Called with either one element to determine next action, or a batch during optimization. Returns tensor([[left0exp,right0exp]...])."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# ----------- CREATION ----------- #
resize_func = T.Compose([T.ToPILImage(),
                        T.Resize(64, interpolation=Image.CUBIC),
                        T.ToTensor()])
def get_screen_batch(env, device=DEVICE):
    # Get environment screen and transpose it into PyTorch order (CHW = Channels, Height, Width).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))

    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.0
    screen = torch.from_numpy(screen)

    # Resize and add a batch dimension (BCHW)
    return resize_func(screen).unsqueeze(0).to(device)


def create_dqn(env, memory_size=MEMORY_SIZE, device=DEVICE):
    # Extract screen dimensions (after resizing) and number of actions from environment
    screen_shape = get_screen_batch(env)[0].detach().cpu().numpy().transpose((1, 2, 0)).shape
    screen_height, screen_width = screen_shape[0], screen_shape[1]
    n_actions = env.action_space.n

    # Create networks
    policy_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())  # Create RMS optimizer
    memory = ReplayMemory(memory_size)  # Create replay memory for batches

    dqn_tuple = policy_net, target_net, optimizer, memory # Group all variable in a tuple
    return dqn_tuple


# ----------- TRAINING ----------- #
def train_dqn(dqn_tuple, env, num_epochs=NUM_EPOCHS, epochs_per_target_upd=EPOCHS_PER_TARGET_UPD, device=DEVICE):
    policy_net, target_net, optimizer, memory = dqn_tuple   # Extract variables from tuple
    policy_net.train()

    steps_done = 0
    epochs_rewards = []
    for epoch in range(num_epochs):        
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen_batch(env)
        current_screen = get_screen_batch(env)        
        state = current_screen - last_screen
        epoch_reward = 0
        done = False
        while not done:
            # Select and perform an action
            action = select_action(state, policy_net, steps_done)
            steps_done += 1
            _, reward, done, _ = env.step(action.item())
            epoch_reward += reward
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen_batch(env)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model(policy_net, target_net, memory, optimizer)

        # When epoch completed, store reward
        epochs_rewards.append(epoch_reward)
        #plot_rewards(epochs_rewards)    # TODO: Maybe remove this if Visual Studio keeps failing plotting

        # Store current model
        torch.save(policy_net.state_dict(), "dqn_policy_net.h5")
        torch.save(target_net.state_dict(), "dqn_target_net.h5")
            
        # Update the target network, copying all weights and biases in DQN
        if epoch % epochs_per_target_upd == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Print epoch information
        print(f"Epoch {epoch+1}/{num_epochs} | {steps_done} steps done and a reward of {epoch_reward}")

    #plt.show() # TODO: Remove this if plot_rewards if removed


    epochs_rewards = np.array(epochs_rewards)
    return epochs_rewards


def select_action(state, policy_net, steps_done, eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY, device=DEVICE):
    n_actions = policy_net.get_num_outputs()
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
        math.exp(-1. * steps_done / eps_decay)
    
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model(policy_net, target_net, memory, optimizer, device=DEVICE):
    if len(memory) < BATCH_SIZE:    # If memory does not contain enough elements for a batch, exist the method
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def plot_rewards(epochs_rewards):
    plt.figure(2)
    plt.clf()
    rewards_tensor = torch.tensor(epochs_rewards, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(rewards_tensor.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_tensor) >= 100:
        means = rewards_tensor.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


# ----------- RUNNING METHOD ----------- #
def run_dqn(game_name):
    env = gym.make(game_name).unwrapped # Unwrap for being environment agnostic

    print("Creating model...")
    dqn_tuple = create_dqn(env)
    print("Training...")
    results = train_dqn(dqn_tuple, env)
    print(f"Results:\n{results}")
    # TODO: Store results

    env.close()
    plt.ioff()

    return results