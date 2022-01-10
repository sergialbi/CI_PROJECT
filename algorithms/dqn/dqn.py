import os
import time
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
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
from environments.BipedalWalker import BipedalWalker
from environments.Breakout import Breakout


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
                        T.Resize(FRAME_SIZE[0], interpolation=Image.CUBIC),
                        T.ToTensor()])
def get_screen_batch(env, device=DEVICE):
    # Get environment screen and transpose it into PyTorch order (CHW = Channels, Height, Width).
    screen = env.env.render(mode='rgb_array').transpose((2, 0, 1))

    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.0
    screen = torch.from_numpy(screen)

    # Resize and add a batch dimension (BCHW)
    return resize_func(screen).unsqueeze(0)


def plot_screen_batch(screen, title=""):
    img = screen[0].cpu().detach().numpy().transpose(1,2,0)
    plt.imshow(img)
    plt.title(title)
    plt.show()


def create_dqn(env, memory_size=MEMORY_SIZE, device=DEVICE):
    # Extract screen dimensions (after resizing) and number of actions from environment
    screen_shape = get_screen_batch(env)[0].detach().cpu().numpy().transpose((1, 2, 0)).shape
    screen_height, screen_width = screen_shape[0], screen_shape[1]
    #plot_screen_batch(get_screen_batch(env), "Example of input")

    # Set number of actions using the environment
    n_actions = env.get_action_space().n
    if isinstance(env, Breakout):
        n_actions -= 2      # Discart NOOP and FIRE action

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
    
    epochs_rewards = np.zeros(num_epochs)
    for epoch in range(num_epochs):        
        # Initialize the environment and state
        steps_done = 0
        env.start()
        last_screen = get_screen_batch(env)
        current_screen = get_screen_batch(env)        
        state = current_screen - last_screen
        epoch_reward = 0
        done = False
        while not done:
            # Select and perform an action
            action = select_action(state, policy_net)            
            _, reward, done = env.step(action.item())
            steps_done += 1
            epoch_reward += reward
            reward = torch.tensor([reward])

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
        epochs_rewards[epoch] = epoch_reward
            
        # Update the target network, copying all weights and biases in DQN
        if epoch % epochs_per_target_upd == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Print epoch information
        print(f"Epoch {epoch+1}/{num_epochs} | {steps_done} steps done and a reward of {epoch_reward}")

    return epochs_rewards


def select_action(state, policy_net, device=DEVICE):
    with torch.no_grad():
        state_at_device = state.to(device)
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        return policy_net(state_at_device).max(1)[1].view(1, 1)


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
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward).to(device)

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
    non_final_next_states = non_final_next_states.to(device)
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


# ----------- RESULTS ----------- #
def store_rewards(rewards, game_name):
    # Get folder
    folder = ""
    if game_name == WALKER:
        folder = PATH_RESULTS_DQN_WALKER
    elif game_name == BREAKOUT:
        folder = PATH_RESULTS_DQN_BREAKOUT
    
    # Create folder if it does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Get filename and path
    filename = f"rewards_{game_name}_{time.strftime('%Y%m%d-%H%M%S')}.npy"
    path = os.path.join(folder, filename)

    # Store
    np.save(path, rewards)

    return path


def plot_rewards(rewards, title):
    plt.plot(rewards)
    plt.title(f"Rewards for {title}")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.show()


def plot_rewards_from_file(filepath):
    filename = os.path.basename(filepath)
    results = np.load(filepath)
    plot_rewards(results, filename)


# ----------- RUNNING METHOD ----------- #
def run_dqn(game_name):
    # Get game environment
    if game_name == WALKER:
        env = BipedalWalker(reward_scale=1)
    elif game_name == BREAKOUT:
        env = Breakout(1, FRAME_SIZE, reward_scale=1)
    else:
        raise Exception(f"Game name {game_name} not recognized")

    # Perform procedure
    print(f"Selected device = {DEVICE}")

    print("Creating model...")
    dqn_tuple = create_dqn(env)
    
    print("Training...")
    rewards = train_dqn(dqn_tuple, env)
    
    print("Storing rewards...")
    store_rewards(rewards, game_name)
    
    print("Plotting rewards...")
    plot_rewards(rewards, game_name)


    # Close environment
    env.end()

    return rewards