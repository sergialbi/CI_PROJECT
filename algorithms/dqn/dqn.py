from constants.constants_general import *
from constants.constants_dqn import *
from environments.BipedalWalker import BipedalWalker
from environments.Breakout import Breakout
from environments.CartPole import CartPole

import os
import time
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# ----------- DQN MODELs ----------- #
class DQN(nn.Module):
    def __init__(self, height, width, n_outputs):
        super(DQN, self).__init__()
        self.n_outputs = n_outputs

        # Create convolutional layers
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so here it is computed.
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(width, 8, 4), 4, 2)
        convh = conv2d_size_out(conv2d_size_out(height, 8, 4), 4, 2)
        linear_input_size = convw * convh * 32

        # Final hidden layer and output layer (fully connected)
        self.fc1 = nn.Linear(linear_input_size, 256)
        self.head = nn.Linear(256, self.n_outputs)

    def get_num_outputs(self):
        return self.n_outputs

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)   # Flatten except for batch
        x = F.relu(self.fc1(x))
        return self.head(x)



class DQN_Big(nn.Module):
    def __init__(self, height, width, n_outputs):
        super(DQN_Big, self).__init__()
        self.n_outputs = n_outputs
        
        # Create convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so here it is computed.        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(width, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(height, 8, 4), 4, 2), 3, 1)        
        linear_input_size = convw * convh * 64

        # Final hidden layer and output layer (fully connected)
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, self.n_outputs)

    def get_num_outputs(self):
        return self.n_outputs

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)   # Flatten except for batch
        x = F.relu(self.fc1(x))
        return self.head(x)


# Get selected model
selected_model = DQN
if SELECTED_MODEL == "Big":
    selected_model = DQN_Big


# Auxiliar function
def conv2d_size_out(size, kernel_size, stride):
    return (size - (kernel_size - 1) - 1) // stride  + 1


# ----------- REPLAY MEMORY ----------- #
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
def get_state_batch(env_state, frame_diff=FRAME_DIFF):
    array = env_state.transpose((2, 0, 1)).astype(np.float32)
    if frame_diff:
        array[:-1] -= array[1:] # Compute differences between frames
    return torch.from_numpy(array).unsqueeze(0)


def plot_state_batch(screen, title=""):
    print(screen.shape)
    img = screen[0].cpu().detach().numpy().transpose(1,2,0)[:, :, -1]
    plt.imshow(img)
    plt.title(title)
    plt.show()


def create_dqn(env, model=selected_model, frame_size=FRAME_SIZE, memory_size=MEMORY_SIZE, device=DEVICE):
    # Set number of actions using the environment
    n_actions = env.get_action_space().n
    #if isinstance(env, Breakout):
    #    n_actions -= 2      # Discart NOOP and FIRE action

    # Create networks
    policy_net = model(frame_size[0], frame_size[1], n_actions).to(device)
    target_net = model(frame_size[0], frame_size[1], n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters(), lr=LR, momentum=MOMENTUM, eps=MIN_SQR_GRAD)  # Create RMS optimizer
    memory = ReplayMemory(memory_size)  # Create replay memory for batches

    dqn_tuple = policy_net, target_net, optimizer, memory # Group all variable in a tuple
    return dqn_tuple



# ----------- TRAINING ----------- #
def train_dqn(dqn_tuple, env, num_episodes=NUM_EPISODES, device=DEVICE):
    policy_net, target_net, optimizer, memory = dqn_tuple   # Extract variables from tuple
    policy_net.train()
    
    steps_done = 0
    episodes_rewards = np.zeros(num_episodes)
    for episode in range(num_episodes):        
        # Initialize the environment and state
        env_state = env.start()        
        state = get_state_batch(env_state)

        # Start episode loop
        episode_reward = 0
        done = False
        start_steps = steps_done
        while not done:
            # Select and perform an action            
            steps_done += 1
            action = select_action(state, policy_net, steps_done)
            env_state, reward, done = env.step(action.item())
            episode_reward += reward
            reward = torch.tensor([reward])

            # Observe new state            
            if done:
                next_state = None
            else:
                next_state = get_state_batch(env_state)
            
            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state            

            # Perform one step of the optimization (with target network as reference)
            if steps_done > STEPS_BEFORE_START_LEARNING and steps_done % STEPS_PER_POLICY_UPD == 0:
                optimize_model(policy_net, target_net, memory, optimizer)
            
            # Update the target network, copying all weights and biases from policy DQN
            if steps_done > STEPS_BEFORE_START_LEARNING and steps_done % STEPS_PER_TARGET_UPD == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # When episode completed, store reward
        episodes_rewards[episode] = episode_reward        
        
        # Print episode information
        print(f"Epoch {episode+1}/{num_episodes} | {steps_done-start_steps} steps with a reward of {episode_reward}")

    return episodes_rewards


def select_action(state, policy_net, steps_done, eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY, device=DEVICE):
    action = 0    
    
    # If sample is lower than epsilon threshold, perform a random (exploratory) action
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
    if sample <= eps_threshold:
        n_actions = policy_net.get_num_outputs()
        action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    # Otherwise, use the policy net
    else:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action = policy_net(state.to(device)).max(1)[1].view(1, 1)
    
    return action


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
    expected_state_action_values = (1-GAMMA) * reward_batch + GAMMA * next_state_values

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model (clamping the parameters for stability)
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
    elif game_name == CARTPOLE:
        folder = PATH_RESULTS_DQN_CARTPOLE
    
    # Create folder if it does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Get filename and path
    filename = f"rewards_{game_name}_{time.strftime('%Y%m%d-%H%M%S')}_{SELECTED_MODEL}.npy"
    path = os.path.join(folder, filename)

    # Store
    np.save(path, rewards)

    return path


def plot_rewards_from_file(filepath, title=None, store_img=False):
    results = np.load(filepath) # Load rewards array    

    # Get image file path if required
    img_filepath = None
    if store_img:
        base, ext = os.path.splitext(filepath)
        img_filepath = base+".png"
    
    # Get filename as title
    if title is None:        
        filename = os.path.basename(filepath)
        filename, _ = os.path.splitext(filename)
        title = filename
    
    # Plot the rewards and store the corresponding image if required
    plot_rewards(results, title, img_filepath=img_filepath)


def plot_rewards(rewards, title, num_avg_points=10, img_filepath=None):
    # Get averaging
    avg_divisor = len(rewards) // num_avg_points
    avg_indexes = np.linspace(0, len(rewards), num_avg_points, dtype=np.int)
    average_rewards = np.empty(len(avg_indexes))
    last_idx = 0
    for i, real_idx in enumerate(avg_indexes):
        if last_idx == real_idx:
            average_rewards[i] = rewards[real_idx]
        else:
            average_rewards[i] = np.mean(rewards[last_idx:real_idx])
        last_idx = real_idx

    # Perform plots
    plt.plot(rewards, color="blue", label="Raw rewards")
    plt.plot(avg_indexes, average_rewards, color="orange", label=f"Mean for each {avg_divisor} rewards")
    plt.title(title)
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    # Store image if necessary
    if img_filepath is not None:
        plt.savefig(img_filepath, dpi=300, bbox_inches='tight')

    plt.show()


# ----------- RUNNING METHOD ----------- #
def run_dqn(game_name):
    # Get game environment
    if game_name == BREAKOUT:
        env = Breakout(NUM_STACKED_FRAMES, FRAME_SIZE, reward_scale=1, only_right_left=False)
    elif game_name == CARTPOLE:
        env = CartPole(reward_scale=1, num_stacked=NUM_STACKED_FRAMES, frame_resize=FRAME_SIZE)
    else:
        raise Exception(f"Game {game_name} not compatible with DQN")

    # Perform procedure
    print(f"Selected device = {DEVICE}")

    print("Creating model...")
    dqn_tuple = create_dqn(env)
    
    print("Training...")
    rewards = train_dqn(dqn_tuple, env)

    print("Storing models")
    policy_net, target_net, optimizer, memory = dqn_tuple   # Extract variables from tuple
    torch.save(policy_net.state_dict(), os.path.join(PATH_RESULTS_DQN,f"policy_{SELECTED_MODEL}_net.chkp"))
    torch.save(target_net.state_dict(), os.path.join(PATH_RESULTS_DQN,f"target_{SELECTED_MODEL}_net.chkp"))
    
    print("Storing rewards...")
    filepath = store_rewards(rewards, game_name)
    
    print("Plotting rewards...")
    plot_rewards_from_file(filepath, store_img=True)


    # Close environment
    env.end()

    return rewards