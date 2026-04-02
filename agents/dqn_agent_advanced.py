import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # Enhanced network capacity for better representation
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class DQNAgentAdvanced:
    def __init__(self, state_size, action_size, device='cpu', learning_rate=0.0005):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Double DQN: two separate networks
        self.q_network = DQN(state_size, action_size).to(device)
        self.target_network = DQN(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Larger replay buffer and batch size
        self.memory = ReplayBuffer(20000)
        self.batch_size = 128  # Increased for smoother learning
        self.gamma = 0.99
        
        # Improved epsilon settings
        self.epsilon = 1.0
        self.epsilon_min = 0.005  # Lower minimum for less random mistakes
        self.epsilon_decay = 0.97  # Aggressive decay
        self.update_target_every = 200  # Less frequent updates for stability
        self.steps = 0
        
        # Track training
        self.loss_history = []
        
    def act(self, state, eval_mode=False, valid_actions_mask=None):
        """Choose action with optional action masking"""
        # For evaluation mode, use greedy policy
        if eval_mode:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
            
            # Apply action mask if provided (reduce action space)
            if valid_actions_mask is not None:
                # Set invalid actions to very negative values
                mask = torch.FloatTensor(valid_actions_mask).to(self.device)
                q_values = q_values + (mask - 1) * 1e6
            
            return q_values.argmax().item()
        
        # For training, use epsilon-greedy
        if np.random.random() < self.epsilon:
            # If mask provided, only sample from valid actions
            if valid_actions_mask is not None:
                valid_indices = [i for i, valid in enumerate(valid_actions_mask) if valid]
                if valid_indices:
                    return np.random.choice(valid_indices)
            return np.random.randint(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        
        # Apply action mask
        if valid_actions_mask is not None:
            mask = torch.FloatTensor(valid_actions_mask).to(self.device)
            q_values = q_values + (mask - 1) * 1e6
        
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        
    def replay(self):
        """Double DQN implementation - reduces overestimation"""
        if len(self.memory) < self.batch_size:
            return 0
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # DOUBLE DQN: Use current network to select actions, target network to evaluate
        with torch.no_grad():
            # Get actions from current network
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            # Get Q values from target network for those actions
            next_q = self.target_network(next_states).gather(1, next_actions).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network with delay
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        return loss_value
    
    def get_action_mask(self, state, env):
        """Generate mask for valid actions (reduces action space)"""
        # This helps the agent focus on meaningful actions
        # For now, return all valid (we'll implement smart masking later)
        return [1] * self.action_size
    
    def save(self, path):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'loss_history': self.loss_history
        }, path)
        
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', 1.0)
        self.steps = checkpoint.get('steps', 0)
        self.loss_history = checkpoint.get('loss_history', [])