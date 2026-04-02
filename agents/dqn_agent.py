import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
        
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity=20000):  # Increased capacity for better learning
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

class DQNAgent:
    def __init__(self, state_size, action_size, device='cpu', learning_rate=0.0005):  # Lower LR for stability
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        self.q_network = DQN(state_size, action_size).to(device)
        self.target_network = DQN(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # IMPROVED: Larger replay buffer and batch size
        self.memory = ReplayBuffer(20000)  # Increased from 10000
        self.batch_size = 128  # Increased from 64
        
        self.gamma = 0.99
        
        # IMPROVED: Better epsilon decay parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998  # Slower decay for more exploration
        self.epsilon_decay_steps = 0
        
        # IMPROVED: Less frequent target updates for stability
        self.update_target_every = 200  # Increased from 100
        self.steps = 0
        
        # Track training progress
        self.loss_history = []
        
    def act(self, state, eval_mode=False):
        """Choose action using epsilon-greedy"""
        # Ensure state is properly formatted
        if isinstance(state, list):
            state = np.array(state)
        
        # For evaluation mode, use greedy policy
        if eval_mode:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
            return q_values.argmax().item()
        
        # For training, use epsilon-greedy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
        
    def replay(self):
        """Train the agent with improved stability"""
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
        
        # Target Q values (Double DQN improvement)
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
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Decay epsilon based on steps (slower decay)
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network less frequently
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Store loss for monitoring
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        return loss_value
    
    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'loss_history': self.loss_history
        }, path)
        
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', 1.0)
        self.steps = checkpoint.get('steps', 0)
        self.loss_history = checkpoint.get('loss_history', [])
        
    def get_action_probs(self, state):
        """Get action probabilities for visualization"""
        if isinstance(state, list):
            state = np.array(state)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        probs = torch.softmax(q_values, dim=1)
        return probs.cpu().numpy()[0]
    
    def get_epsilon(self):
        """Return current epsilon value"""
        return self.epsilon
    
    def reset_epsilon(self):
        """Reset epsilon for new training session"""
        self.epsilon = 1.0