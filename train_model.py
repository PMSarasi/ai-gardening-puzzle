# train_model.py
import sys
import os
sys.path.append('.')

import torch
import numpy as np
from environment.garden_env import GardenEnv
from agents.dqn_agent import DQNAgent
import time

def train_gardener(grid_size=3, episodes=200, save_path="models/trained_gardener.pth"):
    """Train a gardener agent"""
    
    print("="*50)
    print("🌱 Training AI Gardener")
    print("="*50)
    print(f"Garden Size: {grid_size}x{grid_size}")
    print(f"Training Episodes: {episodes}")
    print("="*50)
    
    # Create environment
    env = GardenEnv(grid_size=grid_size)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"\n📊 Environment Info:")
    print(f"  - State size: {state_size}")
    print(f"  - Action size: {action_size}")
    print(f"  - Max steps per episode: {env.max_steps}")
    
    # Create agent
    device = torch.device("cpu")
    agent = DQNAgent(state_size, action_size, device, learning_rate=0.001)
    
    # Training metrics
    rewards_history = []
    avg_rewards = []
    
    print("\n🚀 Starting Training...\n")
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_steps = 0
        
        for step in range(env.max_steps):
            # Get action
            action = agent.act(state)
            
            # Take step
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Remember and train
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()
            
            total_reward += reward
            state = next_state
            episode_steps += 1
            
            if done or truncated:
                break
        
        rewards_history.append(total_reward)
        
        # Calculate moving average
        if len(rewards_history) >= 10:
            avg_reward = np.mean(rewards_history[-10:])
            avg_rewards.append(avg_reward)
        else:
            avg_reward = total_reward
        
        # Progress indicator
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1:4d}/{episodes} | "
                  f"Reward: {total_reward:6.2f} | "
                  f"Avg (10): {avg_reward:6.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    # Save the trained model
    os.makedirs("models", exist_ok=True)
    agent.save(save_path)
    
    print("\n" + "="*50)
    print("✅ Training Complete!")
    print(f"Model saved to: {save_path}")
    print("="*50)
    
    # Show summary statistics
    print("\n📈 Training Summary:")
    print(f"  - Best Reward: {max(rewards_history):.2f}")
    print(f"  - Final Reward: {rewards_history[-1]:.2f}")
    print(f"  - Average Reward (last 10): {np.mean(rewards_history[-10:]):.2f}")
    print(f"  - Average Reward (all): {np.mean(rewards_history):.2f}")
    
    # Test the trained agent
    print("\n🧪 Testing trained agent...")
    test_rewards = []
    for test in range(5):
        state, _ = env.reset()
        test_reward = 0
        for step in range(env.max_steps):
            action = agent.act(state, eval_mode=True)
            next_state, reward, done, truncated, _ = env.step(action)
            test_reward += reward
            state = next_state
            if done or truncated:
                break
        test_rewards.append(test_reward)
        print(f"  Test {test+1}: {test_reward:.2f}")
    
    print(f"\n📊 Test Summary:")
    print(f"  - Average Test Reward: {np.mean(test_rewards):.2f}")
    print(f"  - Best Test Reward: {max(test_rewards):.2f}")
    
    return agent, rewards_history

if __name__ == "__main__":
    # Train the agent
    agent, rewards = train_gardener(grid_size=3, episodes=200)
    
    print("\n🎉 You can now use 'Watch AI Agent' mode in the app!")