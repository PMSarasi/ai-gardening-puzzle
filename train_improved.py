# train_improved.py
import sys
import os
sys.path.append('.')

import torch
import numpy as np
from environment.garden_env import GardenEnv
from agents.dqn_agent import DQNAgent
import time
import matplotlib.pyplot as plt

def train_improved_gardener(grid_size=3, episodes=1000, save_path="models/best_gardener.pth"):
    """Improved training with better exploration"""
    
    print("="*60)
    print("🌱 Training AI Gardener - IMPROVED VERSION")
    print("="*60)
    print(f"Garden Size: {grid_size}x{grid_size}")
    print(f"Training Episodes: {episodes}")
    print("="*60)
    
    # Create environment
    env = GardenEnv(grid_size=grid_size)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"\n📊 Environment Info:")
    print(f"  - State size: {state_size}")
    print(f"  - Action size: {action_size}")
    print(f"  - Max steps: {env.max_steps}")
    
    # Create agent with slower epsilon decay
    device = torch.device("cpu")
    agent = DQNAgent(state_size, action_size, device, learning_rate=0.0005)  # Lower LR for stability
    
    # Training metrics
    rewards_history = []
    avg_rewards = []
    best_reward = -float('inf')
    
    print("\n🚀 Starting Improved Training...\n")
    print("="*60)
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_steps = 0
        
        for step in range(env.max_steps):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()
            
            total_reward += reward
            state = next_state
            episode_steps += 1
            
            if done or truncated:
                break
        
        rewards_history.append(total_reward)
        
        # Calculate moving averages
        if len(rewards_history) >= 50:
            avg_reward_50 = np.mean(rewards_history[-50:])
            avg_rewards.append(avg_reward_50)
        else:
            avg_reward_50 = total_reward
        
        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(save_path.replace('.pth', '_best.pth'))
        
        # Progress reporting with more detail
        if (episode + 1) % 50 == 0:
            recent_avg = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else total_reward
            print(f"Episode {episode + 1:4d}/{episodes} | "
                  f"Reward: {total_reward:6.2f} | "
                  f"Avg (50): {recent_avg:6.2f} | "
                  f"Best: {best_reward:6.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
        
        # Early stopping if plateaued
        if episode > 500 and len(avg_rewards) > 20:
            recent_improvement = avg_rewards[-1] - avg_rewards[-20]
            if recent_improvement < 0.5 and agent.epsilon < 0.05:
                print(f"\n⚠️ Plateau detected at episode {episode+1}")
                print(f"Stopping early to prevent overfitting...")
                break
    
    # Save final model
    agent.save(save_path)
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print(f"Best Model: {save_path.replace('.pth', '_best.pth')}")
    print(f"Final Model: {save_path}")
    print("="*60)
    
    # Show summary statistics
    print("\n📈 Training Summary:")
    print(f"  - Best Reward: {best_reward:.2f}")
    print(f"  - Final Reward: {rewards_history[-1]:.2f}")
    print(f"  - Average (last 50): {np.mean(rewards_history[-50:]):.2f}")
    print(f"  - Average (all): {np.mean(rewards_history):.2f}")
    
    # Test the best model
    print("\n🧪 Testing Best Model...")
    best_agent = DQNAgent(state_size, action_size, device)
    best_agent.load(save_path.replace('.pth', '_best.pth'))
    
    test_rewards = []
    for test in range(10):
        state, _ = env.reset()
        test_reward = 0
        for step in range(env.max_steps):
            action = best_agent.act(state, eval_mode=True)
            next_state, reward, done, truncated, _ = env.step(action)
            test_reward += reward
            state = next_state
            if done or truncated:
                break
        test_rewards.append(test_reward)
        print(f"  Test {test+1:2d}: {test_reward:.2f}")
    
    print(f"\n📊 Test Summary:")
    print(f"  - Average: {np.mean(test_rewards):.2f}")
    print(f"  - Std Dev: {np.std(test_rewards):.2f}")
    print(f"  - Best: {max(test_rewards):.2f}")
    print(f"  - Worst: {min(test_rewards):.2f}")
    
    # Plot results
    plot_training_results(rewards_history, avg_rewards)
    
    return agent, rewards_history

def plot_training_results(rewards, avg_rewards):
    """Plot training results"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.5, label='Episode Reward')
    if len(rewards) > 50:
        moving_avg = np.convolve(rewards, np.ones(50)/50, mode='valid')
        plt.plot(range(49, len(rewards)), moving_avg, 'r-', label='Moving Avg (50)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(rewards[-100:], bins=20, edgecolor='black')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution (Last 100 Episodes)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    print("\n📊 Training plot saved as 'training_results.png'")
    plt.show()

if __name__ == "__main__":
    # Train with improved parameters
    agent, rewards = train_improved_gardener(
        grid_size=3, 
        episodes=1000,  # More episodes
    )
    
    print("\n🎉 Training complete! Run the app to see your improved AI Gardener!")