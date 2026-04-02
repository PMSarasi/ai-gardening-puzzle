# train_final.py - FINAL OPTIMIZED VERSION
import sys
import os
sys.path.append('.')

import torch
import numpy as np
from environment.garden_env import GardenEnv
from agents.dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import time

class EpsilonScheduler:
    """Custom epsilon scheduler for better exploration"""
    def __init__(self, start=1.0, end=0.01, decay_steps=500):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.current = start
        
    def get_epsilon(self, episode):
        """Linear decay for first decay_steps, then constant"""
        if episode < self.decay_steps:
            decay = (self.start - self.end) * (1 - episode / self.decay_steps)
            self.current = self.start - (self.start - self.end) * (episode / self.decay_steps)
            return max(self.end, self.current)
        return self.end
    
    def reset(self):
        self.current = self.start

def train_final_gardener(grid_size=3, max_episodes=800, save_path="models/final_gardener.pth"):
    """Final optimized training with proper exploration"""
    
    print("="*70)
    print("🌱 FINAL OPTIMIZED TRAINING - AI GARDENER")
    print("="*70)
    print(f"Garden Size: {grid_size}x{grid_size}")
    print(f"Max Episodes: {max_episodes}")
    print("="*70)
    
    # Create environment
    env = GardenEnv(grid_size=grid_size)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"\n📊 Environment Info:")
    print(f"  - State size: {state_size}")
    print(f"  - Action size: {action_size}")
    print(f"  - Max steps: {env.max_steps}")
    
    # Create agent with optimal parameters
    device = torch.device("cpu")
    agent = DQNAgent(
        state_size, 
        action_size, 
        device, 
        learning_rate=0.0005
    )
    
    # Custom epsilon scheduler
    epsilon_scheduler = EpsilonScheduler(start=1.0, end=0.05, decay_steps=400)
    
    # Training metrics
    rewards_history = []
    epsilon_history = []
    best_reward = -float('inf')
    patience_counter = 0
    best_episode = 0
    
    print("\n🚀 Starting Optimized Training...\n")
    print("="*70)
    print(f"{'Episode':<10} {'Reward':<10} {'Avg(50)':<10} {'Best':<10} {'Epsilon':<10} {'Status':<15}")
    print("="*70)
    
    start_time = time.time()
    
    for episode in range(max_episodes):
        # Update epsilon using scheduler
        agent.epsilon = epsilon_scheduler.get_epsilon(episode)
        epsilon_history.append(agent.epsilon)
        
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
            avg_50 = np.mean(rewards_history[-50:])
        else:
            avg_50 = total_reward
        
        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(save_path.replace('.pth', '_best.pth'))
            best_episode = episode
            patience_counter = 0
            status = "🔥 NEW BEST!"
        else:
            patience_counter += 1
            status = ""
        
        # Progress reporting
        if (episode + 1) % 50 == 0 or episode == 0:
            print(f"{episode+1:<10} {total_reward:<10.2f} {avg_50:<10.2f} "
                  f"{best_reward:<10.2f} {agent.epsilon:<10.3f} {status}")
        
        # Early stopping if no improvement for long time
        if patience_counter > 150 and episode > 300:
            print(f"\n⚠️ Early stopping at episode {episode+1} - no improvement for 150 episodes")
            break
    
    # Save final model
    agent.save(save_path)
    
    training_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("✅ Training Complete!")
    print(f"Training Time: {training_time/60:.1f} minutes")
    print(f"Best Model: {save_path.replace('.pth', '_best.pth')} (Episode {best_episode+1})")
    print(f"Final Model: {save_path}")
    print("="*70)
    
    # Show summary statistics
    print("\n📈 Training Summary:")
    print(f"  - Best Reward: {best_reward:.2f}")
    print(f"  - Final Reward: {rewards_history[-1]:.2f}")
    print(f"  - Average (last 50): {np.mean(rewards_history[-50:]):.2f}")
    print(f"  - Average (all): {np.mean(rewards_history):.2f}")
    
    # Test the best model
    print("\n🧪 Testing Best Model (10 episodes)...")
    best_agent = DQNAgent(state_size, action_size, device)
    best_agent.load(save_path.replace('.pth', '_best.pth'))
    best_agent.epsilon = 0.01  # Minimal exploration for testing
    
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
        print(f"  Test {test+1:2d}: {test_reward:.2f} {'⭐' if test_reward > 80 else '✓' if test_reward > 60 else '⚠️'}")
    
    print(f"\n📊 Test Summary:")
    print(f"  - Average: {np.mean(test_rewards):.2f}")
    print(f"  - Std Dev: {np.std(test_rewards):.2f}")
    print(f"  - Best: {max(test_rewards):.2f}")
    print(f"  - Worst: {min(test_rewards):.2f}")
    print(f"  - Success Rate (>60): {sum(1 for r in test_rewards if r > 60)/10*100:.0f}%")
    print(f"  - Excellent Rate (>80): {sum(1 for r in test_rewards if r > 80)/10*100:.0f}%")
    
    # Generate comprehensive plots
    generate_plots(rewards_history, epsilon_history, test_rewards)
    
    return agent, rewards_history, test_rewards

def generate_plots(rewards, epsilons, test_rewards):
    """Generate comprehensive training plots"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Training Rewards
    axes[0, 0].plot(rewards, alpha=0.5, linewidth=0.5, label='Episode Reward')
    if len(rewards) > 50:
        moving_avg = np.convolve(rewards, np.ones(50)/50, mode='valid')
        axes[0, 0].plot(range(49, len(rewards)), moving_avg, 'r-', linewidth=2, label='Moving Avg (50)')
    axes[0, 0].axhline(y=np.mean(rewards[-100:]), color='g', linestyle='--', label=f'Final Avg: {np.mean(rewards[-100:]):.1f}')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Epsilon Decay
    axes[0, 1].plot(epsilons, 'b-', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Epsilon')
    axes[0, 1].set_title('Exploration Rate (Epsilon)')
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0.05, color='r', linestyle='--', label='Target (0.05)')
    axes[0, 1].legend()
    
    # Plot 3: Reward Distribution
    axes[0, 2].hist(rewards[-200:], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 2].axvline(x=np.mean(rewards[-200:]), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards[-200:]):.1f}')
    axes[0, 2].set_xlabel('Reward')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Reward Distribution (Last 200 Episodes)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Test Results
    axes[1, 0].bar(range(1, len(test_rewards)+1), test_rewards, color=['green' if r > 80 else 'orange' if r > 60 else 'red' for r in test_rewards])
    axes[1, 0].axhline(y=np.mean(test_rewards), color='blue', linestyle='--', linewidth=2, label=f'Avg: {np.mean(test_rewards):.1f}')
    axes[1, 0].set_xlabel('Test Episode')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].set_title('Final Test Results')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Moving Average Trend
    if len(rewards) > 100:
        ma_100 = np.convolve(rewards, np.ones(100)/100, mode='valid')
        axes[1, 1].plot(range(99, len(rewards)), ma_100, 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Moving Average (100)')
        axes[1, 1].set_title('Learning Trend')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(range(len(ma_100)), ma_100, 1)
        p = np.poly1d(z)
        axes[1, 1].plot(range(99, len(rewards)), p(range(len(ma_100))), "r--", alpha=0.8, label=f'Trend: {z[0]:.2f}/ep')
        axes[1, 1].legend()
    
    # Plot 6: Performance Summary
    stats_text = f"""
    📊 FINAL PERFORMANCE SUMMARY
    
    Training Stats:
    • Best Reward: {max(rewards):.2f}
    • Final Avg (50): {np.mean(rewards[-50:]):.2f}
    • Training Episodes: {len(rewards)}
    
    Test Stats:
    • Average: {np.mean(test_rewards):.2f}
    • Std Dev: {np.std(test_rewards):.2f}
    • Best: {max(test_rewards):.2f}
    • Worst: {min(test_rewards):.2f}
    
    Quality Metrics:
    • Success Rate (>60): {sum(1 for r in test_rewards if r > 60)/len(test_rewards)*100:.0f}%
    • Excellent Rate (>80): {sum(1 for r in test_rewards if r > 80)/len(test_rewards)*100:.0f}%
    """
    
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('final_training_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Final plots saved as 'final_training_results.png'")
    plt.show()

if __name__ == "__main__":
    print("="*70)
    print("🎯 FINAL OPTIMIZED AI GARDENER TRAINING")
    print("="*70)
    print("\nThis version features:")
    print("  ✓ Proper epsilon scheduling (linear decay to 0.05 over 400 episodes)")
    print("  ✓ More exploration for better strategy discovery")
    print("  ✓ Early stopping to prevent overfitting")
    print("  ✓ Comprehensive performance tracking")
    print("="*70)
    
    agent, rewards, test_rewards = train_final_gardener(
        grid_size=3,
        max_episodes=800,
        save_path="models/final_gardener.pth"
    )
    
    print("\n" + "="*70)
    print("🎉 FINAL OPTIMIZATION COMPLETE!")
    print("="*70)
    print("\nExpected Results:")
    print("  • Test Average: 75-85 (up from 73)")
    print("  • Std Deviation: <12 (down from 16.75)")
    print("  • Success Rate: >90% (>60 reward)")
    print("  • Excellent Rate: >50% (>80 reward)")
    print("\nRun the app to see your improved AI Gardener in action!")
    print("streamlit run main.py")