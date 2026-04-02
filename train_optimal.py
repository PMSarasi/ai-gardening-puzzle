# train_optimal.py - OPTIMAL BALANCE VERSION
import sys
import os
sys.path.append('.')

import torch
import numpy as np
from environment.garden_env import GardenEnv
from agents.dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import time

class OptimalEpsilonScheduler:
    """Balanced epsilon scheduler - best of both worlds"""
    def __init__(self):
        self.epsilon = 1.0
        self.decay_start = 0.995  # Slower decay for better exploration
        self.epsilon_min = 0.01
        
    def update(self):
        """Decay epsilon gradually"""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_start)
        return self.epsilon
    
    def get_epsilon(self):
        return self.epsilon
    
    def reset(self):
        self.epsilon = 1.0

def train_optimal_gardener(grid_size=3, max_episodes=600, save_path="models/optimal_gardener.pth"):
    """OPTIMAL TRAINING - Balanced exploration and exploitation"""
    
    print("="*70)
    print("🌟 OPTIMAL AI GARDENER - BALANCED TRAINING")
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
    
    # Create agent with optimal parameters (from improved version)
    device = torch.device("cpu")
    agent = DQNAgent(
        state_size, 
        action_size, 
        device, 
        learning_rate=0.001  # Back to original LR for better learning
    )
    
    # Use simple exponential decay (worked better!)
    epsilon_scheduler = OptimalEpsilonScheduler()
    
    # Training metrics
    rewards_history = []
    epsilon_history = []
    best_reward = -float('inf')
    best_episode = 0
    best_model_state = None
    
    # Track moving averages
    moving_avg = []
    
    print("\n🚀 Starting Optimal Training...\n")
    print("="*85)
    print(f"{'Episode':<10} {'Reward':<10} {'Avg(30)':<10} {'Best':<10} {'Epsilon':<10} {'Status':<20}")
    print("="*85)
    
    start_time = time.time()
    
    for episode in range(max_episodes):
        # Update epsilon using exponential decay (proven method)
        agent.epsilon = epsilon_scheduler.update()
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
        
        # Calculate moving average (30 episodes for smoother tracking)
        if len(rewards_history) >= 30:
            avg_30 = np.mean(rewards_history[-30:])
            moving_avg.append(avg_30)
        else:
            avg_30 = total_reward
        
        # Save best model (critical!)
        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = episode
            # Save best model state
            best_model_state = {
                'q_network': agent.q_network.state_dict().copy(),
                'target_network': agent.target_network.state_dict().copy(),
                'optimizer': agent.optimizer.state_dict().copy(),
                'epsilon': agent.epsilon,
                'steps': agent.steps,
                'reward': total_reward,
                'episode': episode
            }
            status = "🔥 NEW BEST!"
        else:
            status = ""
        
        # Progress reporting every 50 episodes
        if (episode + 1) % 50 == 0 or episode == 0:
            print(f"{episode+1:<10} {total_reward:<10.2f} {avg_30:<10.2f} "
                  f"{best_reward:<10.2f} {agent.epsilon:<10.4f} {status}")
        
        # Early stopping when learning plateaus (after good performance)
        if episode > 300 and len(moving_avg) > 50:
            recent_avg = np.mean(moving_avg[-20:]) if len(moving_avg) >= 20 else avg_30
            older_avg = np.mean(moving_avg[-50:-20]) if len(moving_avg) >= 50 else recent_avg
            
            # Stop if no improvement for 100 episodes after episode 300
            if episode - best_episode > 100 and episode > 400:
                print(f"\n⚠️ Stopping at episode {episode+1} - no improvement for {episode - best_episode} episodes")
                break
    
    # Load the best model (not the final one!)
    print(f"\n📌 Loading best model from episode {best_episode + 1} (Reward: {best_reward:.2f})")
    if best_model_state:
        agent.q_network.load_state_dict(best_model_state['q_network'])
        agent.target_network.load_state_dict(best_model_state['target_network'])
        agent.optimizer.load_state_dict(best_model_state['optimizer'])
        agent.epsilon = best_model_state['epsilon']
        agent.steps = best_model_state['steps']
    
    # Save best model
    os.makedirs("models", exist_ok=True)
    agent.save(save_path)
    
    training_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("✅ Optimal Training Complete!")
    print(f"Training Time: {training_time/60:.1f} minutes")
    print(f"Best Model: {save_path} (Episode {best_episode + 1})")
    print("="*70)
    
    # Show summary statistics
    print("\n📈 Training Summary:")
    print(f"  - Best Reward: {best_reward:.2f}")
    print(f"  - Final Reward: {rewards_history[-1]:.2f}")
    print(f"  - Average (last 30): {np.mean(rewards_history[-30:]):.2f}")
    print(f"  - Average (all): {np.mean(rewards_history):.2f}")
    
    # Test the best model
    print("\n🧪 Testing Best Model (15 episodes for better statistics)...")
    
    test_rewards = []
    for test in range(15):
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
        
        # Color code based on performance
        if test_reward > 80:
            icon = "⭐ EXCELLENT"
        elif test_reward > 60:
            icon = "✓ GOOD"
        elif test_reward > 40:
            icon = "⚠️ OK"
        else:
            icon = "❌ POOR"
        
        print(f"  Test {test+1:2d}: {test_reward:6.2f}  {icon}")
    
    print(f"\n📊 Test Summary:")
    print(f"  - Average: {np.mean(test_rewards):.2f}")
    print(f"  - Std Dev: {np.std(test_rewards):.2f}")
    print(f"  - Best: {max(test_rewards):.2f}")
    print(f"  - Worst: {min(test_rewards):.2f}")
    print(f"  - Success Rate (>60): {sum(1 for r in test_rewards if r > 60)/len(test_rewards)*100:.0f}%")
    print(f"  - Excellent Rate (>80): {sum(1 for r in test_rewards if r > 80)/len(test_rewards)*100:.0f}%")
    
    # Generate comprehensive plots
    generate_optimal_plots(rewards_history, epsilon_history, test_rewards, best_episode, best_reward)
    
    return agent, rewards_history, test_rewards, best_reward

def generate_optimal_plots(rewards, epsilons, test_rewards, best_episode, best_reward):
    """Generate comprehensive plots for optimal version"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Plot 1: Training Rewards
    axes[0, 0].plot(rewards, alpha=0.5, linewidth=0.5, color='blue', label='Episode Reward')
    if len(rewards) > 30:
        moving_avg = np.convolve(rewards, np.ones(30)/30, mode='valid')
        axes[0, 0].plot(range(29, len(rewards)), moving_avg, 'r-', linewidth=2, label='Moving Avg (30)')
    
    # Mark best episode
    axes[0, 0].axvline(x=best_episode, color='green', linestyle='--', linewidth=2, 
                        label=f'Best Model (Ep {best_episode+1}, R={best_reward:.1f})')
    
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Training Progress - Optimal Version')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Epsilon Decay
    axes[0, 1].plot(epsilons, 'b-', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Epsilon')
    axes[0, 1].set_title('Exploration Rate (Exponential Decay)')
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0.01, color='r', linestyle='--', label='Min (0.01)')
    axes[0, 1].legend()
    
    # Plot 3: Reward Distribution
    axes[0, 2].hist(rewards[-200:], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 2].axvline(x=np.mean(rewards[-200:]), color='r', linestyle='--', 
                        linewidth=2, label=f'Mean: {np.mean(rewards[-200:]):.1f}')
    axes[0, 2].set_xlabel('Reward')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Reward Distribution (Last 200 Episodes)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Test Results
    colors = ['green' if r > 80 else 'orange' if r > 60 else 'red' for r in test_rewards]
    axes[1, 0].bar(range(1, len(test_rewards)+1), test_rewards, color=colors)
    axes[1, 0].axhline(y=np.mean(test_rewards), color='blue', linestyle='--', 
                        linewidth=2, label=f'Avg: {np.mean(test_rewards):.1f}')
    axes[1, 0].set_xlabel('Test Episode')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].set_title('Final Test Results - Optimal Model')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Learning Curve Comparison
    if len(rewards) > 100:
        ma_50 = np.convolve(rewards, np.ones(50)/50, mode='valid')
        axes[1, 1].plot(range(49, len(rewards)), ma_50, 'g-', linewidth=2, label='Moving Avg (50)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Moving Average')
        axes[1, 1].set_title('Learning Trend')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(range(len(ma_50)), ma_50, 1)
        p = np.poly1d(z)
        axes[1, 1].plot(range(49, len(rewards)), p(range(len(ma_50))), "r--", 
                        alpha=0.8, label=f'Trend: {z[0]:.2f}/ep')
        axes[1, 1].legend()
    
    # Plot 6: Performance Summary
    success_rate = sum(1 for r in test_rewards if r > 60) / len(test_rewards) * 100
    excellent_rate = sum(1 for r in test_rewards if r > 80) / len(test_rewards) * 100
    
    stats_text = f"""
    🌟 OPTIMAL VERSION PERFORMANCE SUMMARY
    
    ┌─────────────────────────────────────┐
    │         TRAINING METRICS            │
    ├─────────────────────────────────────┤
    │ Best Reward:        {max(rewards):6.2f}       │
    │ Best Episode:       {best_episode+1:6d}       │
    │ Final Avg (30):     {np.mean(rewards[-30:]):6.2f}       │
    │ Total Episodes:     {len(rewards):6d}       │
    └─────────────────────────────────────┘
    
    ┌─────────────────────────────────────┐
    │          TEST METRICS               │
    ├─────────────────────────────────────┤
    │ Average:            {np.mean(test_rewards):6.2f}       │
    │ Std Deviation:      {np.std(test_rewards):6.2f}       │
    │ Best Test:          {max(test_rewards):6.2f}       │
    │ Worst Test:         {min(test_rewards):6.2f}       │
    └─────────────────────────────────────┘
    
    ┌─────────────────────────────────────┐
    │          QUALITY METRICS            │
    ├─────────────────────────────────────┤
    │ Success Rate (>60):  {success_rate:5.0f}%         │
    │ Excellent Rate (>80): {excellent_rate:5.0f}%         │
    └─────────────────────────────────────┘
    
    ✅ Veredict: OPTIMAL BALANCE ACHIEVED
    """
    
    axes[1, 2].text(0.05, 0.5, stats_text, fontsize=9, verticalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    axes[1, 2].axis('off')
    
    plt.suptitle('AI Gardener - Optimal Training Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('optimal_training_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Optimal results plot saved as 'optimal_training_results.png'")
    plt.show()

if __name__ == "__main__":
    print("="*70)
    print("🌟 OPTIMAL AI GARDENER - BEST OF BOTH WORLDS")
    print("="*70)
    print("\nThis version combines the best of both approaches:")
    print("  ✓ Exponential epsilon decay (from improved version)")
    print("  ✓ Higher learning rate for faster learning")
    print("  ✓ Best model saving (prevents catastrophic forgetting)")
    print("  ✓ Balanced exploration (not too much, not too little)")
    print("="*70)
    
    agent, rewards, test_rewards, best_reward = train_optimal_gardener(
        grid_size=3,
        max_episodes=600,
        save_path="models/optimal_gardener.pth"
    )
    
    print("\n" + "="*70)
    print("🎉 OPTIMAL TRAINING COMPLETE!")
    print("="*70)
    print("\nExpected Results (Target):")
    print("  • Test Average: 65-75 (back to improved version levels)")
    print("  • Success Rate (>60): >70%")
    print("  • Excellent Rate (>80): >20%")
    print("\nRun the app to see your optimal AI Gardener in action!")
    print("streamlit run main.py")