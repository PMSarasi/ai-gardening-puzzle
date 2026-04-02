# train_stable.py - STABLE LEARNING VERSION (Based on your best improved version)
import sys
import os
sys.path.append('.')

import torch
import numpy as np
from environment.garden_env import GardenEnv
from agents.dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import time
import copy

class StableEpsilonScheduler:
    """Aggressive epsilon decay for faster stabilization"""
    def __init__(self):
        self.epsilon = 1.0
        self.decay_rate = 0.97  # More aggressive decay
        self.epsilon_min = 0.01
        
    def update(self):
        """Decay epsilon aggressively"""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_rate)
        return self.epsilon
    
    def get_epsilon(self):
        return self.epsilon
    
    def reset(self):
        self.epsilon = 1.0

def train_stable_gardener(grid_size=3, max_episodes=350, save_path="models/stable_gardener.pth"):
    """STABLE TRAINING - Based on your best performing improved version"""
    
    print("="*70)
    print("✅ STABLE AI GARDENER - PROVEN APPROACH")
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
    
    # Create agent with proven parameters
    device = torch.device("cpu")
    agent = DQNAgent(
        state_size, 
        action_size, 
        device, 
        learning_rate=0.001  # Back to original that worked
    )
    
    # Use aggressive epsilon decay
    epsilon_scheduler = StableEpsilonScheduler()
    
    # Training metrics
    rewards_history = []
    epsilon_history = []
    test_rewards_history = []  # Track test performance during training
    
    # Best model tracking
    best_reward = -float('inf')
    best_episode = 0
    best_model_state = None
    best_test_avg = -float('inf')
    
    # Track moving average for stability
    moving_avg_30 = []
    
    print("\n🚀 Starting Stable Training...\n")
    print("="*85)
    print(f"{'Episode':<10} {'Reward':<10} {'Avg(30)':<10} {'Best':<10} {'Epsilon':<10} {'Test(10)':<12} {'Status':<15}")
    print("="*85)
    
    start_time = time.time()
    
    # Track consecutive good episodes for early stopping
    good_episodes = 0
    
    for episode in range(max_episodes):
        # Update epsilon
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
        
        # Calculate moving average
        if len(rewards_history) >= 30:
            avg_30 = np.mean(rewards_history[-30:])
            moving_avg_30.append(avg_30)
        else:
            avg_30 = total_reward
        
        # Quick test every 50 episodes to monitor generalization
        test_avg = None
        if (episode + 1) % 50 == 0 or total_reward > best_reward:
            test_rewards = []
            for test in range(10):
                test_state, _ = env.reset()
                test_total = 0
                for step in range(env.max_steps):
                    test_action = agent.act(test_state, eval_mode=True)
                    test_next, test_reward, test_done, test_trunc, _ = env.step(test_action)
                    test_total += test_reward
                    test_state = test_next
                    if test_done or test_trunc:
                        break
                test_rewards.append(test_total)
            test_avg = np.mean(test_rewards)
            test_rewards_history.append(test_avg)
        
        # Save best model based on training reward
        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = episode
            # Deep copy the model state
            best_model_state = {
                'q_network': copy.deepcopy(agent.q_network.state_dict()),
                'target_network': copy.deepcopy(agent.target_network.state_dict()),
                'optimizer': copy.deepcopy(agent.optimizer.state_dict()),
                'epsilon': agent.epsilon,
                'steps': agent.steps,
                'reward': total_reward,
                'episode': episode
            }
            status = "🔥 NEW BEST!"
            good_episodes = 0
        else:
            status = ""
            if total_reward > 40:  # Good episode threshold
                good_episodes += 1
            else:
                good_episodes = 0
        
        # Progress reporting
        if (episode + 1) % 50 == 0 or episode == 0:
            test_str = f"{test_avg:.1f}" if test_avg is not None else "N/A"
            print(f"{episode+1:<10} {total_reward:<10.2f} {avg_30:<10.2f} "
                  f"{best_reward:<10.2f} {agent.epsilon:<10.4f} {test_str:<12} {status}")
        
        # Early stopping when model is stable and performing well
        if episode > 150 and good_episodes > 20:
            print(f"\n✅ Model stable at episode {episode+1} - stopping early")
            break
        
        # Also stop if test performance degrades significantly
        if test_avg is not None and best_test_avg > 0 and test_avg < best_test_avg * 0.7:
            print(f"\n⚠️ Test performance dropping - stopping at episode {episode+1}")
            break
    
    # Load the best model (NOT the final one!)
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
    print("✅ Stable Training Complete!")
    print(f"Training Time: {training_time/60:.1f} minutes")
    print(f"Best Model: {save_path} (Episode {best_episode + 1})")
    print("="*70)
    
    # Show summary statistics
    print("\n📈 Training Summary:")
    print(f"  - Best Training Reward: {best_reward:.2f}")
    print(f"  - Final Training Reward: {rewards_history[-1]:.2f}")
    print(f"  - Average (last 30): {np.mean(rewards_history[-30:]):.2f}")
    print(f"  - Average (all): {np.mean(rewards_history):.2f}")
    
    # Comprehensive testing
    print("\n🧪 Comprehensive Testing (25 episodes for robust statistics)...")
    
    test_rewards = []
    for test in range(25):
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
    
    # Statistics
    test_mean = np.mean(test_rewards)
    test_std = np.std(test_rewards)
    test_best = max(test_rewards)
    test_worst = min(test_rewards)
    success_rate = sum(1 for r in test_rewards if r > 60) / len(test_rewards) * 100
    excellent_rate = sum(1 for r in test_rewards if r > 80) / len(test_rewards) * 100
    
    print(f"\n📊 Test Summary:")
    print(f"  - Average: {test_mean:.2f}")
    print(f"  - Std Dev: {test_std:.2f}")
    print(f"  - Best: {test_best:.2f}")
    print(f"  - Worst: {test_worst:.2f}")
    print(f"  - Success Rate (>60): {success_rate:.0f}%")
    print(f"  - Excellent Rate (>80): {excellent_rate:.0f}%")
    
    # Generate plots
    generate_stable_plots(rewards_history, epsilon_history, test_rewards, test_rewards_history, 
                          best_episode, best_reward, test_mean)
    
    # Final verdict
    print("\n" + "="*70)
    print("🎯 FINAL VERDICT")
    print("="*70)
    if test_mean >= 60:
        print("✅ EXCELLENT: Model generalizes well and performs consistently")
    elif test_mean >= 45:
        print("👍 GOOD: Model learns reasonable strategies with some variance")
    elif test_mean >= 30:
        print("⚠️ ACCEPTABLE: Model shows learning but needs improvement")
    else:
        print("❌ POOR: Model needs significant improvement")
    
    return agent, rewards_history, test_rewards, test_mean

def generate_stable_plots(rewards, epsilons, test_rewards, test_history, 
                          best_episode, best_reward, test_mean):
    """Generate comprehensive plots for stable version"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Plot 1: Training Rewards
    axes[0, 0].plot(rewards, alpha=0.5, linewidth=0.5, color='blue', label='Episode Reward')
    if len(rewards) > 30:
        moving_avg = np.convolve(rewards, np.ones(30)/30, mode='valid')
        axes[0, 0].plot(range(29, len(rewards)), moving_avg, 'r-', linewidth=2, label='Moving Avg (30)')
    
    axes[0, 0].axvline(x=best_episode, color='green', linestyle='--', linewidth=2, 
                        label=f'Best Model (Ep {best_episode+1}, R={best_reward:.1f})')
    axes[0, 0].axhline(y=test_mean, color='purple', linestyle=':', linewidth=2, 
                        label=f'Test Avg: {test_mean:.1f}')
    
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Training Progress - Stable Version')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Epsilon Decay
    axes[0, 1].plot(epsilons, 'b-', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Epsilon')
    axes[0, 1].set_title('Exploration Rate (Aggressive Decay)')
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0.05, color='r', linestyle='--', label='Target (0.05 by ep 100)')
    axes[0, 1].legend()
    
    # Plot 3: Reward Distribution
    axes[0, 2].hist(rewards[-150:], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 2].axvline(x=np.mean(rewards[-150:]), color='r', linestyle='--', 
                        linewidth=2, label=f'Mean: {np.mean(rewards[-150:]):.1f}')
    axes[0, 2].set_xlabel('Reward')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Training Reward Distribution (Last 150)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Test Results
    colors = ['green' if r > 60 else 'orange' if r > 40 else 'red' for r in test_rewards]
    axes[1, 0].bar(range(1, len(test_rewards)+1), test_rewards, color=colors)
    axes[1, 0].axhline(y=np.mean(test_rewards), color='blue', linestyle='--', 
                        linewidth=2, label=f'Avg: {np.mean(test_rewards):.1f}')
    axes[1, 0].fill_between(range(1, len(test_rewards)+1), 
                             np.mean(test_rewards) - np.std(test_rewards),
                             np.mean(test_rewards) + np.std(test_rewards),
                             alpha=0.2, color='blue', label='±1 Std')
    axes[1, 0].set_xlabel('Test Episode')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].set_title('Final Test Results - Stable Model')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Learning Trend
    if len(rewards) > 100:
        ma_50 = np.convolve(rewards, np.ones(50)/50, mode='valid')
        axes[1, 1].plot(range(49, len(rewards)), ma_50, 'g-', linewidth=2, label='Moving Avg (50)')
        
        # Add trend line for last 100 episodes
        if len(ma_50) > 100:
            recent = ma_50[-100:]
            z = np.polyfit(range(len(recent)), recent, 1)
            p = np.poly1d(z)
            axes[1, 1].plot(range(len(rewards)-100, len(rewards)), 
                           p(range(len(recent))), "r--", 
                           alpha=0.8, label=f'Trend: {z[0]:.3f}/ep')
        
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Moving Average')
        axes[1, 1].set_title('Learning Trend')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Performance Summary
    success_rate = sum(1 for r in test_rewards if r > 60) / len(test_rewards) * 100
    excellent_rate = sum(1 for r in test_rewards if r > 80) / len(test_rewards) * 100
    
    stats_text = f"""
    ✅ STABLE VERSION PERFORMANCE SUMMARY
    
    ┌─────────────────────────────────────┐
    │         TRAINING METRICS            │
    ├─────────────────────────────────────┤
    │ Best Training Reward:  {max(rewards):5.1f}       │
    │ Best Episode:          {best_episode+1:5d}       │
    │ Final Avg (30):        {np.mean(rewards[-30:]):5.1f}       │
    │ Total Episodes:        {len(rewards):5d}       │
    └─────────────────────────────────────┘
    
    ┌─────────────────────────────────────┐
    │          TEST METRICS               │
    ├─────────────────────────────────────┤
    │ Average:               {np.mean(test_rewards):5.1f}       │
    │ Std Deviation:         {np.std(test_rewards):5.1f}       │
    │ Best Test:             {max(test_rewards):5.1f}       │
    │ Worst Test:            {min(test_rewards):5.1f}       │
    └─────────────────────────────────────┘
    
    ┌─────────────────────────────────────┐
    │          QUALITY METRICS            │
    ├─────────────────────────────────────┤
    │ Success Rate (>60):    {success_rate:5.0f}%         │
    │ Excellent Rate (>80):  {excellent_rate:5.0f}%         │
    └─────────────────────────────────────┘
    
    📌 Key Improvements:
    • Aggressive epsilon decay (0.97)
    • Stop training at best model
    • Regular test monitoring
    • Early stopping on stability
    """
    
    axes[1, 2].text(0.05, 0.5, stats_text, fontsize=9, verticalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    axes[1, 2].axis('off')
    
    plt.suptitle('AI Gardener - Stable Training Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('stable_training_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Stable results plot saved as 'stable_training_results.png'")
    plt.show()

if __name__ == "__main__":
    print("="*70)
    print("✅ STABLE AI GARDENER - PROVEN APPROACH")
    print("="*70)
    print("\nThis version implements the proven fixes:")
    print("  ✓ Aggressive epsilon decay (0.97) - reaches 0.05 by episode 100")
    print("  ✓ Stop training at best model (prevents forgetting)")
    print("  ✓ Regular test monitoring during training")
    print("  ✓ Early stopping when model stabilizes")
    print("  ✓ Based on your best performing improved version")
    print("="*70)
    
    agent, rewards, test_rewards, test_avg = train_stable_gardener(
        grid_size=3,
        max_episodes=350,
        save_path="models/stable_gardener.pth"
    )
    
    print("\n" + "="*70)
    print("🎉 STABLE TRAINING COMPLETE!")
    print("="*70)
    
    if test_avg >= 60:
        print("\n✅ EXCELLENT! Your AI Gardener is now stable and generalizes well!")
        print("   Run the app to see it in action: streamlit run main.py")
    elif test_avg >= 45:
        print("\n👍 GOOD! Your AI Gardener has learned reasonable strategies.")
        print("   Run the app to see it in action: streamlit run main.py")
    else:
        print("\n⚠️ The model still needs improvement. Let's analyze the results together.")