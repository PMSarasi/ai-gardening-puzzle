# train_advanced.py - FINAL UPGRADED VERSION
import sys
import os
sys.path.append('.')

import torch
import numpy as np
from environment.garden_env_advanced import GardenEnv
from agents.dqn_agent_advanced import DQNAgentAdvanced
import matplotlib.pyplot as plt
import time
import copy

def train_advanced_gardener(grid_size=3, max_episodes=350, save_path="models/advanced_gardener.pth"):
    """Advanced training with Double DQN + Enhanced Rewards"""
    
    print("="*70)
    print("🚀 ADVANCED AI GARDENER - DOUBLE DQN + ENHANCED REWARDS")
    print("="*70)
    print(f"Garden Size: {grid_size}x{grid_size}")
    print(f"Max Episodes: {max_episodes}")
    print("="*70)
    
    env = GardenEnv(grid_size=grid_size)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"\n📊 Environment Info:")
    print(f"  - State size: {state_size}")
    print(f"  - Action size: {action_size}")
    print(f"  - Max steps: {env.max_steps}")
    
    device = torch.device("cpu")
    agent = DQNAgentAdvanced(state_size, action_size, device, learning_rate=0.0005)
    
    # Training metrics
    rewards_history = []
    epsilon_history = []
    test_rewards_history = []
    
    best_reward = -float('inf')
    best_episode = 0
    best_model_state = None
    
    print("\n🚀 Starting Advanced Training...\n")
    print("="*95)
    print(f"{'Episode':<10} {'Reward':<10} {'Avg(30)':<10} {'Best':<10} {'Epsilon':<10} {'Test(10)':<12} {'Status':<15}")
    print("="*95)
    
    start_time = time.time()
    good_episodes = 0
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(env.max_steps):
            # Get action mask (optional)
            action_mask = env.get_action_mask()
            action = agent.act(state, eval_mode=False, valid_actions_mask=action_mask)
            
            next_state, reward, done, truncated, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()
            
            total_reward += reward
            state = next_state
            
            if done or truncated:
                break
        
        rewards_history.append(total_reward)
        epsilon_history.append(agent.epsilon)
        
        # Moving average
        if len(rewards_history) >= 30:
            avg_30 = np.mean(rewards_history[-30:])
        else:
            avg_30 = total_reward
        
        # Test every 50 episodes
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
        
        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = episode
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
            if total_reward > 50:
                good_episodes += 1
        
        # Progress reporting
        if (episode + 1) % 50 == 0:
            test_str = f"{test_avg:.1f}" if test_avg is not None else "N/A"
            print(f"{episode+1:<10} {total_reward:<10.2f} {avg_30:<10.2f} "
                  f"{best_reward:<10.2f} {agent.epsilon:<10.4f} {test_str:<12} {status}")
        
        # Early stopping when stable and good
        if episode > 150 and good_episodes > 20:
            print(f"\n✅ Model stabilized at episode {episode+1}")
            break
    
    # Load best model
    print(f"\n📌 Loading best model from episode {best_episode + 1} (Reward: {best_reward:.2f})")
    if best_model_state:
        agent.q_network.load_state_dict(best_model_state['q_network'])
        agent.target_network.load_state_dict(best_model_state['target_network'])
        agent.optimizer.load_state_dict(best_model_state['optimizer'])
        agent.epsilon = best_model_state['epsilon']
        agent.steps = best_model_state['steps']
    
    os.makedirs("models", exist_ok=True)
    agent.save(save_path)
    
    training_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("✅ Advanced Training Complete!")
    print(f"Training Time: {training_time/60:.1f} minutes")
    print(f"Best Model: {save_path} (Episode {best_episode + 1})")
    print("="*70)
    
    # Comprehensive testing
    print("\n🧪 Testing Advanced Model (30 episodes)...")
    
    test_rewards = []
    for test in range(30):
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
    success_rate = sum(1 for r in test_rewards if r > 60) / len(test_rewards) * 100
    excellent_rate = sum(1 for r in test_rewards if r > 80) / len(test_rewards) * 100
    
    print(f"\n📊 Test Summary:")
    print(f"  - Average: {test_mean:.2f} ± {test_std:.2f}")
    print(f"  - Best: {max(test_rewards):.2f}")
    print(f"  - Worst: {min(test_rewards):.2f}")
    print(f"  - Success Rate (>60): {success_rate:.0f}%")
    print(f"  - Excellent Rate (>80): {excellent_rate:.0f}%")
    
    # Generate comparison plots
    generate_comparison_plots(rewards_history, test_rewards, best_reward, test_mean)
    
    return agent, rewards_history, test_rewards, test_mean

def generate_comparison_plots(rewards, test_rewards, best_reward, test_mean):
    """Generate comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training Progress
    axes[0, 0].plot(rewards, alpha=0.5, linewidth=0.5, color='blue', label='Episode Reward')
    if len(rewards) > 30:
        moving_avg = np.convolve(rewards, np.ones(30)/30, mode='valid')
        axes[0, 0].plot(range(29, len(rewards)), moving_avg, 'r-', linewidth=2, label='Moving Avg (30)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Advanced Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Test Results Distribution
    colors = ['green' if r > 60 else 'orange' if r > 40 else 'red' for r in test_rewards]
    axes[0, 1].bar(range(1, len(test_rewards)+1), test_rewards, color=colors)
    axes[0, 1].axhline(y=test_mean, color='blue', linestyle='--', linewidth=2, label=f'Avg: {test_mean:.1f}')
    axes[0, 1].set_xlabel('Test Episode')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].set_title('Advanced Model Test Results')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Reward Distribution
    axes[1, 0].hist(test_rewards, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1, 0].axvline(x=test_mean, color='r', linestyle='--', linewidth=2, label=f'Mean: {test_mean:.1f}')
    axes[1, 0].set_xlabel('Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Test Reward Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Performance Summary
    success_rate = sum(1 for r in test_rewards if r > 60) / len(test_rewards) * 100
    excellent_rate = sum(1 for r in test_rewards if r > 80) / len(test_rewards) * 100
    
    stats_text = f"""
    🚀 ADVANCED MODEL PERFORMANCE
    
    ┌─────────────────────────────────────┐
    │         KEY IMPROVEMENTS            │
    ├─────────────────────────────────────┤
    │ ✓ Double DQN Implementation         │
    │ ✓ Enhanced Reward Shaping           │
    │ ✓ Lower Epsilon Min (0.005)         │
    │ ✓ Larger Batch Size (128)           │
    │ ✓ Improved Network Architecture     │
    └─────────────────────────────────────┘
    
    ┌─────────────────────────────────────┐
    │          TEST RESULTS               │
    ├─────────────────────────────────────┤
    │ Average:       {test_mean:.2f} ± {np.std(test_rewards):.2f}    │
    │ Best:          {max(test_rewards):.2f}            │
    │ Success Rate:  {success_rate:.0f}%                │
    │ Excellent:     {excellent_rate:.0f}%                │
    └─────────────────────────────────────┘
    
    📌 Expected: 65-75 avg with 60%+ success rate
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    axes[1, 1].axis('off')
    
    plt.suptitle('Advanced AI Gardener - Double DQN Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('advanced_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Advanced results saved as 'advanced_results.png'")
    plt.show()

if __name__ == "__main__":
    print("="*70)
    print("🎯 ADVANCED AI GARDENER - FINAL UPGRADE")
    print("="*70)
    print("\nThis version includes ALL performance upgrades:")
    print("  ✓ Double DQN (reduces overestimation)")
    print("  ✓ Enhanced reward shaping (clear guidance)")
    print("  ✓ Lower epsilon (0.005 for fewer mistakes)")
    print("  ✓ Larger batch size (128 for stability)")
    print("  ✓ Improved network architecture")
    print("="*70)
    
    agent, rewards, test_rewards, test_avg = train_advanced_gardener(
        grid_size=3,
        max_episodes=350,
        save_path="models/advanced_gardener.pth"
    )
    
    print("\n" + "="*70)
    print("🎉 ADVANCED TRAINING COMPLETE!")
    print("="*70)
    
    if test_avg >= 65:
        print("\n✅ EXCELLENT! Your model has reached advanced performance levels!")
    elif test_avg >= 55:
        print("\n👍 GOOD! Significant improvement over the stable version!")
    else:
        print("\n⚠️ The model needs more training. Consider increasing episodes to 500.")