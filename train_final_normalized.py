# train_final_normalized.py - CORRECTED VERSION
import sys
import os
sys.path.append('.')

import torch
import numpy as np
from environment.garden_env_normalized import GardenEnvNormalized
from agents.dqn_agent_fixed import DQNAgentFixed
import matplotlib.pyplot as plt
import time
import copy

def train_normalized_gardener(grid_size=3, max_episodes=500, save_path="models/final_normalized.pth"):
    """Proper training with normalized rewards"""
    
    print("="*70)
    print("✅ NORMALIZED AI GARDENER - CORRECT REWARD SCALING")
    print("="*70)
    print(f"Garden Size: {grid_size}x{grid_size}")
    print(f"Max Episodes: {max_episodes}")
    print("="*70)
    
    env = GardenEnvNormalized(grid_size=grid_size)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"\n📊 Environment Info:")
    print(f"  - State size: {state_size}")
    print(f"  - Action size: {action_size}")
    print(f"  - Max steps: {env.max_steps}")
    print(f"  - Expected reward range: 0-100 per episode")
    
    device = torch.device("cpu")
    agent = DQNAgentFixed(state_size, action_size, device, learning_rate=0.0005)
    
    # Training metrics
    rewards_history = []
    epsilon_history = []
    test_rewards_history = []
    
    best_reward = -float('inf')
    best_episode = 0
    best_model_state = None
    
    print("\n🚀 Starting Normalized Training...\n")
    print("="*95)
    print(f"{'Episode':<10} {'Reward':<10} {'Avg(30)':<10} {'Best':<10} {'Epsilon':<10} {'Test(10)':<12} {'Status':<15}")
    print("="*95)
    
    start_time = time.time()
    improvement_window = []
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(env.max_steps):
            action = agent.act(state, eval_mode=False)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
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
        
        # Test every 50 episodes with proper thresholds
        test_avg = None
        if (episode + 1) % 50 == 0:
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
        
        # Save best model based on test performance (better metric)
        if test_avg is not None and test_avg > best_reward:
            best_reward = test_avg
            best_episode = episode
            best_model_state = {
                'q_network': copy.deepcopy(agent.q_network.state_dict()),
                'target_network': copy.deepcopy(agent.target_network.state_dict()),
                'optimizer': copy.deepcopy(agent.optimizer.state_dict()),
                'epsilon': agent.epsilon,
                'steps': agent.steps,
                'reward': total_reward,
                'episode': episode,
                'test_avg': test_avg
            }
            status = "🔥 NEW BEST!"
        else:
            status = ""
        
        # Progress reporting
        if (episode + 1) % 50 == 0:
            test_str = f"{test_avg:.1f}" if test_avg is not None else "N/A"
            print(f"{episode+1:<10} {total_reward:<10.2f} {avg_30:<10.2f} "
                  f"{best_reward:<10.2f} {agent.epsilon:<10.4f} {test_str:<12} {status}")
        
        # Early stopping if no improvement for 100 episodes (but only after 200)
        if episode > 200 and len(test_rewards_history) > 4:
            recent_improvement = test_rewards_history[-1] - test_rewards_history[-2] if len(test_rewards_history) > 1 else 0
            if recent_improvement < 0.5 and test_rewards_history[-1] < best_reward * 0.9:
                print(f"\n⚠️ Plateau detected at episode {episode+1}")
                break
    
    # Load best model
    print(f"\n📌 Loading best model from episode {best_episode + 1}")
    if best_model_state:
        agent.q_network.load_state_dict(best_model_state['q_network'])
        agent.target_network.load_state_dict(best_model_state['target_network'])
        agent.optimizer.load_state_dict(best_model_state['optimizer'])
        agent.epsilon = best_model_state['epsilon']
        agent.steps = best_model_state['steps']
        best_test = best_model_state.get('test_avg', best_reward)
        print(f"   Best Test Avg: {best_test:.2f}")
    
    os.makedirs("models", exist_ok=True)
    agent.save(save_path)
    
    training_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("✅ Normalized Training Complete!")
    print(f"Training Time: {training_time/60:.1f} minutes")
    print(f"Best Model: {save_path}")
    print("="*70)
    
    # Comprehensive testing with proper thresholds
    print("\n🧪 Testing Best Model (30 episodes)...")
    print("   Thresholds: >60 = GOOD, >80 = EXCELLENT\n")
    
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
        
        # Proper evaluation thresholds
        if test_reward > 80:
            icon = "⭐ EXCELLENT"
        elif test_reward > 60:
            icon = "✓ GOOD"
        elif test_reward > 45:
            icon = "⚠️ OK"
        else:
            icon = "❌ POOR"
        
        print(f"  Test {test+1:2d}: {test_reward:6.2f}  {icon}")
    
    # Statistics with proper interpretation
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
    
    # Generate plots
    generate_final_plots(rewards_history, test_rewards, test_mean, success_rate, excellent_rate)
    
    return agent, rewards_history, test_rewards, test_mean

def generate_final_plots(rewards, test_rewards, test_mean, success_rate, excellent_rate):
    """Generate final comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training Progress
    axes[0, 0].plot(rewards, alpha=0.5, linewidth=0.5, color='blue', label='Episode Reward')
    if len(rewards) > 30:
        moving_avg = np.convolve(rewards, np.ones(30)/30, mode='valid')
        axes[0, 0].plot(range(29, len(rewards)), moving_avg, 'r-', linewidth=2, label='Moving Avg (30)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Normalized Training Progress (Target: 0-100)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=60, color='g', linestyle='--', alpha=0.5, label='Good Threshold (60)')
    axes[0, 0].axhline(y=80, color='gold', linestyle='--', alpha=0.5, label='Excellent Threshold (80)')
    
    # Plot 2: Test Results
    colors = ['green' if r > 60 else 'orange' if r > 45 else 'red' for r in test_rewards]
    axes[0, 1].bar(range(1, len(test_rewards)+1), test_rewards, color=colors)
    axes[0, 1].axhline(y=test_mean, color='blue', linestyle='--', linewidth=2, label=f'Avg: {test_mean:.1f}')
    axes[0, 1].axhline(y=60, color='g', linestyle=':', alpha=0.7, label='Good (60)')
    axes[0, 1].set_xlabel('Test Episode')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].set_title('Final Test Results (30 Episodes)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Reward Distribution
    axes[1, 0].hist(test_rewards, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1, 0].axvline(x=test_mean, color='r', linestyle='--', linewidth=2, label=f'Mean: {test_mean:.1f}')
    axes[1, 0].axvline(x=60, color='g', linestyle=':', alpha=0.7, label='Good (60)')
    axes[1, 0].set_xlabel('Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Test Reward Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Performance Summary
    stats_text = f"""
    ✅ NORMALIZED AI GARDENER - FINAL RESULTS
    
    ┌─────────────────────────────────────┐
    │         NORMALIZED REWARDS          │
    │         (Target: 0-100)             │
    ├─────────────────────────────────────┤
    │ Test Average:    {test_mean:.2f} ± {np.std(test_rewards):.2f}    │
    │ Best Test:       {max(test_rewards):.2f}            │
    │ Worst Test:      {min(test_rewards):.2f}            │
    └─────────────────────────────────────┘
    
    ┌─────────────────────────────────────┐
    │          QUALITY METRICS            │
    ├─────────────────────────────────────┤
    │ Success Rate (>60):  {success_rate:.0f}%                │
    │ Excellent Rate (>80): {excellent_rate:.0f}%                │
    └─────────────────────────────────────┘
    
    📊 Interpretation:
    • Good: >60 reward (real learning)
    • Excellent: >80 reward (expert level)
    • Success Rate shows true generalization
    
    🔧 Features:
    ✓ Normalized rewards (0-100 range)
    ✓ Double DQN
    ✓ Random initial states
    ✓ Proper epsilon (min 0.02)
    """
    
    axes[1, 1].text(0.05, 0.5, stats_text, fontsize=10, verticalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[1, 1].axis('off')
    
    plt.suptitle('Normalized AI Gardener - Correct Performance Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('normalized_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Normalized results saved as 'normalized_results.png'")
    plt.show()

if __name__ == "__main__":
    print("="*70)
    print("🎯 NORMALIZED AI GARDENER - FINAL CORRECT VERSION")
    print("="*70)
    print("\nThis version has FIXED all issues:")
    print("  ✓ Normalized rewards (0-100 range, not 500+)")
    print("  ✓ Proper evaluation thresholds (>60 = GOOD, >80 = EXCELLENT)")
    print("  ✓ Higher epsilon min (0.02) for exploration")
    print("  ✓ Random initial states to prevent memorization")
    print("  ✓ Test-based best model selection")
    print("="*70)
    
    agent, rewards, test_rewards, test_avg = train_normalized_gardener(
        grid_size=3,
        max_episodes=500,
        save_path="models/final_normalized.pth"
    )
    
    print("\n" + "="*70)
    print("🎉 FINAL CORRECTED TRAINING COMPLETE!")
    print("="*70)
    
    if test_avg >= 65:
        print("\n✅ EXCELLENT! Your model shows strong RL performance!")
        print("   The rewards are now in the proper 0-100 range.")
    elif test_avg >= 50:
        print("\n👍 GOOD! Your model is learning effectively.")
        print("   Results are now interpretable and meaningful.")
    else:
        print("\n⚠️ Model is still learning. Consider running more episodes.")
    
    print("\n📊 To see real performance, look at Success Rate (>60):")
    print("   This shows true generalization capability.")