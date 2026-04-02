import numpy as np
from tqdm import tqdm
import torch
import time
import matplotlib.pyplot as plt

class GardenerTrainer:
    def __init__(self, env, agent, visualizer=None):
        self.env = env
        self.agent = agent
        self.visualizer = visualizer
        
        # Training metrics
        self.rewards_history = []
        self.losses_history = []
        self.epsilon_history = []
        self.stats_history = []
        
    def train(self, episodes=500, render_every=10, save_every=50, save_path="models/trained_gardener.pth"):
        """Train the agent"""
        progress_bar = tqdm(range(episodes), desc="Training Gardener")
        
        for episode in progress_bar:
            state, _ = self.env.reset()
            total_reward = 0
            episode_losses = []
            episode_steps = 0
            
            for step in range(self.env.max_steps):
                # Agent selects action
                action = self.agent.act(state)
                
                # Execute action
                next_state, reward, done, truncated, _ = self.env.step(action)
                
                # Store experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Train agent
                loss = self.agent.replay()
                if loss > 0:
                    episode_losses.append(loss)
                
                total_reward += reward
                state = next_state
                episode_steps += 1
                
                if done:
                    break
            
            # Record metrics
            self.rewards_history.append(total_reward)
            if episode_losses:
                self.losses_history.append(np.mean(episode_losses))
            else:
                self.losses_history.append(0)
            self.epsilon_history.append(self.agent.epsilon)
            
            # Get garden stats
            stats = self.env.get_garden_stats()
            stats['total_reward'] = total_reward
            stats['episode'] = episode
            stats['steps'] = episode_steps
            self.stats_history.append(stats)
            
            # Calculate moving averages
            avg_reward = np.mean(self.rewards_history[-100:]) if len(self.rewards_history) >= 100 else np.mean(self.rewards_history)
            avg_loss = np.mean(self.losses_history[-100:]) if len(self.losses_history) >= 100 else np.mean(self.losses_history)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Reward': f'{total_reward:.2f}',
                'Avg Reward': f'{avg_reward:.2f}',
                'Epsilon': f'{self.agent.epsilon:.3f}',
                'Health': f'{stats["avg_health"]:.1f}'
            })
            
            # Save model periodically
            if (episode + 1) % save_every == 0:
                self.agent.save(save_path)
                progress_bar.write(f"Model saved to {save_path}")
            
            # Render visualization if visualizer exists
            if self.visualizer and episode % render_every == 0:
                # Create training plot
                fig = self._create_training_plot(episode)
                if hasattr(self.visualizer, 'update_training_plot'):
                    self.visualizer.update_training_plot(fig)
                else:
                    # Fallback to printing stats
                    progress_bar.write(f"Episode {episode}: Reward={total_reward:.2f}, Avg Reward={avg_reward:.2f}, Health={stats['avg_health']:.1f}")
        
        # Save final model
        self.agent.save(save_path)
        return self.rewards_history, self.losses_history
    
    def evaluate(self, episodes=10, render=False):
        """Evaluate the agent's performance"""
        eval_rewards = []
        eval_stats = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            
            for step in range(self.env.max_steps):
                action = self.agent.act(state, eval_mode=True)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = next_state
                
                if render:
                    self.env.render()
                
                if done:
                    break
            
            eval_rewards.append(total_reward)
            eval_stats.append(self.env.get_garden_stats())
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards),
            'rewards': eval_rewards,
            'stats': eval_stats
        }
    
    def _create_training_plot(self, episode):
        """Create matplotlib figure for training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Reward plot
        axes[0, 0].plot(self.rewards_history, alpha=0.5, label='Episode Reward')
        if len(self.rewards_history) > 10:
            moving_avg = np.convolve(self.rewards_history, np.ones(10)/10, mode='valid')
            axes[0, 0].plot(range(9, len(self.rewards_history)), moving_avg, 'r-', label='Moving Avg (10)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[0, 1].plot(self.losses_history, 'g-', alpha=0.5)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Health plot
        health_values = [s['avg_health'] for s in self.stats_history]
        axes[1, 0].plot(health_values, 'b-', alpha=0.5)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Average Health')
        axes[1, 0].set_title('Plant Health')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Growth plot
        growth_values = [s['avg_growth'] for s in self.stats_history]
        axes[1, 1].plot(growth_values, 'm-', alpha=0.5)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Growth Stage')
        axes[1, 1].set_title('Plant Growth')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def compare_policies(self, episodes=10):
        """Compare random policy vs trained policy"""
        print("Comparing Random Policy vs Trained Policy...")
        
        # Random policy evaluation
        random_rewards = []
        for _ in range(episodes):
            state = self.env.reset()
            total_reward = 0
            for _ in range(self.env.max_steps):
                action = np.random.randint(self.env.action_space.n)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = next_state
                if done:
                    break
            random_rewards.append(total_reward)
        
        # Trained policy evaluation
        trained_rewards = []
        for _ in range(episodes):
            state = self.env.reset()
            total_reward = 0
            for _ in range(self.env.max_steps):
                action = self.agent.act(state, eval_mode=True)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = next_state
                if done:
                    break
            trained_rewards.append(total_reward)
        
        comparison = {
            'random_mean': np.mean(random_rewards),
            'random_std': np.std(random_rewards),
            'trained_mean': np.mean(trained_rewards),
            'trained_std': np.std(trained_rewards),
            'improvement': np.mean(trained_rewards) - np.mean(random_rewards)
        }
        
        print(f"\nComparison Results:")
        print(f"Random Policy: {comparison['random_mean']:.2f} ± {comparison['random_std']:.2f}")
        print(f"Trained Policy: {comparison['trained_mean']:.2f} ± {comparison['trained_std']:.2f}")
        print(f"Improvement: {comparison['improvement']:.2f}")
        
        return comparison