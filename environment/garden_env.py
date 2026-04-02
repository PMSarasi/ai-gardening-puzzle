import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from enum import Enum

class PlantType(Enum):
    FLOWER = 0
    VEGETABLE = 1
    TREE = 2

class Weather:
    def __init__(self):
        self.water_effect = 0.0
        self.sun_effect = 0.0
        self.temperature = 25.0
        self.weather_type = 'sunny'
        self._update_weather()
    
    def _update_weather(self):
        """Randomly generate weather conditions with balanced probabilities"""
        weather_types = ['sunny', 'rainy', 'cloudy', 'storm']
        self.weather_type = np.random.choice(weather_types, p=[0.4, 0.3, 0.2, 0.1])
        
        if self.weather_type == 'sunny':
            self.water_effect = -0.3
            self.sun_effect = 0.5
            self.temperature = 28.0
        elif self.weather_type == 'rainy':
            self.water_effect = 0.8
            self.sun_effect = -0.2
            self.temperature = 20.0
        elif self.weather_type == 'cloudy':
            self.water_effect = 0.1
            self.sun_effect = -0.1
            self.temperature = 22.0
        else:  # storm
            self.water_effect = 1.2
            self.sun_effect = -0.5
            self.temperature = 18.0
    
    def step(self):
        """Advance weather to next day"""
        self._update_weather()
        return self.get_weather_state()
    
    def get_weather_state(self):
        """Return weather as a tuple for observation"""
        return (self.water_effect, self.sun_effect, self.temperature)

class Plant:
    def __init__(self, plant_type, x, y):
        self.type = plant_type
        self.x = x
        self.y = y
        self.water = 3.0  # 0-5
        self.sunlight = 3.0  # 0-5
        self.soil = 3.0  # 0-5
        self.growth_stage = 0  # 0-3 (seedling, small, grown, mature)
        self.health = 100  # 0-100
        self.age = 0
        
    def update(self, weather, action=None):
        """Update plant state based on weather and gardener action"""
        # Apply weather effects
        self.water += weather.water_effect
        self.sunlight += weather.sun_effect
        
        # Apply gardener action if any
        if action == "water":
            self.water += 2
        elif action == "fertilize":
            self.soil += 1.5
        elif action == "prune":
            self.health += 10
            self.growth_stage = min(3, self.growth_stage + 0.2)
        
        # Clamp values
        self.water = np.clip(self.water, 0, 5)
        self.sunlight = np.clip(self.sunlight, 0, 5)
        self.soil = np.clip(self.soil, 0, 5)
        
        # Calculate growth
        self._calculate_growth()
        
        # Age the plant
        self.age += 1
        
    def _calculate_growth(self):
        """Calculate growth based on resources"""
        growth_factor = (self.water/5 + self.sunlight/5 + self.soil/5) / 3
        # Health changes based on growth factor
        health_change = growth_factor * 5 - 2
        self.health = min(100, max(0, self.health + health_change))
        
        # Advance growth stage if healthy
        if self.health > 70 and self.growth_stage < 3:
            self.growth_stage += 0.02 * growth_factor
        elif self.health < 30 and self.growth_stage > 0:
            self.growth_stage -= 0.01
        
        self.growth_stage = np.clip(self.growth_stage, 0, 3)
    
    def get_state_vector(self):
        """Get plant state as normalized vector"""
        return [
            self.type.value / 2,  # normalize plant type (0-1)
            self.water / 5,
            self.sunlight / 5,
            self.soil / 5,
            self.growth_stage / 3,
            self.health / 100
        ]

class GardenEnv(gym.Env):
    def __init__(self, grid_size=3):  # Changed default to 3 for faster training
        super().__init__()
        self.grid_size = grid_size
        self.num_plants = grid_size * grid_size
        
        # Action space: for each plant (water, fertilize, prune, wait)
        self.action_space = spaces.Discrete(4 * self.num_plants)
        
        # Observation space: plant states + weather
        obs_size = (self.num_plants * 6) + 3
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )
        
        self.plants = []
        self.weather = None
        self.current_step = 0
        self.max_steps = 100
        
        # Reward tracking
        self.reward_scale = 1.0
        self.episode_reward = 0
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        """Reset the garden for a new episode"""
        super().reset(seed=seed)
        
        self.plants = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                plant_type = np.random.choice(list(PlantType), p=[0.4, 0.4, 0.2])
                self.plants.append(Plant(plant_type, i, j))
        
        self.weather = Weather()
        self.current_step = 0
        self.episode_reward = 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one action in the environment"""
        # Decode action: which plant and what action
        plant_idx = action // 4
        action_type = action % 4
        
        actions = ["water", "fertilize", "prune", "wait"]
        selected_action = actions[action_type] if plant_idx < len(self.plants) else "wait"
        
        # Apply weather change for this step
        self.weather.step()
        
        # Apply action to specific plant
        if plant_idx < len(self.plants):
            self.plants[plant_idx].update(self.weather, selected_action)
        
        # Update all other plants with just weather
        for i, plant in enumerate(self.plants):
            if i != plant_idx:
                plant.update(self.weather, None)
        
        # Calculate reward
        reward = self._calculate_reward(plant_idx, selected_action)
        
        # Scale reward for better learning
        reward = np.clip(reward, -2, 2) * self.reward_scale
        
        self.episode_reward += reward
        
        # Check if episode is done
        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False
        
        # Add small bonus for completing episode
        if done:
            reward += self._get_episode_completion_bonus()
        
        return self._get_observation(), reward, done, truncated, {}
    
    def _calculate_reward(self, plant_idx, action):
        """Improved reward calculation with better balance"""
        if plant_idx >= len(self.plants):
            return -0.1
            
        plant = self.plants[plant_idx]
        reward = 0
        
        # Health-based reward (more balanced)
        if plant.health > 80:
            reward += 0.3
        elif plant.health > 50:
            reward += 0.1
        elif plant.health < 30:
            reward -= 0.4
        elif plant.health < 20:
            reward -= 0.6
        
        # Growth reward (encourage progress)
        reward += plant.growth_stage * 0.15
        
        # Resource balance reward
        if 2.0 < plant.water < 4.0:
            reward += 0.15
        elif plant.water < 1.5 or plant.water > 4.5:
            reward -= 0.2
            
        if 2.0 < plant.soil < 4.0:
            reward += 0.15
        elif plant.soil < 1.5 or plant.soil > 4.5:
            reward -= 0.2
        
        # Action-specific rewards (more balanced)
        if action == "water":
            if plant.water < 2.0:
                reward += 0.5
            elif plant.water < 3.0:
                reward += 0.2
            elif plant.water > 4.5:
                reward -= 0.3
            else:
                reward += 0.05
                
        elif action == "fertilize":
            if plant.soil < 2.0:
                reward += 0.5
            elif plant.soil < 3.0:
                reward += 0.2
            elif plant.soil > 4.5:
                reward -= 0.3
            else:
                reward += 0.05
                
        elif action == "prune":
            if plant.health < 50:
                reward += 0.6
            elif plant.health < 70:
                reward += 0.3
            elif plant.health > 90:
                reward -= 0.2
            else:
                reward += 0.1
        
        elif action == "wait":
            # Penalty for waiting when action is needed
            if plant.water < 2.0 or plant.soil < 2.0 or plant.health < 50:
                reward -= 0.3
        
        # Bonus for mature plants
        if plant.growth_stage >= 2.9:
            reward += 0.2
        
        return reward
    
    def _get_episode_completion_bonus(self):
        """Calculate bonus based on overall garden health at episode end"""
        avg_health = np.mean([p.health for p in self.plants])
        avg_growth = np.mean([p.growth_stage for p in self.plants])
        
        bonus = 0
        if avg_health > 70:
            bonus += 2.0
        if avg_growth > 2.0:
            bonus += 1.0
        if avg_health > 80 and avg_growth > 2.5:
            bonus += 2.0
            
        return bonus
    
    def _get_observation(self):
        """Create observation vector with proper normalization"""
        obs = []
        for plant in self.plants:
            obs.extend(plant.get_state_vector())
        
        # Add weather info
        water_effect, sun_effect, temperature = self.weather.get_weather_state()
        obs.extend([
            (water_effect + 1.5) / 3.0,  # Normalize to 0-1
            (sun_effect + 1.0) / 2.0,
            temperature / 40.0
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def render(self):
        """Simple text-based rendering for debugging"""
        print(f"\n--- Day {self.current_step} ---")
        print(f"Weather: {self.weather.weather_type}, Temp: {self.weather.temperature:.1f}°C")
        print(f"Episode Reward: {self.episode_reward:.2f}")
        
        for i, plant in enumerate(self.plants):
            if i % self.grid_size == 0:
                print()
            
            # Determine emoji based on plant type and growth
            if plant.growth_stage < 1:
                stage_emoji = '🌱'
            elif plant.growth_stage < 2:
                stage_emoji = '🌿' if plant.type.value == 2 else '🌱'
            elif plant.growth_stage < 3:
                stage_emoji = '🌳' if plant.type.value == 2 else '🌸' if plant.type.value == 0 else '🥕'
            else:
                stage_emoji = '🌻' if plant.type.value == 0 else '🍅' if plant.type.value == 1 else '🌲'
            
            # Health indicator
            health_color = '🟢' if plant.health > 70 else '🟡' if plant.health > 40 else '🔴'
            
            print(f"{stage_emoji}{health_color} ", end='')
        print()
        
        # Show resource levels for first plant as example
        if self.plants:
            p = self.plants[0]
            print(f"Sample plant - Water: {p.water:.1f}/5, Soil: {p.soil:.1f}/5, Health: {p.health:.1f}%")
    
    def get_garden_stats(self):
        """Get overall garden statistics"""
        avg_health = np.mean([p.health for p in self.plants])
        avg_growth = np.mean([p.growth_stage for p in self.plants])
        avg_water = np.mean([p.water for p in self.plants])
        avg_soil = np.mean([p.soil for p in self.plants])
        
        return {
            'avg_health': avg_health,
            'avg_growth': avg_growth,
            'avg_water': avg_water,
            'avg_soil': avg_soil,
            'num_plants': self.num_plants,
            'day': self.current_step,
            'episode_reward': self.episode_reward
        }