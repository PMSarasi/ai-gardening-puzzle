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
        else:
            self.water_effect = 1.2
            self.sun_effect = -0.5
            self.temperature = 18.0
    
    def step(self):
        self._update_weather()
        return self.get_weather_state()
    
    def get_weather_state(self):
        return (self.water_effect, self.sun_effect, self.temperature)

class Plant:
    def __init__(self, plant_type, x, y):
        self.type = plant_type
        self.x = x
        self.y = y
        self.water = 3.0
        self.sunlight = 3.0
        self.soil = 3.0
        self.growth_stage = 0
        self.health = 100
        self.age = 0
        self.last_action = None
        
    def update(self, weather, action=None):
        # Apply weather effects
        self.water += weather.water_effect
        self.sunlight += weather.sun_effect
        
        # Apply gardener action
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
        
        self._calculate_growth()
        self.age += 1
        self.last_action = action
        
    def _calculate_growth(self):
        growth_factor = (self.water/5 + self.sunlight/5 + self.soil/5) / 3
        health_change = growth_factor * 5 - 2
        self.health = min(100, max(0, self.health + health_change))
        
        if self.health > 70 and self.growth_stage < 3:
            self.growth_stage += 0.02 * growth_factor
        elif self.health < 30 and self.growth_stage > 0:
            self.growth_stage -= 0.01
        
        self.growth_stage = np.clip(self.growth_stage, 0, 3)
    
    def get_state_vector(self):
        return [
            self.type.value / 2,
            self.water / 5,
            self.sunlight / 5,
            self.soil / 5,
            self.growth_stage / 3,
            self.health / 100
        ]

class GardenEnvNormalized(gym.Env):
    def __init__(self, grid_size=3):
        super().__init__()
        self.grid_size = grid_size
        self.num_plants = grid_size * grid_size
        
        self.action_space = spaces.Discrete(4 * self.num_plants)
        obs_size = (self.num_plants * 6) + 3
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )
        
        self.plants = []
        self.weather = None
        self.current_step = 0
        self.max_steps = 100
        self.episode_reward = 0
        self.last_action = None
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Add randomness to initial states (prevents memorization)
        self.plants = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                plant_type = np.random.choice(list(PlantType), p=[0.4, 0.4, 0.2])
                plant = Plant(plant_type, i, j)
                # Random initial variation (5-10% variation)
                plant.water = np.clip(3.0 + np.random.randn() * 0.3, 2, 4)
                plant.soil = np.clip(3.0 + np.random.randn() * 0.3, 2, 4)
                plant.health = np.clip(100 + np.random.randn() * 5, 80, 100)
                self.plants.append(plant)
        
        self.weather = Weather()
        self.current_step = 0
        self.episode_reward = 0
        self.last_action = None
        
        return self._get_observation(), {}
    
    def step(self, action):
        plant_idx = action // 4
        action_type = action % 4
        
        actions = ["water", "fertilize", "prune", "wait"]
        selected_action = actions[action_type] if plant_idx < len(self.plants) else "wait"
        
        self.weather.step()
        
        # Apply action
        if plant_idx < len(self.plants):
            self.plants[plant_idx].update(self.weather, selected_action)
        
        # Update other plants
        for i, plant in enumerate(self.plants):
            if i != plant_idx:
                plant.update(self.weather, None)
        
        # Calculate NORMALIZED reward (0-100 range)
        reward = self._calculate_reward_normalized(plant_idx, selected_action)
        
        self.episode_reward += reward
        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False
        
        # Episode completion bonus (normalized)
        if done:
            reward += self._get_episode_bonus_normalized()
        
        self.last_action = selected_action
        
        return self._get_observation(), reward, done, truncated, {}
    
    def _calculate_reward_normalized(self, plant_idx, action):
        """NORMALIZED reward function (target: 0-100 per episode)"""
        if plant_idx >= len(self.plants):
            return -0.05
            
        plant = self.plants[plant_idx]
        reward = 0.0  # Start at 0, not large positive
        
        # ============= HEALTH REWARDS (Small, incremental) =============
        if plant.health > 85:
            reward += 0.08  # Excellent health
        elif plant.health > 70:
            reward += 0.05  # Good health
        elif plant.health > 55:
            reward += 0.02  # Decent health
        elif plant.health < 35:
            reward -= 0.10  # Unhealthy
        elif plant.health < 20:
            reward -= 0.20  # Critical!
        
        # ============= GROWTH REWARDS (Progressive) =============
        # Growth stage reward (max 0.6 over full growth)
        reward += plant.growth_stage * 0.05
        
        # Milestone bonuses (one-time)
        if plant.growth_stage >= 0.8 and plant.growth_stage < 0.9:
            reward += 0.3  # Reached seedling
        elif plant.growth_stage >= 1.8 and plant.growth_stage < 1.9:
            reward += 0.5  # Reached small plant
        elif plant.growth_stage >= 2.8:
            reward += 0.8  # Reached mature plant!
        
        # ============= RESOURCE MANAGEMENT =============
        # Optimal water (2.5-3.5 is ideal)
        if 2.5 < plant.water < 3.5:
            reward += 0.06
        elif plant.water < 1.5:
            reward -= 0.08
        elif plant.water > 4.5:
            reward -= 0.05
        
        # Optimal soil
        if 2.5 < plant.soil < 3.5:
            reward += 0.06
        elif plant.soil < 1.5:
            reward -= 0.08
        elif plant.soil > 4.5:
            reward -= 0.05
        
        # ============= ACTION GUIDANCE (Clear signals) =============
        if action == "water":
            if plant.water < 2.2:
                reward += 0.25  # Good! Plant needed water
            elif plant.water < 3.0:
                reward += 0.10  # Decent timing
            elif plant.water > 4.2:
                reward -= 0.15  # Bad! Overwatering
            else:
                reward += 0.02
        
        elif action == "fertilize":
            if plant.soil < 2.2:
                reward += 0.25  # Good! Plant needed nutrients
            elif plant.soil < 3.0:
                reward += 0.10  # Decent timing
            elif plant.soil > 4.2:
                reward -= 0.15  # Bad! Over-fertilizing
            else:
                reward += 0.02
        
        elif action == "prune":
            if plant.health < 55:
                reward += 0.30  # Great! Saved unhealthy plant
            elif plant.health < 70:
                reward += 0.12  # Good preventive care
            elif plant.health > 88:
                reward -= 0.05  # Unnecessary pruning
            else:
                reward += 0.03
        
        elif action == "wait":
            # Penalty for waiting when action is needed
            if plant.water < 2.2 or plant.soil < 2.2 or plant.health < 55:
                reward -= 0.12
            # Small reward for waiting when everything is optimal
            elif plant.water > 3.0 and plant.soil > 3.0 and plant.health > 80:
                reward += 0.05
        
        # ============= PENALTIES FOR REPETITION =============
        if plant.last_action == action and action != "wait":
            reward -= 0.03
        
        # Clip reward to reasonable range
        reward = np.clip(reward, -0.5, 0.8)
        
        return reward
    
    def _get_episode_bonus_normalized(self):
        """Normalized episode bonus (max 10 points)"""
        avg_health = np.mean([p.health for p in self.plants])
        avg_growth = np.mean([p.growth_stage for p in self.plants])
        
        bonus = 0.0
        
        # Health bonus (max 5)
        if avg_health > 85:
            bonus += 5.0
        elif avg_health > 75:
            bonus += 3.0
        elif avg_health > 65:
            bonus += 1.5
        elif avg_health > 55:
            bonus += 0.5
        
        # Growth bonus (max 5)
        if avg_growth > 2.5:
            bonus += 5.0
        elif avg_growth > 2.0:
            bonus += 3.0
        elif avg_growth > 1.5:
            bonus += 1.5
        elif avg_growth > 1.0:
            bonus += 0.5
        
        # Perfect garden bonus
        if avg_health > 85 and avg_growth > 2.5:
            bonus += 2.0
        
        return bonus
    
    def _get_observation(self):
        obs = []
        for plant in self.plants:
            obs.extend(plant.get_state_vector())
        
        water_effect, sun_effect, temperature = self.weather.get_weather_state()
        obs.extend([
            (water_effect + 1.5) / 3.0,
            (sun_effect + 1.0) / 2.0,
            temperature / 40.0
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def get_action_mask(self):
        return [1] * self.action_space.n
    
    def get_garden_stats(self):
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