import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os

class ForestFireEnv(gym.Env):
    """
    Custom Environment for Forest Fire UAV Path Planning
    - Grid-based forest environment
    - UAVs scan grid cells for fire risk
    - Rewards are given for covering high-risk zones
    """

    def __init__(self, grid_size=10, max_steps=100, uav_battery=100, fire_map=None, reward_weight_coverage=1.0, reward_weight_risk=1.0):
        super(ForestFireEnv, self).__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.uav_battery = uav_battery
        self.fire_map = fire_map
        self.reward_weight_coverage = reward_weight_coverage
        self.reward_weight_risk = reward_weight_risk
        # Action space: 0=Up, 1=Down, 2=Left, 3=Right, 4=Hover
        self.action_space = spaces.Discrete(5)

        # Observation space: [UAV_x, UAV_y, battery] + flattened local risk map (3x3 grid around UAV)
        # This provides more context to the agent about nearby fire risks
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(3 + 9,), 
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Load or generate fire risk map
        self.grid = self._load_fire_risk_map()
        
        # Initialize tracking variables
        self.visited = np.zeros((self.grid_size, self.grid_size))
        self.uav_pos = [np.random.randint(0, self.grid_size), 
                        np.random.randint(0, self.grid_size)]  # Random start position
        self.battery = self.uav_battery
        self.steps = 0
        self.total_coverage = 0
        self.risk_covered = 0
        
        obs = self._get_obs()
        info = {}
        return obs, info

    def _load_fire_risk_map(self):
        """Use the passed fire map if provided, else load or generate"""
        if self.fire_map is not None:
            return self.fire_map
        
        try:
            if os.path.exists("data/fire_risk_predictions.csv"):
                df = pd.read_csv("data/fire_risk_predictions.csv")
                fire_map = df.pivot(index='x', columns='y', values='predicted_risk').to_numpy()
                return fire_map
            else:
                print("🔥 Warning: fire_risk_predictions.csv not found. Falling back to random map.")
                return np.random.rand(self.grid_size, self.grid_size)
        except Exception as e:
            print(f"Error loading fire risk map: {e}")
            return np.random.rand(self.grid_size, self.grid_size)


    def _get_obs(self):
        """Get observation: UAV position, battery level, and local fire risk map"""
        x, y = self.uav_pos
        
        # Get local 3x3 grid around UAV (with padding for edges)
        local_map = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                map_x = min(max(x + (i-1), 0), self.grid_size-1)
                map_y = min(max(y + (j-1), 0), self.grid_size-1)
                local_map[i, j] = self.grid[map_x, map_y]
        
        # Flatten the local map
        local_map_flat = local_map.flatten()
        
        # Combine position, battery and local map
        obs = np.concatenate([
            [x / self.grid_size, y / self.grid_size, self.battery / self.uav_battery],
            local_map_flat
        ]).astype(np.float32)
        
        return obs

    def step(self, action):
        x, y = self.uav_pos
        reward = 0

        if isinstance(action, np.ndarray):
            action = np.argmax(action)  # For DDPG/continuous actions

        # Move UAV based on action
        if action == 0 and x > 0:  # Up
            x -= 1
        elif action == 1 and x < self.grid_size - 1:  # Down
            x += 1
        elif action == 2 and y > 0:  # Left
            y -= 1
        elif action == 3 and y < self.grid_size - 1:  # Right
            y += 1
        elif action == 4:  # Hover
            pass

        # Update UAV position
        self.uav_pos = [x, y]
        fire_risk = self.grid[x][y]

        # Calculate reward based on fire risk and whether cell was visited
        if self.visited[x][y] == 0:
            self.total_coverage += 1
            self.risk_covered += fire_risk

            # Base reward depending on fire risk level
            if fire_risk > 0.7:
                risk_reward = 15.0  # High risk
            elif fire_risk > 0.4:
                risk_reward = 7.0   # Medium risk
            else:
                risk_reward = 2.0   # Low risk

            # Apply weighted reward for fire risk
            reward += self.reward_weight_risk * risk_reward

            # Weighted coverage bonus
            coverage_percentage = self.total_coverage / (self.grid_size * self.grid_size)
            reward += self.reward_weight_coverage * (coverage_percentage * 5.0)

        else:
            reward -= 2.0  # Penalty for revisiting

        # Mark as visited
        self.visited[x][y] = 1

        # Update battery and steps
        self.battery -= 1
        self.steps += 1

        # Check if episode is done
        terminated = self.battery <= 0 or self.steps >= self.max_steps

        # Battery depletion penalty
        if self.battery <= 0:
            reward -= 20.0

        # Metrics for info
        coverage = self.total_coverage / (self.grid_size * self.grid_size)
        risk_coverage = self.risk_covered / np.sum(self.grid)

        obs = self._get_obs()
        truncated = False
        info = {
            'coverage': coverage,
            'risk_coverage': risk_coverage,
            'battery': self.battery,
            'steps': self.steps,
            'position': self.uav_pos
        }

        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        """Render the environment"""
        grid_display = np.copy(self.grid)
        visited_display = np.copy(self.visited)
        
        # Mark UAV position
        x, y = self.uav_pos
        
        print("\n" + "="*50)
        print(f"Step: {self.steps}/{self.max_steps} | Battery: {self.battery}/{self.uav_battery}")
        print(f"UAV Position: ({x}, {y}) | Fire Risk: {self.grid[x][y]:.2f}")
        
        coverage = self.total_coverage / (self.grid_size * self.grid_size) * 100
        print(f"Coverage: {coverage:.1f}% | Risk Coverage: {self.risk_covered/np.sum(self.grid)*100:.1f}%")
        
        print("\nFire Risk Map (UAV=X):")
        for i in range(self.grid_size):
            row = ""
            for j in range(self.grid_size):
                if [i, j] == self.uav_pos:
                    row += " X  "
                else:
                    row += f"{self.grid[i][j]:.2f} "
            print(row)
            
        print("\nVisited Map (1=visited):")
        print(visited_display)
        print("="*50)