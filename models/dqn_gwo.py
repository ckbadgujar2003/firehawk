import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random
import time
import copy
from tqdm import tqdm

# Define Experience tuple type
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQN(nn.Module):
    """Q-Network for DQN"""
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        # Initialize weights
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3.weight.data.normal_(0, 0.1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self, buffer_size=100000):
        self.memory = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        """Add an experience to memory"""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=min(batch_size, len(self.memory)))
        
        states = np.vstack([e.state for e in experiences])
        actions = np.vstack([np.array([e.action]) for e in experiences])
        rewards = np.vstack([np.array([e.reward]) for e in experiences])
        next_states = np.vstack([e.next_state for e in experiences])
        dones = np.vstack([np.array([e.done]) for e in experiences]).astype(np.uint8)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)

class GreyWolfOptimizer:
    """Grey Wolf Optimizer for DQN parameter optimization"""
    
    def __init__(self, model, num_wolves=10, num_iterations=50):
        self.model = model
        self.num_wolves = num_wolves
        self.num_iterations = num_iterations
        
        self.wolves = []  # Population (each wolf is a set of model parameters)
        self.fitness = []  # Fitness of each wolf
        
    def initialize_population(self):
        """Initialize wolf population with variations of current model weights"""
        # Get current model weights as the first wolf
        base_weights = self.get_flattened_weights(self.model)
        
        self.wolves = []
        self.fitness = []
        
        # Add current model as first wolf
        self.wolves.append(copy.deepcopy(self.model.state_dict()))
        self.fitness.append(0)  # Will be updated during evaluation
        
        # Generate variations for other wolves
        for _ in range(self.num_wolves - 1):
            new_weights = base_weights + np.random.normal(0, 0.1, base_weights.shape)
            state_dict = self.create_model_from_weights(new_weights)
            self.wolves.append(state_dict)
            self.fitness.append(0)
    
    def get_flattened_weights(self, model):
        """Extract and flatten weights from a PyTorch model"""
        weights = []
        for param in model.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)
    
    def create_model_from_weights(self, flattened_weights):
        """Create a model state dict from flattened weights"""
        state_dict = {}
        current_idx = 0
        
        for name, param in self.model.named_parameters():
            param_shape = param.data.shape
            param_size = param.data.numel()
            
            # Extract weights for this parameter and reshape
            param_weights = flattened_weights[current_idx:current_idx + param_size]
            param_weights = param_weights.reshape(param_shape)
            
            state_dict[name] = torch.tensor(param_weights, dtype=param.data.dtype)
            current_idx += param_size
            
        return state_dict
        
    def get_flattened_weights_from_dict(self, state_dict):
        """Flatten weights from a state dictionary"""
        weights = []
        for name in sorted(state_dict.keys()):
            weights.append(state_dict[name].cpu().numpy().flatten())
        return np.concatenate(weights)
    
    def evaluate_population(self, env, episodes=3, max_steps=100):
        """Evaluate all wolves in the population"""
        temp_model = copy.deepcopy(self.model)
        
        for i in range(self.num_wolves):
            # Load wolf weights into temporary model
            temp_model.load_state_dict(self.wolves[i])
            temp_model.eval()
            
            # Evaluate performance over multiple episodes
            total_reward = 0
            for _ in range(episodes):
                state, _ = env.reset()
                episode_reward = 0
                
                for _ in range(max_steps):
                    # Select action
                    state_tensor = torch.FloatTensor(state)
                    with torch.no_grad():
                        action_values = temp_model(state_tensor)
                    action = torch.argmax(action_values).item()
                    
                    # Take action
                    next_state, reward, done, truncated, _ = env.step(action)
                    episode_reward += reward
                    
                    if done or truncated:
                        break
                        
                    state = next_state
                
                total_reward += episode_reward
            
            # Update fitness
            self.fitness[i] = total_reward / episodes
    
    def optimize(self, env):
        """Run the Grey Wolf optimization process"""
        print("Starting Grey Wolf optimization...")
        self.initialize_population()
        
        # Evaluate initial population
        self.evaluate_population(env)
        
        # Main optimization loop
        for iteration in range(self.num_iterations):
            # Sort wolves based on fitness
            indices = np.argsort(self.fitness)[::-1]  # Descending order
            
            # Update alpha, beta, and delta wolves
            alpha_idx = indices[0]
            beta_idx = indices[1] if len(indices) > 1 else alpha_idx
            delta_idx = indices[2] if len(indices) > 2 else beta_idx
            
            alpha = self.get_flattened_weights_from_dict(self.wolves[alpha_idx])
            beta = self.get_flattened_weights_from_dict(self.wolves[beta_idx])
            delta = self.get_flattened_weights_from_dict(self.wolves[delta_idx])
            
            # Update a parameter that decreases linearly from 2 to 0
            a = 2 - iteration * (2 / self.num_iterations)
            
            # Update each wolf's position
            for i in range(self.num_wolves):
                if i == alpha_idx or i == beta_idx or i == delta_idx:
                    continue  # Don't update the leaders
                
                X = self.get_flattened_weights_from_dict(self.wolves[i])
                
                # Calculate new position based on alpha, beta, and delta
                X1 = self._update_position(X, alpha, a)
                X2 = self._update_position(X, beta, a)
                X3 = self._update_position(X, delta, a)
                
                # New position is average of three positions
                new_X = (X1 + X2 + X3) / 3
                
                # Update wolf position
                self.wolves[i] = self.create_model_from_weights(new_X)
            
            # Evaluate updated population
            self.evaluate_population(env)
            
            # Print progress
            if iteration % 5 == 0:
                best_fitness = self.fitness[alpha_idx]
                print(f"GWO Iteration {iteration}/{self.num_iterations}, Best Fitness: {best_fitness:.2f}")
        
        # Return best wolf
        best_idx = np.argmax(self.fitness)
        return self.wolves[best_idx], self.fitness[best_idx]
    
    def _update_position(self, X, leader, a):
        """Update position based on leader wolf"""
        r1 = np.random.rand(*X.shape)
        r2 = np.random.rand(*X.shape)
        
        A = 2 * a * r1 - a  # Coefficient vector
        C = 2 * r2          # Coefficient vector
        
        D = np.abs(C * leader - X)
        return leader - A * D


class DQNWithGWO:
    """
    DQN agent with Grey Wolf Optimizer.
    """
    def __init__(self, state_dim, action_dim, grid_size,
                hidden_dim=128, lr=1e-3, gamma=0.99, tau=0.001,
                buffer_size=100000, batch_size=64,
                num_wolves=10, num_iterations=50,
                device="cpu"):
        
        print("✅ DQNWithGWO constructor loaded")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.grid_size = grid_size
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # Initialize DQN model
        # Adjust parameters based on your DQN constructor
        self.model = DQN(state_dim, action_dim).to(device)
        self.target_model = DQN(state_dim, action_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        
        # Initialize GWO optimizer
        self.gwo_optimizer = GreyWolfOptimizer(
            model=self.model,
            num_wolves=num_wolves,
            num_iterations=num_iterations
        )
        
        self.t_step = 0
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.gwo_optimize_every = 50  # How often to run GWO optimization
        
        print(f"DQN+GWO Agent Initialized: state_dim={state_dim}, action_dim={action_dim}")

    def select_action(self, state, evaluate=False):
        """
        Select an action using epsilon-greedy policy
        """
        state = torch.FloatTensor(state).to(self.device)
        
        # Epsilon-greedy action selection
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.model(state)
                return torch.argmax(q_values).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in replay memory
        """
        # Convert action to numpy array if it isn't already
        # Make sure action is stored as a single integer value
        if isinstance(action, (list, np.ndarray)):
            action = action[0] if isinstance(action, list) else action.item()
        
        self.memory.add(state, action, reward, next_state, done)
    
    def learn(self):
        """
        Update model parameters using a batch from replay memory
        """
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert numpy arrays to tensors if they aren't already
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states).to(self.device)
        if not isinstance(actions, torch.Tensor):
            # Make sure actions is a 2D tensor with shape (batch_size, 1)
            actions = torch.LongTensor(actions).reshape(-1, 1).to(self.device)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(self.device)
        if not isinstance(next_states, torch.Tensor):
            next_states = torch.FloatTensor(next_states).to(self.device)
        if not isinstance(dones, torch.Tensor):
            dones = torch.FloatTensor(dones).reshape(-1, 1).to(self.device)
        
        # Get Q values
        q_values = self.model(states).gather(1, actions)
        
        # Get next Q values with target network
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1, keepdim=True)[0]
        
        # Compute target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.mse_loss(q_values, target_q_values)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network with soft update
        self.soft_update(self.model, self.target_model)
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Increment step counter
        self.t_step += 1
        
        # Run ABC optimization periodically
        if self.t_step % self.gwo_optimize_every == 0:
            self.gwo_optimize()
    
    def soft_update(self, local_model, target_model):
        """
        Soft update of target network parameters
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def gwo_optimize(self, env=None):
        """
        Run GWO optimization if environment is provided
        """
        if env is not None:
            print("Running GWO optimization...")
            best_solution, best_fitness = self.gwo_optimizer.optimize(env)
            self.model.load_state_dict(best_solution)
            self.target_model.load_state_dict(best_solution)
            print(f"GWO optimization complete. Best fitness: {best_fitness:.2f}")
    
    def save(self, path):
        """
        Save model parameters
        """
        torch.save({
            'model': self.model.state_dict(),
            'target_model': self.target_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """
        Load model parameters
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.target_model.load_state_dict(checkpoint['target_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Model loaded from {path}")