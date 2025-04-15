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

class DQNWithABC:
    """
    DQN agent with Artificial Bee Colony optimization.
    """
    def __init__(self, state_dim, action_dim, grid_size,
                hidden_dim=128, lr=1e-3, gamma=0.99, tau=0.001,
                buffer_size=100000, batch_size=64,
                colony_size=20, limit=10, num_iterations=100,
                device="cpu"):
        
        print("✅ DQNWithABC constructor loaded")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.grid_size = grid_size
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # Initialize DQN model - adjust parameters based on your DQN constructor
        # Try different combinations based on the error message
        try:
            # Option 1: DQN might expect (state_dim, action_dim, hidden_dim)
            self.model = DQN(state_dim, action_dim, hidden_dim).to(device)
            self.target_model = DQN(state_dim, action_dim, hidden_dim).to(device)
        except TypeError:
            try:
                # Option 2: DQN might expect (state_dim, action_dim)
                self.model = DQN(state_dim, action_dim).to(device)
                self.target_model = DQN(state_dim, action_dim).to(device)
            except:
                # If all else fails, print more specific instructions
                print("Error initializing DQN model. Please check your DQN class constructor and adjust accordingly.")
                raise
        
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        
        # Initialize ABC optimizer
        self.abc_optimizer = ArtificialBeeColony(
            model=self.model,
            colony_size=colony_size,
            limit=limit,
            num_iterations=num_iterations
        )
        
        self.t_step = 0
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.abc_optimize_every = 50  # How often to run ABC optimization
        
        print(f"DQN+ABC Agent Initialized: state_dim={state_dim}, action_dim={action_dim}")


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
        if self.t_step % self.abc_optimize_every == 0:
            self.abc_optimize()
    
    def soft_update(self, local_model, target_model):
        """
        Soft update of target network parameters
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def abc_optimize(self, env=None):
        """
        Run ABC optimization if environment is provided
        """
        if env is not None:
            print("Running ABC optimization...")
            best_solution, best_fitness = self.abc_optimizer.optimize(env)
            self.model.load_state_dict(best_solution)
            self.target_model.load_state_dict(best_solution)
            print(f"ABC optimization complete. Best fitness: {best_fitness:.2f}")
    
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

class ArtificialBeeColony:
    """Artificial Bee Colony algorithm for optimizing DQN network weights"""
    
    def __init__(self, model, colony_size=20, limit=10, num_iterations=100):
        self.model = model
        self.colony_size = colony_size  # Number of bees in the colony
        self.limit = limit  # Max trials before abandoning a solution
        self.num_iterations = num_iterations
        
        # Initialize solution space
        self.solutions = []  # Each solution is a set of model weights
        self.fitness = []  # Fitness of each solution
        self.trials = []  # Trial counter for each solution
        
    def initialize_solutions(self):
        """Initialize solutions with variations of current model weights"""
        # Get current model weights
        base_weights = self.get_flattened_weights(self.model)
        
        self.solutions = []
        self.fitness = []
        self.trials = []
        
        # Add current model as first solution
        self.solutions.append(copy.deepcopy(self.model.state_dict()))
        self.fitness.append(0)  # Will be updated during evaluation
        self.trials.append(0)
        
        # Generate variations for other solutions
        for _ in range(self.colony_size - 1):
            new_weights = base_weights + np.random.normal(0, 0.1, base_weights.shape)
            state_dict = self.create_model_from_weights(new_weights)
            self.solutions.append(state_dict)
            self.fitness.append(0)
            self.trials.append(0)
    
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
    
    def evaluate_solutions(self, env, episodes=3, max_steps=100):
        """Evaluate all solutions in the colony"""
        temp_model = copy.deepcopy(self.model)
        
        for i in range(self.colony_size):
            # Load solution weights into temporary model
            temp_model.load_state_dict(self.solutions[i])
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
    
    def employed_bees_phase(self):
        """Employed bees search for better solutions near their current solutions"""
        for i in range(self.colony_size):
            # Create a modified solution
            current_weights = self.get_flattened_weights_from_dict(self.solutions[i])
            
            # Select random solution to compare (different from current)
            k = i
            while k == i:
                k = random.randint(0, self.colony_size - 1)
                
            neighbor_weights = self.get_flattened_weights_from_dict(self.solutions[k])
            
            # Create a new solution by modifying one dimension
            j = random.randint(0, len(current_weights) - 1)
            phi = random.uniform(-1, 1)
            
            new_weights = current_weights.copy()
            new_weights[j] = current_weights[j] + phi * (current_weights[j] - neighbor_weights[j])
            
            # Convert to state dict
            new_solution = self.create_model_from_weights(new_weights)
            
            # We'll update this during evaluation in onlooker phase
            return new_solution
    
    def get_flattened_weights_from_dict(self, state_dict):
        """Flatten weights from a state dictionary"""
        weights = []
        for name in sorted(state_dict.keys()):
            weights.append(state_dict[name].cpu().numpy().flatten())
        return np.concatenate(weights)
    
    def onlooker_bees_phase(self, env):
        """Onlooker bees select solutions based on fitness"""
        # Calculate selection probabilities
        sum_fitness = sum(self.fitness)
        if sum_fitness == 0:  # Handle case where all fitness values are zero
            probabilities = [1.0 / self.colony_size] * self.colony_size
        else:
            probabilities = [f / sum_fitness for f in self.fitness]
            
        # Create new solutions based on probabilities
        for _ in range(self.colony_size):
            # Select a solution based on probability
            i = random.choices(range(self.colony_size), weights=probabilities)[0]
            
            # Create new solution
            new_solution = self.employed_bees_phase()
            
            # Evaluate new solution
            temp_model = copy.deepcopy(self.model)
            temp_model.load_state_dict(new_solution)
            
            # Evaluate performance
            total_reward = 0
            episodes = 2
            
            for _ in range(episodes):
                state, _ = env.reset()
                episode_reward = 0
                
                for _ in range(100):  # Max steps per episode
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
            
            new_fitness = total_reward / episodes
            
            # Update if better
            if new_fitness > self.fitness[i]:
                self.solutions[i] = new_solution
                self.fitness[i] = new_fitness
                self.trials[i] = 0
            else:
                self.trials[i] += 1
    
    def scout_bees_phase(self):
        """Scout bees replace abandoned solutions"""
        for i in range(self.colony_size):
            # If solution has been tried too many times, replace it
            if self.trials[i] >= self.limit:
                # Generate random weights
                param_shapes = [(name, param.shape) for name, param in self.model.named_parameters()]
                new_state_dict = {}
                
                for name, shape in param_shapes:
                    new_state_dict[name] = torch.randn(shape) * 0.1
                
                self.solutions[i] = new_state_dict
                self.trials[i] = 0
    
    def optimize(self, env):
        """Run the ABC optimization process"""
        print("Starting ABC optimization...")
        self.initialize_solutions()
        
        # Evaluate initial solutions
        self.evaluate_solutions(env)
        
        # Track best solution
        best_solution = None
        best_fitness = float('-inf')
        
        # Main optimization loop
        for iteration in range(self.num_iterations):
            # Employed bees phase
            self.employed_bees_phase()
            
            # Onlooker bees phase
            self.onlooker_bees_phase(env)
            
            # Scout bees phase
            self.scout_bees_phase()
            
            # Find current best solution
            current_best_idx = np.argmax(self.fitness)
            if self.fitness[current_best_idx] > best_fitness:
                best_fitness = self.fitness[current_best_idx]
                best_solution = copy.deepcopy(self.solutions[current_best_idx])
            
            if iteration % 10 == 0:
                print(f"ABC Iteration {iteration}/{self.num_iterations}, Best Fitness: {best_fitness:.2f}")
        
        return best_solution, best_fitness

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

class DQNWithMetaheuristics:
    """DQN algorithm enhanced with ABC and GWO metaheuristics"""
    
    def __init__(self, state_size, action_size, buffer_size=100000, batch_size=64, gamma=0.99,
                 tau=1e-3, lr=5e-4, update_every=4, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.device = device
        
        # Q-Networks
        self.qnetwork_local = DQN(state_size, action_size).to(device)
        self.qnetwork_target = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        
        # Replay memory
        self.memory = ReplayBuffer(buffer_size)
        
        # Metaheuristic optimizers
        self.abc = ArtificialBeeColony(
            self.qnetwork_local,
            colony_size=20,
            limit=10,
            num_iterations=50
        )
        
        self.gwo = GreyWolfOptimizer(
            self.qnetwork_local,
            num_wolves=10,
            num_iterations=30
        )
        
        # Initialize step counter
        self.t_step = 0
        self.episode = 0
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Metaheuristic schedule
        self.abc_optimize_every = 50  # Episodes
        self.gwo_optimize_every = 100  # Episodes
    
    def step(self, state, action, reward, next_state, done):
        """Store experience and learn if it's time"""
        # Save experience in replay memory
        self.memory.add(state, np.array([[action]]), reward, next_state, done)
        
        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
            
        # Update episode counter if episode ends
        if done:
            self.episode += 1
            # Decay exploration rate
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Check if it's time for metaheuristic optimization
            do_abc = (self.episode % self.abc_optimize_every == 0)
            do_gwo = (self.episode % self.gwo_optimize_every == 0)
            
            return do_abc, do_gwo
            
        return False, False
    
    def act(self, state, eval_mode=False):
        """Return action for given state as per current policy"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection
        if not eval_mode and random.random() < self.epsilon:
            return random.choice(np.arange(self.action_size))
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # Greedy action
        return np.argmax(action_values.cpu().data.numpy())
    
    def learn(self, experiences):
        """Update value parameters using batch of experience tuples"""
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values from target model
        with torch.no_grad():
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        self.optimizer.step()
        
        # Update target network
        self.soft_update()
    
    def soft_update(self):
        """Soft update target network parameters"""
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def run_abc_optimization(self, env):
        """Run ABC optimization on the DQN model"""
        print("\nRunning Artificial Bee Colony optimization...")
        best_solution, best_fitness = self.abc.optimize(env)
        
        # Apply best solution
        self.qnetwork_local.load_state_dict(best_solution)
        self.qnetwork_target.load_state_dict(best_solution)
        print(f"ABC optimization complete! Best fitness: {best_fitness:.2f}")
    
    def run_gwo_optimization(self, env):
        """Run Grey Wolf optimization on the DQN model"""
        print("\nRunning Grey Wolf optimization...")
        best_solution, best_fitness = self.gwo.optimize(env)
        
        # Apply best solution
        self.qnetwork_local.load_state_dict(best_solution)
        self.qnetwork_target.load_state_dict(best_solution)
        print(f"GWO optimization complete! Best fitness: {best_fitness:.2f}")
    
    def save(self, folder='models'):
        """Save trained model weights"""
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        torch.save(self.qnetwork_local.state_dict(), f"{folder}/dqn_local.pth")
        torch.save(self.qnetwork_target.state_dict(), f"{folder}/dqn_target.pth")
        print(f"Model saved to {folder}")
        
    def load(self, folder='models'):
        """Load trained model weights"""
        self.qnetwork_local.load_state_dict(torch.load(f"{folder}/dqn_local.pth"))
        self.qnetwork_target.load_state_dict(torch.load(f"{folder}/dqn_target.pth"))
        print(f"Model loaded from {folder}")