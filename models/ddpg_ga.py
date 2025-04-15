import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple

class Actor(nn.Module):
    """Actor network for DDPG"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.tanh(self.fc3(x))

class Critic(nn.Module):
    """Critic network for DDPG"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size + action_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        
    def forward(self, state, action):
        x = self.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """Experience replay buffer for storing experiences"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.memory)

class GeneticOptimizer:
    """Genetic algorithm optimizer for DDPG"""
    def __init__(self, actor, state_dim, action_dim, population_size=20, mutation_rate=0.1):
        self.actor = actor
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []
        
    def initialize_population(self):
        """Initialize population with variations of the current actor"""
        self.population = []
        base_weights = self.get_weights_as_vector(self.actor)
        
        # Add current actor to population
        self.population.append(base_weights)
        
        # Generate variations
        for _ in range(self.population_size - 1):
            mutation = np.random.normal(0, self.mutation_rate, base_weights.shape)
            new_individual = base_weights + mutation
            self.population.append(new_individual)
            
    def get_weights_as_vector(self, network):
        """Convert network weights to a flat vector"""
        weights = []
        for param in network.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)
    
    def set_weights_from_vector(self, network, weights):
        """Set network weights from a flat vector"""
        start = 0
        for param in network.parameters():
            param_shape = param.data.shape
            param_len = np.prod(param_shape)
            param.data = torch.from_numpy(
                weights[start:start+param_len].reshape(param_shape)
            ).float().to(param.device)
            start += param_len
            
    def evaluate_individual(self, weights, env, episodes=3, max_steps=1000):
        """Evaluate individual's fitness by running episodes"""
        # Create a temporary actor with these weights
        temp_actor = Actor(self.state_dim, self.action_dim)
        temp_actor.to(self.actor.device)
        self.set_weights_from_vector(temp_actor, weights)
        
        total_reward = 0
        
        for _ in range(episodes):
            state = env.reset()
            done = False
            episode_steps = 0
            
            while not done and episode_steps < max_steps:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.actor.device)
                with torch.no_grad():
                    action = temp_actor(state_tensor).cpu().numpy().squeeze()
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
                episode_steps += 1
                
        return total_reward / episodes
    
    def select_parents(self, fitness_scores, n_parents):
        """Select parents using tournament selection"""
        parents = []
        for _ in range(n_parents):
            tournament_size = max(2, self.population_size // 5)
            candidates = np.random.choice(len(fitness_scores), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in candidates]
            winner_idx = candidates[np.argmax(tournament_fitness)]
            parents.append(self.population[winner_idx])
        return parents
    
    def crossover(self, parent1, parent2):
        """Perform crossover between two parents"""
        # Single-point crossover
        crossover_point = np.random.randint(0, len(parent1))
        child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        return child
    
    def mutate(self, individual):
        """Apply mutation to an individual"""
        mutation = np.random.normal(0, self.mutation_rate, individual.shape)
        mutated = individual + mutation
        return mutated
    
    def optimize(self, env, n_generations=5):
        """Run the genetic optimization process"""
        self.initialize_population()
        
        for generation in range(n_generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in self.population:
                fitness = self.evaluate_individual(individual, env)
                fitness_scores.append(fitness)
                
            # Select parents
            n_parents = max(2, self.population_size // 2)
            parents = self.select_parents(fitness_scores, n_parents)
            
            # Create new population
            new_population = []
            
            # Elitism: keep the best individual
            best_idx = np.argmax(fitness_scores)
            new_population.append(self.population[best_idx])
            
            # Create children through crossover and mutation
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
                
            self.population = new_population
            
            print(f"Generation {generation+1}/{n_generations}, Best fitness: {max(fitness_scores):.2f}")
            
        # Update actor with best individual
        best_idx = np.argmax(fitness_scores)
        best_weights = self.population[best_idx]
        self.set_weights_from_vector(self.actor, best_weights)
        
        return max(fitness_scores)

class DDPG_GA_Agent:
    """
    Deep Deterministic Policy Gradient with Genetic Algorithm optimization.
    This is the main class that combines DDPG with genetic algorithm optimization.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, 
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.001,
                 buffer_size=100000, batch_size=64, population_size=20,
                 mutation_rate=0.1, crossover_rate=0.5, device="cpu"):
        """
        Initialize the DDPG with GA agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer size
            actor_lr: Learning rate for actor
            critic_lr: Learning rate for critic
            gamma: Discount factor
            tau: Soft update parameter
            buffer_size: Size of replay buffer
            batch_size: Batch size for learning
            population_size: Size of GA population
            mutation_rate: Rate of mutation for GA
            crossover_rate: Rate of crossover for GA
            device: Device to run the model on
        """
        self.device = torch.device(device)
        
        # Initialize actor-critic networks
        self.actor = Actor(state_dim, action_dim, hidden_size=hidden_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_size=hidden_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_size=hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_size=hidden_dim).to(self.device)
        
        # Copy weights from main networks to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Setup optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        
        # RL parameters
        self.gamma = gamma
        self.tau = tau
        
        # Initialize genetic optimizer
        self.genetic_optimizer = GeneticOptimizer(
            self.actor, state_dim, action_dim, 
            population_size=population_size,
            mutation_rate=mutation_rate
        )
        
        # Store additional parameters
        self.crossover_rate = crossover_rate
        
        # Counters and tracking
        self.memory_counter = 0
        self.learn_step_counter = 0
        
        print(f"DDPG+GA Agent Initialized: state_dim={state_dim}, action_dim={action_dim}")
    
    def select_action(self, state, evaluate=False):
        """
        Select an action given the current state.
        
        Args:
            state: Current state
            evaluate: Whether to add noise for exploration
            
        Returns:
            Selected action
        """
        # Print the state to debug
        print(f"State type: {type(state)}, content: {state}")
        
        # Handle different state formats
        if isinstance(state, dict):
            # If state is a dictionary, flatten it to a list
            state_values = []
            for key, value in state.items():
                if isinstance(value, (list, tuple, np.ndarray)):
                    state_values.extend(value)
                else:
                    state_values.append(value)
            state = np.array(state_values, dtype=np.float32)
        elif isinstance(state, list) and any(isinstance(item, (list, tuple)) for item in state):
            # If state is a nested list, flatten it
            flat_state = []
            for item in state:
                if isinstance(item, (list, tuple, np.ndarray)):
                    flat_state.extend(item)
                else:
                    flat_state.append(item)
            state = np.array(flat_state, dtype=np.float32)
        else:
            try:
                # Try to convert directly to numpy array
                state = np.array(state, dtype=np.float32)
            except ValueError:
                # If conversion fails, print detailed info and create a dummy state
                print(f"ERROR: Could not convert state to numpy array.")
                print(f"State structure: {state}")
                # Create a dummy state of zeros with the expected dimension
                state = np.zeros(self.actor.fc1.in_features, dtype=np.float32)
        
        # Ensure state has correct shape for batch processing
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        # Check if dimensions match what the model expects
        expected_dim = self.actor.fc1.in_features
        if state.shape[1] != expected_dim:
            print(f"WARNING: State dimension mismatch. Got {state.shape[1]}, expected {expected_dim}")
            # Pad or truncate to match expected dimensions
            if state.shape[1] < expected_dim:
                # Pad with zeros
                padding = np.zeros((state.shape[0], expected_dim - state.shape[1]), dtype=np.float32)
                state = np.concatenate([state, padding], axis=1)
            else:
                # Truncate
                state = state[:, :expected_dim]
        
        state = torch.FloatTensor(state).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().squeeze()
        self.actor.train()
        
        # Add noise during training for exploration
        if not evaluate:
            noise = np.random.normal(0, 0.1, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
            
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.memory.push(state, action, reward, next_state, done)
        self.memory_counter += 1
    
    def learn(self, state, action, reward, next_state, done):
        """Update the networks based on stored experiences"""
        # Convert state and next_state to proper format
        # Use the same preprocessing as in select_action
        def preprocess_state(s):
            if isinstance(s, dict):
                # If state is a dictionary, flatten it to a list
                s_values = []
                for key, value in s.items():
                    if isinstance(value, (list, tuple, np.ndarray)):
                        s_values.extend(value)
                    else:
                        s_values.append(value)
                return np.array(s_values, dtype=np.float32)
            elif isinstance(s, list) and any(isinstance(item, (list, tuple)) for item in s):
                # If state is a nested list, flatten it
                flat_s = []
                for item in s:
                    if isinstance(item, (list, tuple, np.ndarray)):
                        flat_s.extend(item)
                    else:
                        flat_s.append(item)
                return np.array(flat_s, dtype=np.float32)
            else:
                try:
                    return np.array(s, dtype=np.float32)
                except ValueError:
                    # Return zeros if conversion fails
                    return np.zeros(self.actor.fc1.in_features, dtype=np.float32)
        
        state = preprocess_state(state)
        next_state = preprocess_state(next_state)
        
        # Ensure action is numpy array
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        
        # Store the transition
        self.store_transition(state, action, reward, next_state, done)
        
        # Rest of the method remains the same...
        # Sample batch from memory
        if len(self.memory) < self.batch_size:
            return
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to PyTorch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_value = rewards + (1.0 - dones) * self.gamma * target_q
            
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
        
        self.learn_step_counter += 1
    
    def soft_update(self, local_model, target_model):
        """
        Soft update target network parameters.
        θ_target = τ*θ_local + (1-τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def apply_genetic_optimization(self, env):
        """
        Apply genetic algorithm optimization to the actor network.
        
        Args:
            env: Environment to evaluate individuals
            
        Returns:
            float: Best fitness score
        """
        print("Applying genetic optimization...")
        best_fitness = self.genetic_optimizer.optimize(env)
        
        # Update target network after genetic optimization
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        return best_fitness
    
    def save(self, path):
        """
        Save the models.
        
        Args:
            path: Path to save the models
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """
        Load models from file.
        
        Args:
            path: Path to load the models from
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        print(f"Model loaded from {path}")


# For compatibility with older code
# For compatibility with older code
class DDPGWithGA(DDPG_GA_Agent):
    """
    Alias for DDPG_GA_Agent for backward compatibility.
    """
    def __init__(self, state_dim, action_dim, grid_size, 
                 hidden_dim=128, actor_lr=1e-4, critic_lr=1e-3, 
                 gamma=0.99, tau=0.001, buffer_size=100000, batch_size=64,
                 population_size=20, mutation_rate=0.1, crossover_rate=0.5,
                 device="cpu"):
        """
        Initialize the DDPG with GA agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            grid_size: Size of the grid environment
            hidden_dim: Hidden layer size
            actor_lr: Learning rate for actor
            critic_lr: Learning rate for critic
            gamma: Discount factor
            tau: Soft update parameter
            buffer_size: Size of replay buffer
            batch_size: Batch size for learning
            population_size: Size of GA population
            mutation_rate: Rate of mutation for GA
            crossover_rate: Rate of crossover for GA
            device: Device to run the model on
        """
        print("✅ DDPGWithGA constructor loaded")
        super().__init__(
            state_dim=state_dim, 
            action_dim=action_dim, 
            hidden_dim=hidden_dim,
            actor_lr=actor_lr, 
            critic_lr=critic_lr,
            gamma=gamma, 
            tau=tau,
            buffer_size=buffer_size, 
            batch_size=batch_size,
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            device=device
        )
        self.grid_size = grid_size