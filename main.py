import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import pickle
import random
from tqdm import tqdm

# Import project modules
# Modified import to match the class we created
from fire_prediction.xgboost_predictor import FireRiskPredictor
from env.forest_fire_env import ForestFireEnv
from models.ddpg_ga import DDPGWithGA
from models.dqn_abc import DQN, DQNWithABC, ArtificialBeeColony  # Assuming both classes are in this file
from models.dqn_gwo import DQNWithGWO, GreyWolfOptimizer
from utils.visualizer import FireVisualization

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class ForestFireUAVProject:
    """Main class to run the UAV path planning for forest fire monitoring"""
    
    def __init__(self, config):
        """Initialize the project with configuration parameters
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        self.results_dir = config['results_dir']
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize visualizer
        self.visualizer = FireVisualization(results_dir=self.results_dir)
        
        # Load or create fire risk dataset
        self.load_fire_risk_data()
        
        # Create environment
        self.env = ForestFireEnv(
            grid_size=config['grid_size'],
            fire_map=self.fire_risk_grid,
            max_steps=config['max_steps'],
            reward_weight_coverage=config['reward_weight_coverage'],
            reward_weight_risk=config['reward_weight_risk']
        )
        
        # Initialize the models
        self.initialize_models()
        
        # Metrics storage
        self.training_progress = {}
        self.evaluation_metrics = {}
        self.execution_times = {}
        self.convergence_data = {}
        self.paths = {}
        self.coverage_data = {}
        
    def load_fire_risk_data(self):
        """Load or generate fire risk dataset"""
        data_path = os.path.join(self.config['data_dir'], 'fire_risk_dataset.csv')
        
        if os.path.exists(data_path) and not self.config['regenerate_data']:
            print(f"Loading existing fire risk dataset from {data_path}")
            self.fire_risk_df = pd.read_csv(data_path)
            
            # Convert DataFrame to grid for environment
            grid_size = self.config['grid_size']
            self.fire_risk_grid = np.zeros((grid_size, grid_size))
            
            for _, row in self.fire_risk_df.iterrows():
                x, y = int(row['x']), int(row['y'])
                self.fire_risk_grid[y, x] = row['risk']
                
        else:
            print("Generating new fire risk dataset...")
            # Import here to avoid circular imports
            from fire_prediction.generate_fire_dataset import generate_fire_risk_dataset
            
            self.fire_risk_df, self.fire_risk_grid = generate_fire_risk_dataset(
                grid_size=self.config['grid_size'],
                save_path=data_path
            )
        
        # Visualize fire risk map
        self.visualizer.plot_fire_risk_heatmap(
            self.fire_risk_grid, 
            title="Forest Fire Risk Map",
            save=True
        )
        
    def initialize_models(self):
        """Initialize all models for training and comparison"""
        print("Initializing models...")
    
        grid_size = self.config['grid_size']
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n if hasattr(self.env.action_space, 'n') else self.env.action_space.shape[0]
        
        # Initialize all models with the same state/action dimensions for fair comparison
        self.models = {}
        try:
            print("✅ Initializing DDPG+GA")
            self.models["DDPG+GA"] = DDPGWithGA(
            # Use the correct parameter names from DDPGWithGA constructor
                state_dim=state_dim,
                action_dim=action_dim,
                grid_size=grid_size,
                hidden_dim=self.config.get('hidden_dim', 128),
                actor_lr=self.config.get('actor_lr', 1e-4),
                critic_lr=self.config.get('critic_lr', 1e-3),
                gamma=self.config.get('gamma', 0.99),
                tau=self.config.get('tau', 0.001),
                buffer_size=self.config.get('buffer_size', 100000),
                batch_size=self.config.get('batch_size', 64),
                population_size=self.config.get('population_size', 20),
                mutation_rate=self.config.get('mutation_rate', 0.1),
                crossover_rate=self.config.get('crossover_rate', 0.5),
                device=self.config.get('device', 'cpu')
            )
        except Exception as e:
            print(f"Error initializing DDPG+GA: {e}")

        try:
            self.models["DQN+ABC"] = DQNWithABC(
                state_dim=state_dim,
                action_dim=action_dim,
                grid_size=grid_size,
                hidden_dim=self.config.get('hidden_dim', 128),
                lr=self.config.get('dqn_lr', 1e-3),
                gamma=self.config.get('gamma', 0.99),
                tau=self.config.get('tau', 0.001),
                buffer_size=self.config.get('buffer_size', 100000),
                batch_size=self.config.get('batch_size', 64),
                colony_size=self.config.get('colony_size', 20),
                limit=self.config.get('abc_limit', 10),
                num_iterations=self.config.get('abc_iterations', 100),
                device=self.config.get('device', 'cpu')
            )
        except Exception as e:
            print(f"Error initializing DQN+ABC: {e}") 

        try:
            self.models["DQN+GWO"] = DQNWithGWO(
                state_dim=state_dim,
                action_dim=action_dim,
                grid_size=grid_size,
                hidden_dim=self.config.get('hidden_dim', 128),
                lr=self.config.get('dqn_lr', 1e-3),
                gamma=self.config.get('gamma', 0.99),
                tau=self.config.get('tau', 0.001),
                buffer_size=self.config.get('buffer_size', 100000),
                batch_size=self.config.get('batch_size', 64),
                num_wolves=self.config.get('num_wolves', 10),
                num_iterations=self.config.get('gwo_iterations', 50),
                device=self.config.get('device', 'cpu')
            )
        except Exception as e:
            print(f"Error initializing DQN+GWO: {e}")
    
        # You need to first create a DQN model, then pass it to ArtificialBeeColony
        # First, create a DQN model
        # dqn_model = DQN(
        #     state_size=state_dim,
        #     action_size=action_dim,
        #     hidden_size=self.config.get('hidden_dim', 128)
        # ).to(self.config.get('device', 'cpu'))

        # dqn_optimizer = optim.Adam(dqn_model.parameters(), lr=self.config.get('dqn_lr', 1e-3))
        
        # Then, create the ArtificialBeeColony optimizer with the DQN model
        # self.models["DQN+ABC"] = ArtificialBeeColony(
        #     model=dqn_model,
        #     colony_size=self.config.get('colony_size', 20),
        #     limit=self.config.get('abc_limit', 10),
        #     num_iterations=self.config.get('abc_iterations', 100)
        # )
        
        print(f"Models initialized: {list(self.models.keys())}")

        
    def train_models(self):
        """Train all models and record training progress"""
        print("\n=== Training all models ===")
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            start_time = time.time()
            
            # Training loop
            episodes = self.config['num_episodes']
            rewards_history = []
            coverage_history = []
            high_risk_coverage_history = []
            
            for episode in tqdm(range(episodes)):
                state_tuple = self.env.reset()
                # Extract just the state array, ignore the info dict
                state = state_tuple[0] if isinstance(state_tuple, tuple) else state_tuple
                episode_reward = 0
                done = False
                
                # In main.py, modify your train_models method's learning part:
                while not done:
                    action = model.select_action(state)
                    next_state_tuple, reward, done, truncated, info = self.env.step(action)
                    # Extract just the state array
                    next_state = next_state_tuple[0] if isinstance(next_state_tuple, tuple) else next_state_tuple
                    
                    # Different models might have different learning methods
                    if model_name == "DQN+ABC" or model_name == "DQN+GWO":
                        # These models expect store_transition followed by learn()
                        model.store_transition(state, action, reward, next_state, done)
                        model.learn()  # No parameters needed here
                    elif hasattr(model, 'learn'):
                        # For models that have a learn method taking all parameters
                        model.learn(state, action, reward, next_state, done)
                    elif hasattr(model, 'store_transition') and hasattr(model, 'update'):
                        # For models with separate store and update
                        model.store_transition(state, action, reward, next_state, done)
                        model.update()
                    
                    state = next_state
                    episode_reward += reward
                    
                    # Check if truncated (new in gymnasium)
                    if truncated:
                        done = True
                
                # Collect metrics
                coverage = info['coverage']
                high_risk_coverage = info.get('high_risk_coverage', info.get('risk_coverage', 0))
                
                rewards_history.append(episode_reward)
                coverage_history.append(coverage)
                high_risk_coverage_history.append(high_risk_coverage)
                
                # Logging
                if (episode + 1) % self.config['log_interval'] == 0:
                    avg_reward = np.mean(rewards_history[-self.config['log_interval']:])
                    avg_coverage = np.mean(coverage_history[-self.config['log_interval']:])
                    avg_high_risk = np.mean(high_risk_coverage_history[-self.config['log_interval']:])
                    
                    print(f"Episode {episode+1}/{episodes} | "
                        f"Avg Reward: {avg_reward:.2f} | "
                        f"Avg Coverage: {avg_coverage:.2%} | "
                        f"Avg High-Risk Coverage: {avg_high_risk:.2%}")
            
            end_time = time.time()
            training_time = end_time - start_time
            
            # Store training metrics
            self.training_progress[model_name] = {
                'rewards': rewards_history,
                'coverage': coverage_history,
                'high_risk_coverage': high_risk_coverage_history
            }
            
            self.execution_times[model_name] = {
                'training_time': training_time
            }
            
            print(f"Finished training {model_name} in {training_time:.2f} seconds")
            
            # Save model
            model_path = os.path.join(self.results_dir, f"{model_name.lower().replace('+', '_')}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Plot training curves
            self.visualizer.plot_training_progress(
                data={
                    "DDPG+GA": {
                        "reward": rewards_history,
                        "coverage": coverage_history,
                        "high_risk_coverage": high_risk_coverage_history
                    }
                },
                title=f"{model_name} Training Progress",
                save=True,
                filename=f"{model_name.lower().replace('+', '_')}_training_curves.png"
            )

            
    def evaluate_models(self, num_eval_episodes=10):
        """Evaluate all models and record performance metrics"""
        print("\n=== Evaluating all models ===")
        
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            start_time = time.time()
            
            # Evaluation metrics
            total_rewards = []
            total_coverage = []
            total_high_risk_coverage = []
            total_path_length = []
            paths = []
            
            for episode in range(num_eval_episodes):
                state_tuple = self.env.reset()
                # Extract just the state array, ignore the info dict
                state = state_tuple[0] if isinstance(state_tuple, tuple) else state_tuple
                episode_reward = 0
                done = False
                path = []
                
                while not done:
                    # Use deterministic policy for evaluation
                    action = model.select_action(state, evaluate=True)
                    next_state_tuple, reward, done, truncated, info = self.env.step(action)
                    # Extract just the state array
                    next_state = next_state_tuple[0] if isinstance(next_state_tuple, tuple) else next_state_tuple
                    
                    path.append((info['position'][0], info['position'][1]))
                    state = next_state
                    episode_reward += reward
                    
                    # Check if truncated (new in gymnasium)
                    if truncated:
                        done = True
                
                # Collect metrics
                total_rewards.append(episode_reward)
                total_coverage.append(info['coverage'])
                total_high_risk_coverage.append(info.get('high_risk_coverage', info.get('risk_coverage', 0)))
                total_path_length.append(len(path))
                paths.append(path)
            
            end_time = time.time()
            eval_time = end_time - start_time
            
            # Store evaluation metrics
            self.evaluation_metrics[model_name] = {
                'avg_reward': np.mean(total_rewards),
                'std_reward': np.std(total_rewards),
                'avg_coverage': np.mean(total_coverage),
                'avg_high_risk_coverage': np.mean(total_high_risk_coverage),
                'avg_path_length': np.mean(total_path_length)
            }
            
            self.paths[model_name] = paths[0]  # Store the first path for visualization
            self.execution_times[model_name]['evaluation_time'] = eval_time
            
            print(f"Model: {model_name} | "
                  f"Avg Reward: {self.evaluation_metrics[model_name]['avg_reward']:.2f} | "
                  f"Avg Coverage: {self.evaluation_metrics[model_name]['avg_coverage']:.2%} | "
                  f"Avg High-Risk Coverage: {self.evaluation_metrics[model_name]['avg_high_risk_coverage']:.2%}")
            
            # Visualize path
            self.visualizer.plot_uav_path(
                self.fire_risk_grid,
                paths[0],
                title=f"{model_name} UAV Path",
                save=True,
                filename=f"{model_name.lower().replace('+', '_')}_path.png"
            )
            
    def compare_models(self):
        """Compare all models and generate comparison visualizations"""
        print("\n=== Comparing all models ===")
        
        # Create comparison tables
        metrics_df = pd.DataFrame({
            'Model': list(self.evaluation_metrics.keys()),
            'Avg Reward': [m['avg_reward'] for m in self.evaluation_metrics.values()],
            'Avg Coverage': [m['avg_coverage'] for m in self.evaluation_metrics.values()],
            'Avg High-Risk Coverage': [m['avg_high_risk_coverage'] for m in self.evaluation_metrics.values()],
            'Avg Path Length': [m['avg_path_length'] for m in self.evaluation_metrics.values()],
            'Training Time (s)': [self.execution_times[m]['training_time'] for m in self.evaluation_metrics.keys()],
            'Evaluation Time (s)': [self.execution_times[m]['evaluation_time'] for m in self.evaluation_metrics.keys()]
        })
        
        # Save comparison to CSV
        metrics_df.to_csv(os.path.join(self.results_dir, 'model_comparison.csv'), index=False)
        
        # Generate comparison plots
        self.visualizer.plot_model_comparison(
            metrics_df,
            save=True
        )
        
        # Compare training curves
        self.visualizer.plot_training_comparison(
            {model_name: data['rewards'] for model_name, data in self.training_progress.items()},
            title="Reward Comparison",
            ylabel="Episode Reward",
            save=True,
            filename="reward_comparison.png"
        )
        
        self.visualizer.plot_training_comparison(
            {model_name: data['coverage'] for model_name, data in self.training_progress.items()},
            title="Coverage Comparison",
            ylabel="Area Coverage",
            save=True,
            filename="coverage_comparison.png"
        )
        
        self.visualizer.plot_training_comparison(
            {model_name: data['high_risk_coverage'] for model_name, data in self.training_progress.items()},
            title="High-Risk Coverage Comparison",
            ylabel="High-Risk Area Coverage",
            save=True,
            filename="high_risk_coverage_comparison.png"
        )
        
        # Compare paths on the same grid
        self.visualizer.plot_path_comparison(
            self.fire_risk_grid,
            self.paths,
            title="UAV Path Comparison",
            save=True,
            filename="path_comparison.png"
        )
        
        print("\nComparison complete. Results saved to:", self.results_dir)
        
    def run(self):
        """Run the full experiment pipeline"""
        print("\n=== Starting UAV Path Planning for Forest Fire Monitoring ===")
        print(f"Grid Size: {self.config['grid_size']}x{self.config['grid_size']}")
        print(f"Max Steps: {self.config['max_steps']}")
        print(f"Number of Episodes: {self.config['num_episodes']}")
        
        # Train all models
        self.train_models()
        
        # Evaluate all models
        self.evaluate_models(num_eval_episodes=self.config['num_eval_episodes'])
        
        # Compare models
        self.compare_models()
        
        # Save config for reproducibility
        with open(os.path.join(self.results_dir, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
        
        print("\n=== Experiment completed successfully ===")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='UAV Path Planning for Forest Fire Monitoring')
    
    # Environment settings
    parser.add_argument('--grid_size', type=int, default=10, help='Grid size')
    parser.add_argument('--max_steps', type=int, default=50, help='Maximum steps per episode')
    
    # Training settings
    parser.add_argument('--num_episodes', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--num_eval_episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval during training')
    
    # Model hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for neural networks')
    parser.add_argument('--actor_lr', type=float, default=1e-4, help='Actor learning rate')
    parser.add_argument('--critic_lr', type=float, default=1e-3, help='Critic learning rate')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for DQN')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update parameter')
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='Starting epsilon for exploration')
    parser.add_argument('--epsilon_end', type=float, default=0.01, help='Final epsilon for exploration')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--buffer_size', type=int, default=10000, help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    
    # Metaheuristic parameters
    parser.add_argument('--population_size', type=int, default=30, help='Population size for metaheuristics')
    parser.add_argument('--mutation_rate', type=float, default=0.1, help='Mutation rate for GA')
    parser.add_argument('--crossover_rate', type=float, default=0.8, help='Crossover rate for GA')
    parser.add_argument('--limit', type=int, default=5, help='Limit parameter for ABC')
    
    # Reward weights
    parser.add_argument('--reward_weight_coverage', type=float, default=0.5, help='Weight for coverage reward')
    parser.add_argument('--reward_weight_risk', type=float, default=0.5, help='Weight for risk-based reward')
    
    # Data and results
    parser.add_argument('--data_dir', type=str, default='data', help='Directory for datasets')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory for results')
    parser.add_argument('--regenerate_data', action='store_true', help='Force regeneration of datasets')
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    
    # Convert args to dictionary
    config = vars(args)
    
    # Add timestamp to results directory for unique runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config['results_dir'] = os.path.join(config['results_dir'], f"run_{timestamp}")
    
    # Create directories
    os.makedirs(config['data_dir'], exist_ok=True)
    
    # Run the project
    project = ForestFireUAVProject(config)
    project.run()

if __name__ == "__main__":
    main()