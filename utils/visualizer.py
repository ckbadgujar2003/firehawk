import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

class FireVisualization:
    """Class for visualizing forest fire data and UAV paths"""
    
    def __init__(self, results_dir='results'):
        """Initialize the visualization class
        
        Args:
            results_dir (str): Directory to save visualization results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Set styles for consistent visualizations
        plt.style.use('seaborn-v0_8-whitegrid')
        self.fire_cmap = LinearSegmentedColormap.from_list('fire', ['green', 'yellow', 'orange', 'red', 'darkred'])
        self.coverage_cmap = 'viridis'
        self.path_colors = {
            'DDPG+GA': '#1f77b4',  # blue
            'DQN+ABC': '#ff7f0e',  # orange
            'DQN+GWO': '#2ca02c'   # green
        }
        self.figsize = (10, 8)
        
        # Initialize figure counter for multiple plots in the same run
        self.fig_counter = 0
    
    def plot_fire_risk_heatmap(self, risk_map, title="Forest Fire Risk Map", save=False, filename=None):
        """Plot forest fire risk as a heatmap
        
        Args:
            risk_map (np.array): 2D array of fire risk values
            title (str): Title for the plot
            save (bool): Whether to save the plot
            filename (str): Filename for saved plot (optional)
        """
        plt.figure(figsize=self.figsize)
        plt.imshow(risk_map, cmap=self.fire_cmap, interpolation='nearest')
        plt.colorbar(label='Fire Risk')
        plt.title(title, fontsize=14)
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.xticks(np.arange(0, risk_map.shape[1], step=1))
        plt.yticks(np.arange(0, risk_map.shape[0], step=1))
        plt.grid(False)
        
        # Save the figure if requested
        if save:
            if filename is None:
                filename = f"fire_risk_map_{self.fig_counter}.png"
            filepath = os.path.join(self.results_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved fire risk map to {filepath}")
            self.fig_counter += 1
        
        plt.close()
    
    def plot_uav_path(self, risk_map, path, title="UAV Path", save=False, filename=None):
        """Plot UAV path over fire risk map
        
        Args:
            risk_map (np.array): 2D array of fire risk values
            path (list): List of (x, y) coordinates representing UAV path
            title (str): Title for the plot
            save (bool): Whether to save the plot
            filename (str): Filename for saved plot (optional)
        """
        plt.figure(figsize=self.figsize)
        
        # Plot the fire risk map as background
        plt.imshow(risk_map, cmap=self.fire_cmap, interpolation='nearest')
        
        # Extract path coordinates
        x_coords = [p[0] for p in path]
        y_coords = [p[1] for p in path]
        
        # Plot path with markers and lines
        plt.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7)
        plt.plot(x_coords, y_coords, 'bo', markersize=4)
        
        # Mark start and end points
        plt.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
        plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End')
        
        # Add grid and labels
        plt.colorbar(label='Fire Risk')
        plt.title(title, fontsize=14)
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.xticks(np.arange(0, risk_map.shape[1], step=1))
        plt.yticks(np.arange(0, risk_map.shape[0], step=1))
        plt.grid(False)
        plt.legend()
        
        # Save the figure if requested
        if save:
            if filename is None:
                filename = f"uav_path_{self.fig_counter}.png"
            filepath = os.path.join(self.results_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved UAV path plot to {filepath}")
            self.fig_counter += 1
        
        plt.close()
    
    def plot_path_comparison(self, risk_map, paths_dict, title="UAV Path Comparison", save=False, filename=None):
        """Plot multiple UAV paths on the same risk map for comparison
        
        Args:
            risk_map (np.array): 2D array of fire risk values
            paths_dict (dict): Dictionary mapping model names to their paths
            title (str): Title for the plot
            save (bool): Whether to save the plot
            filename (str): Filename for saved plot (optional)
        """
        plt.figure(figsize=self.figsize)
        
        # Plot the fire risk map as background
        im = plt.imshow(risk_map, cmap=self.fire_cmap, interpolation='nearest')
        
        # Plot each path with different color
        legend_elements = []
        for model_name, path in paths_dict.items():
            color = self.path_colors.get(model_name, 'blue')
            
            # Extract path coordinates
            x_coords = [p[0] for p in path]
            y_coords = [p[1] for p in path]
            
            # Plot path with markers and lines
            plt.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.7)
            plt.plot(x_coords, y_coords, 'o', color=color, markersize=4)
            
            # Mark start points
            plt.plot(x_coords[0], y_coords[0], 'go', markersize=8)
            
            # Add to legend
            legend_elements.append(Line2D([0], [0], color=color, lw=2, label=model_name))
        
        # Add start point to legend
        legend_elements.append(Line2D([0], [0], marker='o', color='g', lw=0, label='Start', 
                             markerfacecolor='g', markersize=8))
        
        # Add grid and labels
        plt.colorbar(im, label='Fire Risk')
        plt.title(title, fontsize=14)
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.xticks(np.arange(0, risk_map.shape[1], step=1))
        plt.yticks(np.arange(0, risk_map.shape[0], step=1))
        plt.grid(False)
        plt.legend(handles=legend_elements)
        
        # Save the figure if requested
        if save:
            if filename is None:
                filename = f"path_comparison_{self.fig_counter}.png"
            filepath = os.path.join(self.results_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved path comparison plot to {filepath}")
            self.fig_counter += 1
        
        plt.close()
    
    def plot_training_progress(self, data, title="Training Progress", save=False, filename=None):
        """Plot training metrics over episodes
        
        Args:
            data (dict): Dictionary with model name and metrics (reward, coverage, high_risk_coverage)
            title (str): Title for the plot
            save (bool): Whether to save the plot
            filename (str): Filename for saved plot (optional)
        """
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # For each model, plot its metrics
        for model_name, metrics in data.items():
            # Create x-axis (episodes)
            episodes = range(1, len(metrics['reward']) + 1)
            
            # Plot reward
            axs[0].plot(episodes, metrics['reward'], label=f"{model_name} Reward")
            
            # Plot coverage
            axs[1].plot(episodes, metrics['coverage'], label=f"{model_name} Coverage")
            
            # Plot high-risk coverage
            axs[2].plot(episodes, metrics['high_risk_coverage'], label=f"{model_name} High-Risk Coverage")
        
        # Set titles and labels
        fig.suptitle(title, fontsize=16)
        
        axs[0].set_title('Episode Reward', fontsize=14)
        axs[0].set_ylabel('Reward')
        axs[0].legend()
        axs[0].grid(True)
        
        axs[1].set_title('Area Coverage', fontsize=14)
        axs[1].set_ylabel('Coverage (%)')
        axs[1].legend()
        axs[1].grid(True)
        
        axs[2].set_title('High-Risk Area Coverage', fontsize=14)
        axs[2].set_xlabel('Episode')
        axs[2].set_ylabel('High-Risk Coverage (%)')
        axs[2].legend()
        axs[2].grid(True)
        
        plt.tight_layout()
        
        # Save the figure if requested
        if save:
            if filename is None:
                filename = f"training_progress_{self.fig_counter}.png"
            filepath = os.path.join(self.results_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved training progress plot to {filepath}")
            self.fig_counter += 1
        
        plt.close()
    
    def plot_training_comparison(self, data_dict, title="Model Comparison", ylabel="Value", save=False, filename=None):
        """Plot comparison of training metrics for multiple models
        
        Args:
            data_dict (dict): Dictionary mapping model names to their metric data
            title (str): Title for the plot
            ylabel (str): Label for y-axis
            save (bool): Whether to save the plot
            filename (str): Filename for saved plot (optional)
        """
        plt.figure(figsize=(12, 8))
        
        # For smoothing curves
        window_size = 10
        
        # For each model, plot its metric with a moving average for smoothing
        for model_name, data in data_dict.items():
            episodes = range(1, len(data) + 1)
            
            # Apply moving average smoothing
            smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
            smoothed_episodes = range(window_size, len(data) + 1)
            
            # Plot smoothed data
            color = self.path_colors.get(model_name, None)
            plt.plot(smoothed_episodes, smoothed_data, label=model_name, linewidth=2, color=color)
        
        # Set titles and labels
        plt.title(title, fontsize=14)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True)
        plt.legend()
        
        # Save the figure if requested
        if save:
            if filename is None:
                filename = f"training_comparison_{self.fig_counter}.png"
            filepath = os.path.join(self.results_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved training comparison plot to {filepath}")
            self.fig_counter += 1
        
        plt.close()
    
    def plot_model_comparison(self, metrics_df, save=False, filename=None):
        """Create comparison plots for model performance metrics
        
        Args:
            metrics_df (pd.DataFrame): DataFrame with model metrics
            save (bool): Whether to save the plots
            filename (str): Base filename for saved plots (optional)
        """
        # Set the style
        sns.set_style("whitegrid")
        
        # Create bar plots for key metrics
        metrics = ['Avg Reward', 'Avg Coverage', 'Avg High-Risk Coverage', 'Training Time (s)']
        
        # Create a 2x2 subplot for the metrics
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        axs = axs.flatten()
        
        for i, metric in enumerate(metrics):
            sns.barplot(x='Model', y=metric, data=metrics_df, ax=axs[i], palette=self.path_colors)
            axs[i].set_title(f'Comparison of {metric}', fontsize=14)
            axs[i].set_xlabel('Model', fontsize=12)
            axs[i].set_ylabel(metric, fontsize=12)
            
            # Add value labels on top of each bar
            for j, p in enumerate(axs[i].patches):
                height = p.get_height()
                axs[i].text(p.get_x() + p.get_width()/2.,
                        height + 0.02,
                        f'{height:.2f}',
                        ha="center")
        
        plt.tight_layout()
        
        # Save the figure if requested
        if save:
            if filename is None:
                filename = f"model_metrics_comparison.png"
            filepath = os.path.join(self.results_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved model comparison plots to {filepath}")
        
        plt.close()
        
        # Create radar chart for comprehensive comparison
        self.plot_radar_chart(metrics_df, save=save)
    
    def plot_radar_chart(self, metrics_df, save=False, filename=None):
        """Create radar chart for comprehensive model comparison
        
        Args:
            metrics_df (pd.DataFrame): DataFrame with model metrics
            save (bool): Whether to save the plot
            filename (str): Filename for saved plot (optional)
        """
        # Select metrics for radar chart
        radar_metrics = ['Avg Reward', 'Avg Coverage', 'Avg High-Risk Coverage', 'Avg Path Length']
        
        # Normalize metrics to 0-1 range for radar chart
        normalized_df = metrics_df.copy()
        for metric in radar_metrics:
            min_val = metrics_df[metric].min()
            max_val = metrics_df[metric].max()
            if max_val != min_val:  # Avoid division by zero
                normalized_df[metric] = (metrics_df[metric] - min_val) / (max_val - min_val)
            else:
                normalized_df[metric] = 1.0
        
        # For path length, invert the values (shorter is better)
        normalized_df['Avg Path Length'] = 1 - normalized_df['Avg Path Length']
        
        # Set up the radar chart
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Number of metrics
        N = len(radar_metrics)
        
        # Compute angle for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], radar_metrics, size=12)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=10)
        plt.ylim(0, 1)
        
        # Plot each model
        for i, (index, row) in enumerate(normalized_df.iterrows()):
            model_name = row['Model']
            values = row[radar_metrics].values.tolist()
            values += values[:1]  # Close the loop
            
            color = self.path_colors.get(model_name, None)
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Model Performance Comparison', size=15)
        
        # Save the figure if requested
        if save:
            if filename is None:
                filename = f"model_radar_comparison.png"
            filepath = os.path.join(self.results_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved radar chart to {filepath}")
        
        plt.close()
    
    def plot_coverage_map(self, grid_size, path, title="UAV Coverage Map", save=False, filename=None):
        """Plot a coverage map showing areas visited by the UAV
        
        Args:
            grid_size (int): Size of the environment grid
            path (list): List of (x, y) coordinates representing UAV path
            title (str): Title for the plot
            save (bool): Whether to save the plot
            filename (str): Filename for saved plot (optional)
        """
        # Create a coverage grid
        coverage_grid = np.zeros((grid_size, grid_size))
        
        # Mark visited cells
        for x, y in path:
            coverage_grid[y, x] = 1
        
        plt.figure(figsize=self.figsize)
        plt.imshow(coverage_grid, cmap='Blues', interpolation='nearest')
        plt.colorbar(label='Visited')
        plt.title(title, fontsize=14)
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.xticks(np.arange(0, grid_size, step=1))
        plt.yticks(np.arange(0, grid_size, step=1))
        plt.grid(False)
        
        # Extract path coordinates
        x_coords = [p[0] for p in path]
        y_coords = [p[1] for p in path]
        
        # Plot path with markers
        plt.plot(x_coords, y_coords, 'r-', linewidth=1.5, alpha=0.7)
        plt.plot(x_coords[0], y_coords[0], 'go', markersize=8, label='Start')
        plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=8, label='End')
        plt.legend()
        
        # Save the figure if requested
        if save:
            if filename is None:
                filename = f"coverage_map_{self.fig_counter}.png"
            filepath = os.path.join(self.results_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved coverage map to {filepath}")
            self.fig_counter += 1
        
        plt.close()
    
    def plot_convergence(self, iterations, fitness_values, title="Optimization Convergence", save=False, filename=None):
        """Plot convergence of optimization algorithms
        
        Args:
            iterations (list): List of iteration numbers
            fitness_values (list): List of fitness values per iteration
            title (str): Title for the plot
            save (bool): Whether to save the plot
            filename (str): Filename for saved plot (optional)
        """
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, fitness_values, 'b-', linewidth=2)
        plt.scatter(iterations, fitness_values, color='blue', s=30)
        
        plt.title(title, fontsize=14)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Best Fitness Value', fontsize=12)
        plt.grid(True)
        
        # Save the figure if requested
        if save:
            if filename is None:
                filename = f"convergence_{self.fig_counter}.png"
            filepath = os.path.join(self.results_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved convergence plot to {filepath}")
            self.fig_counter += 1
        
        plt.close()
    
    def animate_uav_path(self, risk_map, path, save_path=None, interval=200):
        """Create an animation of UAV movement
        
        Note: This requires additional import of FuncAnimation from matplotlib.animation
        
        Args:
            risk_map (np.array): 2D array of fire risk values
            path (list): List of (x, y) coordinates representing UAV path
            save_path (str): Path to save animation
            interval (int): Time interval between frames in milliseconds
        """
        try:
            from matplotlib.animation import FuncAnimation
            
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Plot the fire risk map as background
            im = ax.imshow(risk_map, cmap=self.fire_cmap, interpolation='nearest')
            fig.colorbar(im, label='Fire Risk')
            
            # Extract path coordinates
            x_coords = [p[0] for p in path]
            y_coords = [p[1] for p in path]
            
            # Initialize UAV marker
            uav, = ax.plot([], [], 'bo', markersize=8)
            
            # Initialize path line
            path_line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7)
            
            # Set plot properties
            ax.set_title('UAV Path Animation', fontsize=14)
            ax.set_xlabel('X Coordinate', fontsize=12)
            ax.set_ylabel('Y Coordinate', fontsize=12)
            ax.set_xticks(np.arange(0, risk_map.shape[1], step=1))
            ax.set_yticks(np.arange(0, risk_map.shape[0], step=1))
            ax.grid(False)
            
            # Animation initialization function
            def init():
                uav.set_data([], [])
                path_line.set_data([], [])
                return uav, path_line
            
            # Animation update function
            def update(frame):
                path_line.set_data(x_coords[:frame+1], y_coords[:frame+1])
                uav.set_data(x_coords[frame], y_coords[frame])
                return uav, path_line
            
            # Create animation
            anim = FuncAnimation(fig, update, frames=len(path),
                                  init_func=init, blit=True, interval=interval)
            
            if save_path:
                # Requires ffmpeg or other writer installed
                anim.save(save_path, writer='ffmpeg', fps=5, dpi=100)
                print(f"Animation saved to {save_path}")
            
            plt.close()
            return anim
            
        except ImportError as e:
            print(f"Animation requires additional packages: {e}")
            return None

if __name__ == "__main__":
    # Simple demonstration of the visualizer
    visualizer = FireVisualization(results_dir='test_results')
    
    # Generate sample fire risk map
    grid_size = 10
    risk_map = np.random.rand(grid_size, grid_size)
    risk_map = (risk_map * 10).astype(int) / 10  # Discretize for better visualization
    
    # Generate sample path
    path = [(0, 0)]
    for _ in range(15):
        x, y = path[-1]
        # Random move (up, down, left, right)
        moves = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        # Filter out of bounds moves
        valid_moves = [(a, b) for a, b in moves if 0 <= a < grid_size and 0 <= b < grid_size]
        if valid_moves:
            path.append(valid_moves[np.random.randint(len(valid_moves))])
    
    # Test basic visualizations
    visualizer.plot_fire_risk_heatmap(risk_map, save=True)
    visualizer.plot_uav_path(risk_map, path, save=True)
    visualizer.plot_coverage_map(grid_size, path, save=True)
    
    # Test model comparison visualization
    data = {
        'DDPG+GA': {'reward': np.random.randn(100).cumsum(), 
                    'coverage': np.random.rand(100), 
                    'high_risk_coverage': np.random.rand(100)},
        'DQN+ABC': {'reward': np.random.randn(100).cumsum(), 
                    'coverage': np.random.rand(100), 
                    'high_risk_coverage': np.random.rand(100)}
    }
    visualizer.plot_training_progress(data, save=True)
    
    # Test convergence plot
    iterations = np.arange(50)
    fitness_values = 100 - 90 * np.exp(-0.05 * iterations) + np.random.randn(50) * 5
    visualizer.plot_convergence(iterations, fitness_values, save=True)
    
    print("Visualizer test complete!")