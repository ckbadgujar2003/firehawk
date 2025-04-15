import os
import numpy as np
import pandas as pd
import random
from scipy.ndimage import gaussian_filter

def generate_fire_risk_dataset(grid_size=10, complexity=0.7, save_path=None):
    """Generate a synthetic forest fire risk dataset
    
    Args:
        grid_size (int): Size of the grid (grid_size x grid_size)
        complexity (float): Complexity of the fire risk patterns (0.0-1.0)
        save_path (str, optional): Path to save the CSV dataset. If None, the dataset is not saved.
        
    Returns:
        tuple: (DataFrame with the dataset, 2D numpy array of risk values)
    """
    print(f"Generating fire risk dataset with grid size {grid_size}x{grid_size}...")
    
    # Create a grid of coordinates
    coordinates = []
    for y in range(grid_size):
        for x in range(grid_size):
            coordinates.append((x, y))
    
    # Create dataframe
    df = pd.DataFrame(coordinates, columns=['x', 'y'])
    
    # Generate environmental features with spatial correlation
    
    # Temperature (tends to be higher on one side of the map)
    temp_gradient = np.linspace(15, 35, grid_size)
    base_temp = np.tile(temp_gradient, (grid_size, 1))
    noise = np.random.normal(0, 5, (grid_size, grid_size))
    smoothed_noise = gaussian_filter(noise, sigma=complexity*3)
    temperature = base_temp + smoothed_noise
    df['temperature'] = [temperature[y, x] for x, y in coordinates]
    
    # Humidity (inverse correlation with temperature)
    humidity_base = 100 - (temperature - 15) * 2  # 100% at 15C, decreasing as temp increases
    humidity_noise = np.random.normal(0, 10, (grid_size, grid_size))
    smoothed_humidity_noise = gaussian_filter(humidity_noise, sigma=complexity*2)
    humidity = humidity_base + smoothed_humidity_noise
    humidity = np.clip(humidity, 10, 100)  # Clip to reasonable range
    df['humidity'] = [humidity[y, x] for x, y in coordinates]
    
    # Wind speed
    wind_base = np.random.normal(10, 3, (grid_size, grid_size))
    smoothed_wind = gaussian_filter(wind_base, sigma=complexity*4)
    df['wind_speed'] = [max(0, smoothed_wind[y, x]) for x, y in coordinates]
    
    # Rainfall (patches of rain)
    rainfall_base = np.zeros((grid_size, grid_size))
    num_rain_centers = int(grid_size * 0.3)
    for _ in range(num_rain_centers):
        center_x = np.random.randint(0, grid_size)
        center_y = np.random.randint(0, grid_size)
        radius = np.random.randint(grid_size//4, grid_size//2)
        
        for y in range(grid_size):
            for x in range(grid_size):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < radius:
                    rainfall_base[y, x] += max(0, (radius - dist) / radius * 10)
    
    rainfall = gaussian_filter(rainfall_base, sigma=complexity*2)
    df['rainfall'] = [rainfall[y, x] for x, y in coordinates]
    
    # Vegetation density
    vegetation_centers = []
    for _ in range(int(grid_size * 0.4)):
        center_x = np.random.randint(0, grid_size)
        center_y = np.random.randint(0, grid_size)
        vegetation_centers.append((center_x, center_y))
    
    vegetation_density = np.zeros((grid_size, grid_size))
    for y in range(grid_size):
        for x in range(grid_size):
            for cx, cy in vegetation_centers:
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                vegetation_density[y, x] += max(0, 1 - dist / grid_size)
    
    vegetation_density = gaussian_filter(vegetation_density, sigma=complexity)
    vegetation_density = np.clip(vegetation_density, 0, 1)
    df['vegetation_density'] = [vegetation_density[y, x] for x, y in coordinates]
    
    # Slope (create some mountains and valleys)
    elevation = np.zeros((grid_size, grid_size))
    num_peaks = int(grid_size * 0.2)
    for _ in range(num_peaks):
        peak_x = np.random.randint(0, grid_size)
        peak_y = np.random.randint(0, grid_size)
        peak_height = np.random.uniform(50, 200)
        
        for y in range(grid_size):
            for x in range(grid_size):
                dist = np.sqrt((x - peak_x)**2 + (y - peak_y)**2)
                elevation[y, x] += peak_height * np.exp(-dist / (grid_size * 0.2))
    
    elevation = gaussian_filter(elevation, sigma=complexity*2)
    df['elevation'] = [elevation[y, x] for x, y in coordinates]
    
    # Calculate slopes
    slope = np.zeros((grid_size, grid_size))
    for y in range(1, grid_size-1):
        for x in range(1, grid_size-1):
            dx = (elevation[y, x+1] - elevation[y, x-1]) / 2
            dy = (elevation[y+1, x] - elevation[y-1, x]) / 2
            slope[y, x] = np.sqrt(dx**2 + dy**2) / 100  # Normalize
    
    slope = np.clip(slope, 0, 1)
    df['slope'] = [slope[y, x] for x, y in coordinates]
    
    # Distance to water (create some water bodies)
    water_map = np.ones((grid_size, grid_size)) * grid_size  # Initialize with max distance
    num_water_bodies = int(grid_size * 0.15)
    
    # Create rivers
    for _ in range(num_water_bodies):
        start_x = np.random.randint(0, grid_size)
        start_y = np.random.randint(0, grid_size)
        length = np.random.randint(grid_size//2, grid_size*2)
        x, y = start_x, start_y
        
        for _ in range(length):
            if 0 <= x < grid_size and 0 <= y < grid_size:
                water_map[y, x] = 0  # Mark as water
            
            # Random direction but with tendency to follow elevation gradient
            if np.random.random() < 0.7:
                # Follow steepest descent
                neighbors = []
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        neighbors.append((nx, ny, elevation[ny, nx]))
                
                if neighbors:
                    neighbors.sort(key=lambda n: n[2])  # Sort by elevation
                    x, y = neighbors[0][0], neighbors[0][1]  # Move to lowest neighbor
            else:
                # Random movement
                x += np.random.randint(-1, 2)
                y += np.random.randint(-1, 2)
                x = max(0, min(grid_size-1, x))
                y = max(0, min(grid_size-1, y))
    
    # Create lakes
    for _ in range(num_water_bodies):
        center_x = np.random.randint(0, grid_size)
        center_y = np.random.randint(0, grid_size)
        radius = np.random.randint(1, grid_size//4)
        
        for y in range(grid_size):
            for x in range(grid_size):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < radius:
                    water_map[y, x] = 0  # Mark as water
    
    # Calculate distance to nearest water
    for y in range(grid_size):
        for x in range(grid_size):
            if water_map[y, x] == 0:
                continue  # Skip water cells
            
            min_dist = grid_size * 2
            for wy in range(grid_size):
                for wx in range(grid_size):
                    if water_map[wy, wx] == 0:
                        dist = np.sqrt((x - wx)**2 + (y - wy)**2)
                        min_dist = min(min_dist, dist)
            
            water_map[y, x] = min_dist
    
    # Normalize to [0, 1]
    water_map = water_map / water_map.max()
    df['distance_to_water'] = [water_map[y, x] for x, y in coordinates]
    
    # Days since last fire (random patches with spatial correlation)
    days_since_fire = np.random.exponential(scale=100, size=(grid_size, grid_size))
    days_since_fire = gaussian_filter(days_since_fire, sigma=complexity*3)
    days_since_fire = days_since_fire / days_since_fire.max() * 365  # Scale to days in a year
    df['days_since_last_fire'] = [days_since_fire[y, x] for x, y in coordinates]
    
    # Soil moisture (correlated with rainfall and distance to water)
    soil_moisture = 0.6 * (1 - water_map) + 0.4 * (rainfall / rainfall.max())
    soil_moisture = gaussian_filter(soil_moisture, sigma=complexity*2)
    df['soil_moisture'] = [soil_moisture[y, x] for x, y in coordinates]
    
    # Calculate fire risk based on features
    # Higher risk with: high temperature, low humidity, high wind, low rainfall,
    # high vegetation, high slope, far from water, long time since last fire, low soil moisture
    
    risk = np.zeros((grid_size, grid_size))
    
    # Normalize all factors to [0, 1] range and combine
    temp_factor = (temperature - temperature.min()) / (temperature.max() - temperature.min())
    humidity_factor = 1 - (humidity - humidity.min()) / (humidity.max() - humidity.min())
    wind_factor = (smoothed_wind - smoothed_wind.min()) / (smoothed_wind.max() - smoothed_wind.min() + 1e-10)
    rain_factor = 1 - (rainfall - rainfall.min()) / (rainfall.max() - rainfall.min() + 1e-10)
    veg_factor = vegetation_density
    slope_factor = slope
    water_factor = water_map
    fire_history_factor = (days_since_fire - days_since_fire.min()) / (days_since_fire.max() - days_since_fire.min() + 1e-10)
    soil_factor = 1 - soil_moisture
    
    # Weighted combination of factors
    weights = {
        'temp': 0.15,
        'humidity': 0.15,
        'wind': 0.1,
        'rain': 0.15,
        'vegetation': 0.15,
        'slope': 0.05,
        'water': 0.1,
        'fire_history': 0.05,
        'soil': 0.1
    }
    
    risk = (
        weights['temp'] * temp_factor +
        weights['humidity'] * humidity_factor +
        weights['wind'] * wind_factor +
        weights['rain'] * rain_factor +
        weights['vegetation'] * veg_factor +
        weights['slope'] * slope_factor +
        weights['water'] * water_factor +
        weights['fire_history'] * fire_history_factor +
        weights['soil'] * soil_factor
    )
    
    # Add some randomness
    noise = np.random.normal(0, 0.05, (grid_size, grid_size))
    risk += noise
    
    # Smooth the risk map
    risk = gaussian_filter(risk, sigma=complexity)
    
    # Clip to [0, 1] range
    risk = np.clip(risk, 0, 1)
    
    # Add to dataframe
    df['risk'] = [risk[y, x] for x, y in coordinates]
    
    # Save to CSV if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Dataset saved to {save_path}")
    
    return df, risk

# Example usage
if __name__ == "__main__":
    df, risk_grid = generate_fire_risk_dataset(grid_size=10, save_path="data/fire_risk_dataset.csv")
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Shape: {df.shape}")
    print("\nFeature statistics:")
    print(df.describe())
    
    # Visualize the risk map
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        plt.imshow(risk_grid, cmap='YlOrRd', interpolation='nearest')
        plt.colorbar(label='Fire Risk')
        plt.title('Generated Fire Risk Map')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True, which='both', color='black', linewidth=0.5, alpha=0.2)
        plt.savefig("data/fire_risk_map.png", dpi=300, bbox_inches='tight')
        plt.show()
    except ImportError:
        print("Matplotlib not available for visualization")