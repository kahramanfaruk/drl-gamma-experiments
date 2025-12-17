import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# 1. dirname -> src/utils
# 2. dirname -> src
# 3. dirname -> drl-gamma-experiments (ROOT)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from src.environments.frozenlake_env import make_frozenlake_env

# =============================================================================
# Configuration
# =============================================================================
MAP_NAME = "8x8" # Change to "4x4" if needed
SAVE_DIR = os.path.join(ROOT_DIR, "results")

def visualize_environment(map_name):
    """
    Generates a professional heatmap of the FrozenLake layout.
    Colors:
    - Frozen (F): Light Blue
    - Start (S): Green
    - Goal (G): Gold
    - Hole (H): Dark Gray
    """
    print(f"Generating map for {map_name}...")
    
    # 1. Initialize Env to get the layout
    env = make_frozenlake_env(map_name=map_name)
    desc = env.unwrapped.desc # This returns a numpy array of bytes (e.g., b'S', b'F')
    
    rows, cols = desc.shape
    
    # 2. Prepare Data for Heatmap
    # We map characters to integers to assign colors
    # 0=Frozen, 1=Start, 2=Goal, 3=Hole
    grid_data = np.zeros((rows, cols))
    annotations = np.empty((rows, cols), dtype=object)
    
    for r in range(rows):
        for c in range(cols):
            char = desc[r, c].decode('utf-8') # Decode byte to string
            annotations[r, c] = char
            
            if char == 'S':
                grid_data[r, c] = 1 # Start
            elif char == 'G':
                grid_data[r, c] = 2 # Goal
            elif char == 'H':
                grid_data[r, c] = 3 # Hole
            else:
                grid_data[r, c] = 0 # Frozen (Ice)

    # 3. Define Custom Color Palette
    # Indices correspond to the values 0, 1, 2, 3 set above
    colors = [
        '#E1F5FE', # 0: Frozen (Ice Blue)
        '#66BB6A', # 1: Start (Green)
        '#FFCA28', # 2: Goal (Gold)
        '#424242'  # 3: Hole (Dark Gray)
    ]
    cmap = ListedColormap(colors)
    
    # 4. Plot
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(
        grid_data, 
        annot=annotations, 
        fmt="", 
        cmap=cmap, 
        cbar=False,
        linewidths=1, 
        linecolor='gray', 
        square=True,
        annot_kws={"size": 14, "weight": "bold", "color": "black"} # Text styling
    )
    
    plt.title(f"FrozenLake {map_name} Environment Layout", fontsize=16, fontweight='bold', pad=20)
    
    # Remove axis ticks for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 5. Save
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"environment_map_{map_name}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f" Map saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    visualize_environment(MAP_NAME)