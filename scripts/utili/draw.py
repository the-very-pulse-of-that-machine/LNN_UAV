import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_trajectory(trajectory, targets):
    trajectory = np.array(trajectory)
    targets = np.array(targets)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
     
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label="UAV Trajectory", color="blue", linewidth=2)
    
     
    ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2], color="red", label="Targets", s=50, marker='o')
    
     
    ax.set_title("3D UAV Trajectory with Targets", fontsize=16)
    ax.set_xlabel("X (meters)", fontsize=12)
    ax.set_ylabel("Y (meters)", fontsize=12)
    ax.set_zlabel("Z (meters)", fontsize=12)
    ax.legend()
    
     
    ax.grid(True)
    
     
    plt.show()
