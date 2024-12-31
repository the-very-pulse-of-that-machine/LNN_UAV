import numpy as np
import matplotlib.pyplot as plt

 
data = np.load('neo.npy')
data = data.reshape(-1, 5)

 
vx = data[:, 0]
vy = data[:, 1]
vz = data[:, 2]
tx = data[:, 3]
ty = data[:, 4]


 
time_steps = np.arange(len(data))

 
plt.figure(figsize=(12, 8))

 
plt.subplot(1, 2, 1)
plt.plot(time_steps, vx, label='vx')
plt.plot(time_steps, vy, label='vy')
plt.plot(time_steps, vz, label='vz')
plt.title("Velocity Components (vx, vy, vz)")
plt.xlabel("Time step")
plt.ylabel("Velocity (m/s)")
plt.legend()

 
plt.subplot(1, 2, 2)
plt.plot(time_steps, tx, label='tx')
plt.plot(time_steps, ty, label='ty')
plt.title("Target Position (cx, cy)")
plt.xlabel("Time step")
plt.ylabel("Position")
plt.legend()

 
plt.tight_layout()
plt.show()

