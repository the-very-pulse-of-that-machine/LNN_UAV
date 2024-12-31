import numpy as np
import matplotlib.pyplot as plt

 
data = np.load('003.npy')
data = data.reshape(-1, 13)

 
vx = data[:, 0]
vy = data[:, 1]
vz = data[:, 2]
wa = data[:, 3]
a = data[:, 4]
b = data[:, 5]
H = data[:, 6]
r = data[:, 7]
p = data[:, 8]
y = data[:, 9]
dya = data[:, 10]
cx = data[:, 11]
cy = data[:, 12]

 
time_steps = np.arange(len(data))

 
plt.figure(figsize=(12, 8))

 
plt.subplot(3, 2, 1)
plt.plot(time_steps, vx, label='vx')
plt.plot(time_steps, vy, label='vy')
plt.plot(time_steps, vz, label='vz')
plt.title("Velocity Components (vx, vy, vz)")
plt.xlabel("Time step")
plt.ylabel("Velocity (m/s)")
plt.legend()

 
plt.subplot(3, 2, 2)
plt.plot(time_steps, r, label='Roll (r)')
plt.plot(time_steps, p, label='Pitch (p)')
plt.plot(time_steps, y, label='Yaw (y)')
plt.title("Orientation (Roll, Pitch, Yaw)")
plt.xlabel("Time step")
plt.ylabel("Angle (rad)")
plt.legend()

 
plt.subplot(3, 2, 3)
plt.plot(time_steps, cx, label='cx')
plt.plot(time_steps, cy, label='cy')
plt.title("Target Position (cx, cy)")
plt.xlabel("Time step")
plt.ylabel("Position")
plt.legend()

 
plt.subplot(3, 2, 4)
plt.plot(time_steps, H, label='Height (H)')
plt.title("Height")
plt.xlabel("Time step")
plt.ylabel("Height (m)")
plt.legend()

 
plt.subplot(3, 2, 5)
plt.plot(time_steps, dya, label='dya')
plt.title("dya")
plt.xlabel("Time step")
plt.ylabel("Value")
plt.legend()

 
plt.subplot(3, 2, 6)
plt.plot(time_steps, a, label='a')
plt.plot(time_steps, b, label='b')
plt.title("Position Components (a, b)")
plt.xlabel("Time step")
plt.ylabel("Position (m)")
plt.legend()

 
plt.tight_layout()
plt.show()

