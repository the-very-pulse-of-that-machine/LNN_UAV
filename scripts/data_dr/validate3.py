import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

 
data_path = 'neo.npy'
predictions_path = 'pred.npy'

 
data = np.load(data_path).reshape(-1, 5)   
predictions = np.load(predictions_path)   

 
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1], hspace=0.2, wspace=0.1)

 
ax_main = fig.add_subplot(gs[0, :])   
ax_main.plot(data[:, 0], label="vx (target)")
ax_main.plot(data[:, 1], label="vy (target)")
ax_main.plot(data[:, 2], label="vz (target)")
ax_main.plot(predictions[:, 0], label="vx (predicted)", linestyle='--')
ax_main.plot(predictions[:, 1], label="vy (predicted)", linestyle='--')
ax_main.plot(predictions[:, 2], label="vz (predicted)", linestyle='--')
ax_main.set_ylim((-1, 1))
ax_main.set_title("1. Overall Results")
ax_main.legend(loc="upper right")

 
x_start1, x_end1 = 1300, 2000   
y_start1, y_end1 = -1, 1.25
rect1 = Rectangle((x_start1, y_start1), x_end1 - x_start1, y_end1 - y_start1,
                  linewidth=1, edgecolor='red', facecolor='none', linestyle='--')
ax_main.add_patch(rect1)

x_start2, x_end2 = 30000, 31000   
y_start2, y_end2 = -2, 2
rect2 = Rectangle((x_start2, y_start2), x_end2 - x_start2, y_end2 - y_start2,
                  linewidth=1, edgecolor='blue', facecolor='none', linestyle='--')
ax_main.add_patch(rect2)

 
ax_zoom1 = fig.add_subplot(gs[1, 0])
ax_zoom1.plot(data[:, 0], label="vx (target)")
ax_zoom1.plot(data[:, 1], label="vy (target)")
ax_zoom1.plot(data[:, 2], label="vz (target)")
ax_zoom1.plot(predictions[:, 0], label="vx (predicted)", linestyle='--')
ax_zoom1.plot(predictions[:, 1], label="vy (predicted)", linestyle='--')
ax_zoom1.plot(predictions[:, 2], label="vz (predicted)", linestyle='--')
ax_zoom1.set_xlim(x_start1, x_end1)
ax_zoom1.set_ylim(y_start1, y_end1)
ax_zoom1.set_title("2. Slice_1", fontsize=10)
ax_zoom1.tick_params(axis='both', labelsize=8)

 
ax_zoom2 = fig.add_subplot(gs[1, 1])
ax_zoom2.plot(data[:, 0], label="vx (target)")
ax_zoom2.plot(data[:, 1], label="vy (target)")
ax_zoom2.plot(data[:, 2], label="vz (target)")
ax_zoom2.plot(predictions[:, 0], label="vx (predicted)", linestyle='--')
ax_zoom2.plot(predictions[:, 1], label="vy (predicted)", linestyle='--')
ax_zoom2.plot(predictions[:, 2], label="vz (predicted)", linestyle='--')
ax_zoom2.set_xlim(x_start2, x_end2)
ax_zoom2.set_ylim(y_start2, y_end2)
ax_zoom2.set_title("3. Slice_2", fontsize=10)
ax_zoom2.tick_params(axis='both', labelsize=8)

plt.show()

