import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

 
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)   
        c_0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)   
        out, _ = self.lstm(x, (h_0, c_0))   
        out = self.fc(out[:, -1, :])   
        return out

 
def lstm_inference(model, input_data):
    model.eval()
    with torch.no_grad():
        input_data = torch.Tensor(input_data).unsqueeze(1)   
        predictions = model(input_data)
    return predictions.numpy()

 
input_size = 2       
hidden_size = 29     
output_size = 3      
model_path = 'lstm_model_tiny.pth'   

 
lstm_model = LSTMModel(input_size, hidden_size, output_size)
lstm_model.load_state_dict(torch.load(model_path))
lstm_model.eval()

 
data_path = '004.npy'
data = np.load(data_path).reshape(-1, 5)   
input_data = data[:, -2:]   
target_data = data[:, :3]   

 
predictions = lstm_inference(lstm_model, input_data)

 
np.save('pred.npy', predictions)

 
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1], hspace=0.2, wspace=0.1)

 
ax_main = fig.add_subplot(gs[0, :])   
ax_main.plot(target_data[:, 0], label="vx (target)")
ax_main.plot(target_data[:, 1], label="vy (target)")
ax_main.plot(target_data[:, 2], label="vz (target)")
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
ax_zoom1.plot(target_data[:, 0], label="vx (target)")
ax_zoom1.plot(target_data[:, 1], label="vy (target)")
ax_zoom1.plot(target_data[:, 2], label="vz (target)")
ax_zoom1.plot(predictions[:, 0], label="vx (predicted)", linestyle='--')
ax_zoom1.plot(predictions[:, 1], label="vy (predicted)", linestyle='--')
ax_zoom1.plot(predictions[:, 2], label="vz (predicted)", linestyle='--')
ax_zoom1.set_xlim(x_start1, x_end1)
ax_zoom1.set_ylim(y_start1, y_end1)
ax_zoom1.set_title("2. Slice_1", fontsize=10)
ax_zoom1.tick_params(axis='both', labelsize=8)

 
ax_zoom2 = fig.add_subplot(gs[1, 1])
ax_zoom2.plot(target_data[:, 0], label="vx (target)")
ax_zoom2.plot(target_data[:, 1], label="vy (target)")
ax_zoom2.plot(target_data[:, 2], label="vz (target)")
ax_zoom2.plot(predictions[:, 0], label="vx (predicted)", linestyle='--')
ax_zoom2.plot(predictions[:, 1], label="vy (predicted)", linestyle='--')
ax_zoom2.plot(predictions[:, 2], label="vz (predicted)", linestyle='--')
ax_zoom2.set_xlim(x_start2, x_end2)
ax_zoom2.set_ylim(y_start2, y_end2)
ax_zoom2.set_title("3. Slice_2", fontsize=10)
ax_zoom2.tick_params(axis='both', labelsize=8)

plt.show()

from sklearn.metrics import mean_squared_error

 
mse_vx = mean_squared_error(target_data[:, 0], predictions[:, 0])
mse_vy = mean_squared_error(target_data[:, 1], predictions[:, 1])
mse_vz = mean_squared_error(target_data[:, 2], predictions[:, 2])

 
print(f"MSE for vx: {mse_vx:.4f}")
print(f"MSE for vy: {mse_vy:.4f}")
print(f"MSE for vz: {mse_vz:.4f}")
