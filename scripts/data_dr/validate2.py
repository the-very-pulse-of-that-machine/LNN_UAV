import torch
import numpy as np
import matplotlib.pyplot as plt
from ncps.wirings import AutoNCP
from ncps.torch import LTC
import pandas as pd
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes 

 
def load_model(model_path: str, in_features: int, out_features: int):
     
    wiring = AutoNCP(32, out_features)   
    ltc_model = LTC(in_features, wiring, batch_first=True)   
    
    plt.figure(figsize=(6, 4))
    legend_handles = wiring.draw_graph(layout='kamada', draw_labels=True)
    plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()
    
     
    model = ltc_model
    model.load_state_dict(torch.load(model_path))
    model.eval()   
    return model

 
def inference(model, input_data: torch.Tensor):
    with torch.no_grad():   
        prediction = model(input_data)[0]   
    return prediction

 
def frame_by_frame_inference(input: pd.DataFrame, model_path: str):
     
    data_x = input.values   
    
     
    data_x_tensor = torch.Tensor(data_x)

     
    model = load_model(model_path, in_features=2, out_features=3)

     
    predictions = []
    for i in range(data_x_tensor.shape[0]):
        input_frame = data_x_tensor[i].unsqueeze(0)   
        prediction = inference(model, input_frame)
        predictions.append(prediction.numpy().flatten())   
    
     
    return np.array(predictions)

 
data1 = np.load('neo.npy')   
data1 = data1.reshape(-1, 5)   
data_x =pd.DataFrame(data1[:,-2:]) 

 
model_path = 'ddf5.pth'

 
predictions = frame_by_frame_inference(data_x, model_path)
np.save('pred.npy',predictions) 

 
print("Predictions shape:", predictions.shape)

 
plt.figure(figsize=(10, 4))

 
plt.plot(data1[:, 0], label="vx (target)")   
plt.plot(data1[:, 1], label="vy (target)")   
plt.plot(data1[:, 2], label="vz (target)")   

 
plt.plot(predictions[:, 0], label="vx (predicted)", linestyle='--')   
plt.plot(predictions[:, 1], label="vy (predicted)", linestyle='--')   
plt.plot(predictions[:, 2], label="vz (predicted)", linestyle='--')   

plt.ylim((-1, 1)) 
plt.title("Prediction Results")
plt.legend(loc="upper right")

 
ax = plt.gca() 
x_start, x_end = 50, 200  
y_start, y_end = -1, 1  
 
rect = Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, linewidth=1, edgecolor='red', facecolor='none', linestyle='--') 
ax.add_patch(rect)  
ax_inset = inset_axes(ax, width="40%", height="40%", loc="upper left") 
ax_inset.plot(data1[:, 0], label="vx (target)") 
ax_inset.plot(data1[:, 1], label="vy (target)") 
ax_inset.plot(data1[:, 2], label="vz (target)") 
ax_inset.plot(predictions[:, 0], label="vx (predicted)", linestyle='--') 
ax_inset.plot(predictions[:, 1], label="vy (predicted)", linestyle='--') 
ax_inset.plot(predictions[:, 2], label="vz (predicted)", linestyle='--') 
ax_inset.set_xlim(x_start, x_end) 
ax_inset.set_ylim(y_start, y_end) 
ax_inset.set_title("Zoomed In View", fontsize=10) 
ax_inset.tick_params(axis='both', labelsize=8)

plt.show()

from sklearn.metrics import mean_squared_error

 
mse_vx = mean_squared_error(data1[:, 0], predictions[:, 0])
mse_vy = mean_squared_error(data1[:, 1], predictions[:, 1])
mse_vz = mean_squared_error(data1[:, 2], predictions[:, 2])

 
print(f"MSE for vx: {mse_vx:.4f}")
print(f"MSE for vy: {mse_vy:.4f}")
print(f"MSE for vz: {mse_vz:.4f}")
