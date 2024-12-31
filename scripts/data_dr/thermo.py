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

 
time_steps = 2000
x1 = np.linspace(8, -8, time_steps)   
x2 = (4 - np.abs(x1))*2   
data_x = np.stack([x1, x2], axis=-1).reshape(time_steps, 2)   
df = pd.DataFrame(data_x, columns=['x1', 'x2'])

 
model_path = 'ddf5.pth'

 
predictions = frame_by_frame_inference(df, model_path)
np.save('pred.npy', predictions)

 
print("Predictions shape:", predictions.shape)
correlation_x1_vy = np.corrcoef(df['x1'], predictions[:, 1])[0, 1]   
print(f"Correlation between x1 and vy: {correlation_x1_vy:.4f}")

 
correlation_x2_vx = np.corrcoef(df['x2'], predictions[:, 0])[0, 1]   
print(f"Correlation between x2 and vx: {correlation_x2_vx:.4f}")
 
fig, axs = plt.subplots(5, 1, figsize=(10, 12))   

 
axs[0].plot(df['x2'], label="TarX (Input)")
axs[0].set_title("TarX (Input)")
axs[0].legend()

 
axs[1].plot(df['x1'], label="TarY (Input)")
axs[1].set_title("TarY (Input)")
axs[1].legend()

 
axs[2].plot(predictions[:, 0], label="vx (Predicted)", linestyle='--')
axs[2].set_title("Predicted v_forward")
axs[2].legend()

 
axs[3].plot(predictions[:, 1], label="vy (Predicted)", linestyle='--')
axs[3].set_title("Predicted v_left")
axs[3].legend()

 
axs[4].plot(predictions[:, 2], label="vz (Predicted)", linestyle='--')
axs[4].set_title("Predicted v_up")
axs[4].legend()

plt.tight_layout()   
plt.show()

