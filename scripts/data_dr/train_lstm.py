import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

 
data_path = 'neo.npy'
data = np.load(data_path).reshape(-1, 5)   
input_features = data[:, -2:]   
target_features = data[:, :3]   

 
inputs = torch.tensor(input_features, dtype=torch.float32)
targets = torch.tensor(target_features, dtype=torch.float32)

 
dataset = TensorDataset(inputs, targets)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
         
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
         
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
         
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
         
        out, _ = self.lstm(x, (h0, c0))   
        
         
        out = self.fc(out[:, -1, :])   
        return out

 
input_size = 2
hidden_size = 29
output_size = 3
num_epochs = 50
learning_rate = 0.001

 
model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for inputs_batch, targets_batch in train_loader:
        inputs_batch = inputs_batch.unsqueeze(1).to(device)   
        targets_batch = targets_batch.to(device)
        
         
        outputs = model(inputs_batch)
        loss = criterion(outputs, targets_batch)
        
         
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
     
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs_batch, targets_batch in val_loader:
            inputs_batch = inputs_batch.unsqueeze(1).to(device)
            targets_batch = targets_batch.to(device)
            
            outputs = model(inputs_batch)
            loss = criterion(outputs, targets_batch)
            val_loss += loss.item()
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")


 
model.eval()
with torch.no_grad():
    inputs_test = inputs.unsqueeze(1).to(device)   
    predictions = model(inputs_test).cpu().numpy()

 
np.save('predictions_lstm_tiny.npy', predictions)

 
torch.save(model.state_dict(), 'lstm_model_tiny.pth')
print("Model saved successfully!")


 
plt.figure(figsize=(10, 6))
plt.plot(data[:, 0], label="vx (target)")
plt.plot(data[:, 1], label="vy (target)")
plt.plot(data[:, 2], label="vz (target)")
plt.plot(predictions[:, 0], label="vx (predicted)", linestyle='--')
plt.plot(predictions[:, 1], label="vy (predicted)", linestyle='--')
plt.plot(predictions[:, 2], label="vz (predicted)", linestyle='--')
plt.ylim((-1, 1))
plt.title("LSTM Prediction Results")
plt.legend(loc="upper right")
plt.show()

