import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt
import seaborn as sns
from ncps.wirings import AutoNCP
from ncps.torch import LTC

 
def parse_args():
    parser = argparse.ArgumentParser(description="Train a SequenceLearner model")
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--npy', type=str, required=True, help='Path to the .npy data file')
    parser.add_argument('--epochs', type=int, default=55, help='Number of training epochs')   
    return parser.parse_args()

 
def load_data(npy_file):
    data1 = np.load(npy_file)   
    data1 = data1.reshape(-1, 5)   
    print(f"Data shape: {data1.shape}")

     
    data_x = data1[:, -2:]   
    data_y = data1[:, :3]   

     
    data_x = torch.Tensor(data_x)
    data_y = torch.Tensor(data_y)

    return data_x, data_y

 
class SequenceLearner(pl.LightningModule):
    def __init__(self, model, lr=0.005):
        super().__init__()
        self.model = model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        loss = nn.MSELoss()(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


 
out_features = 3   
in_features = 2   

 
wiring = AutoNCP(32, out_features)   
ltc_model = LTC(in_features, wiring, batch_first=True)   

 
args = parse_args()

 
data_x, data_y = load_data(args.npy)

 
learn = SequenceLearner(ltc_model, lr=0.002)
if args.ckpt:
    learn = SequenceLearner.load_from_checkpoint(args.ckpt,model=ltc_model)  
    print(f"Loaded checkpoint from {args.ckpt}")

 
dataloader = data.DataLoader(
    data.TensorDataset(data_x, data_y), batch_size=1, shuffle=True, num_workers=4
)

 
trainer = pl.Trainer(
    logger=pl.loggers.CSVLogger("log"),
    max_epochs=args.epochs,   
    gradient_clip_val=1,
)

 
trainer.fit(learn, dataloader, ckpt_path=args.ckpt)

 
trainer.save_checkpoint("ddf4.ckpt")
print("Checkpoint saved!")
torch.save(learn.model.state_dict(), 'ddf4.pth')
print("Model weights saved!")

 
sns.set()
with torch.no_grad():
    prediction = ltc_model(data_x)[0].numpy()

 
plt.figure(figsize=(6, 4))

 
plt.plot(data_y[:, 0], label="vx (target)")   
plt.plot(data_y[:, 1], label="vy (target)")   
plt.plot(data_y[:, 2], label="vz (target)")   

 
plt.plot(prediction[:, 0], label="vx (predicted)", linestyle='--')   
plt.plot(prediction[:, 1], label="vy (predicted)", linestyle='--')   
plt.plot(prediction[:, 2], label="vz (predicted)", linestyle='--')  
plt.ylim((-1, 1))
plt.title("After training")
plt.legend(loc="upper right")
plt.show()
