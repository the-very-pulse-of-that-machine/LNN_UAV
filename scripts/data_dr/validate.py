import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
import pytorch_lightning as pl
from ncps.wirings import AutoNCP
from ncps.torch import LTC

 
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

 
class FrameByFrameInferencer:
    def __init__(self, checkpoint_path, input_features=9, output_features=4, model_class=SequenceLearner):
        """
        初始化推理类，加载训练好的模型。
        
        :param checkpoint_path: 模型的权重文件路径
        :param input_features: 输入特征的数量
        :param output_features: 输出特征的数量
        :param model_class: 用于加载和训练模型的类，默认为 SequenceLearner
        """
         
        wiring = AutoNCP(32, output_features)   
        ltc_model = LTC(input_features, wiring, batch_first=True)   
        self.model = model_class(ltc_model, lr=0.001)

         
        self.model = model_class.load_from_checkpoint(checkpoint_path, model=ltc_model)
        self.model.eval()   

    def infer(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        对每一帧数据进行推理，返回包含预测结果的数据帧。
        
        :param input_data: 输入数据帧，必须是一个 shape 为 (N, 9) 的 DataFrame
        :return: 包含预测结果的数据帧，形状为 (N, 4)
        """
         
        input_tensor = torch.Tensor(input_data.values)

         
        predictions = []

         
        with torch.no_grad():
            for i in range(input_tensor.shape[0]):   
                x_frame = input_tensor[i].unsqueeze(0)   
                 
                prediction = self.model.model(x_frame)[0].numpy()   
                predictions.append(prediction.flatten())   

         
        predictions = np.array(predictions)

         
        predictions_df = pd.DataFrame(predictions, columns=["vx", "vy", "vz", "wa"])
        return predictions_df

 

 
inferencer = FrameByFrameInferencer(checkpoint_path="1.ckpt")

 
data_x = np.load('002.npy')   
data_x = data_x.reshape(-1, 13)   
input_data = pd.DataFrame(data_x[:, -9:])   

 
predictions = inferencer.infer(input_data)

 
print(predictions)

 
plt.figure(figsize=(6, 4))

 
data_y = np.load('002.npy')   
data_y = data_y.reshape(-1, 13)
data_y = torch.Tensor(data_y[:, :4])   

 
plt.plot(data_y[:, 0], label="vx (target)", color='blue')   
plt.plot(data_y[:, 1], label="vy (target)", color='green')   
plt.plot(data_y[:, 2], label="vz (target)", color='red')   
plt.plot(data_y[:, 3], label="wa (target)", color='purple')   

 
plt.plot(predictions["vx"], label="vx (predicted)", linestyle='--', color='blue')   
plt.plot(predictions["vy"], label="vy (predicted)", linestyle='--', color='green')   
plt.plot(predictions["vz"], label="vz (predicted)", linestyle='--', color='red')   
plt.plot(predictions["wa"], label="wa (predicted)", linestyle='--', color='purple')   

plt.ylim((-1, 1)) 
plt.title("After training - Model Prediction vs Target") 
plt.legend(loc="upper right") 
plt.show()

