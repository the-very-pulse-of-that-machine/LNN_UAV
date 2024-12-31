import torch
import numpy as np
import pandas as pd
from torch import nn

class LSTMInferencer:
    def __init__(self, model_path: str, in_features: int = 2, out_features: int = 3, hidden_size: int = 64):
        """
        初始化 LSTMInferencer 类，并加载模型。

        Args:
        - model_path: 模型权重文件的路径
        - in_features: 输入特征数
        - out_features: 输出特征数
        - hidden_size: LSTM 隐藏层大小
        """
        self.model_path = model_path
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.model = self.load_model()

    def load_model(self):
        """
        加载预训练的 LSTM 模型。
        """
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
        
         
        model = LSTMModel(self.in_features, self.hidden_size, self.out_features)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()   
        return model

    def inference(self, input_data: torch.Tensor):
        """
        对输入数据进行推理。
        """
        with torch.no_grad():
            prediction = self.model(input_data.unsqueeze(1))   
        return prediction

    def infer(self, input: pd.DataFrame):
        """
        推理接口。
        Args:
        - input: pd.DataFrame，形状为 (2, N)，每列为一个样本
        Returns:
        - np.ndarray，预测结果
        """
         
        data_x = input.values.T   
        data_x_tensor = torch.tensor(data_x, dtype=torch.float32)

         
        predictions = self.inference(data_x_tensor)
        return predictions.numpy()

