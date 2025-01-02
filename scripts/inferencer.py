import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
import pytorch_lightning as pl
from ncps.wirings import AutoNCP
from ncps.torch import LTC
class Inferencer:
    def __init__(self, model_path: str, in_features: int = 2, out_features: int = 3):
        """
        Args:
        - model_path: 模型权重文件的路径
        - in_features: 输入特征数
        - out_features: 输出特征数
        """
        self.model_path = model_path
        self.in_features = in_features
        self.out_features = out_features
        self.model = self.load_model()

    def load_model(self):
        wiring = AutoNCP(32, self.out_features)  
        ltc_model = LTC(self.in_features, wiring, batch_first=True)  
        
        ltc_model.load_state_dict(torch.load(self.model_path))
        ltc_model.eval()   
        return ltc_model

    def inference(self, input_data: torch.Tensor):
        with torch.no_grad():   
            prediction = self.model(input_data)[0]   
        return prediction

    def infer(self, input: pd.DataFrame):
        data_x = input.values   
        data_x_tensor = torch.Tensor(data_x)   

        predictions = []
        
         
        input_frame = data_x_tensor.T
        print(f'Input frame shape: {input_frame.shape}')   

         
        prediction = self.inference(input_frame)
        predictions.append(prediction.numpy().flatten())   
        
        return np.array(predictions)   

