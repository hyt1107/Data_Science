import torch
import torch.nn as nn
import torch.optim as optim
import copy

from wm_base import WatermarkTask

class StudentWatermarkTask(WatermarkTask):
    def __init__(self, clean_model: nn.Module, num_models: int = 10, perturbation_factor: float = 0.01, batch_size: int = 128, device: str = 'cuda'):
        super().__init__(clean_model, num_models, perturbation_factor, batch_size, device)
        
        # 定義水印注入的目標層
        self.target_layer_name = 'layer4.1.conv2.weight'
        
        # 生成一個固定、可重複的水印信號
        # 我們需要知道目標權重的形狀
        temp_model = copy.deepcopy(clean_model)
        target_weight = temp_model.get_parameter(self.target_layer_name)
        
        torch.manual_seed(1337) # 使用一個固定的種子來確保水印信號每次都一樣
        self.watermark_signal = torch.randn_like(target_weight.data)
        
        # 注入強度
        self.scale = 0.001

    def insert_watermark(self, model: nn.Module) -> nn.Module:
        """
        直接修改特定層的權重來注入水印。
        這個過程極快且獨立於數據。
        """
        wm_model = copy.deepcopy(model).to(self.device)
        
        # 找到目標層並修改其權重
        with torch.no_grad():
            target_weight = wm_model.get_parameter(self.target_layer_name)
            # 在原始權重的基礎上，加上我們的水印信號
            target_weight.data += self.scale * self.watermark_signal.to(self.device)
            
        return wm_model

    def extract_features(self, model: nn.Module) -> torch.Tensor:
        """
        直接從目標層提取權重作為特徵。
        """
        model.eval()
        # 找到目標層的權重
        target_weight = model.get_parameter(self.target_layer_name)
        # 將權重展平作為特徵向量
        return target_weight.data.flatten().cpu()

    def train_detector(self, clean_models: list, wm_models: list, epochs: int = 200):
        """
        訓練檢測器來區分正常權重分佈和帶有水印的權重分佈。
        """
        features = []
        labels = []
        for m in clean_models:
            features.append(self.extract_features(m))
            labels.append(0)
        for m in wm_models:
            features.append(self.extract_features(m))
            labels.append(1)
            
        X = torch.stack(features).to(self.device)
        y = torch.tensor(labels, dtype=torch.long).to(self.device)

        input_dim = X.shape[1]
        # 鑒於特徵現在是純粹的權重，一個相對簡單的檢測器可能效果更好
        detector = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        ).to(self.device)
        
        optimizer = torch.optim.Adam(detector.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        detector.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = detector(X)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

        detector.eval()
        return detector