import torch
import torch.nn as nn

class PinballLoss(nn.Module):
    """
    Квантильная функция потерь (Pinball Loss) для одного квантиля.
    
    Математическая логика:
    L(y, y_pred) = max(q * (y - y_pred), (q - 1) * (y - y_pred))
    
    Эта функция асимметрична:
    - При q > 0.5 штраф за недопрогноз (y > y_pred) выше, чем за перепрогноз.
    - Это заставляет модель предсказывать значение, выше которого истинный спрос 
      окажется только в (1-q)% случаев.
    """
    def __init__(self, quantile: float = 0.9):
        super(PinballLoss, self).__init__()
        self.quantile = quantile

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        errors = target - pred
        loss = torch.max(self.quantile * errors, (self.quantile - 1) * errors)
        return torch.mean(loss)

class MultiQuantileLoss(nn.Module):
    """
    Составная функция потерь для обучения модели на несколько квантилей сразу.
    Позволяет модели выучить распределение спроса, а не только среднее значение.
    """
    def __init__(self, quantiles=[0.5, 0.75, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        preds: [batch_size, num_quantiles]
        target: [batch_size, 1]
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            # Извлекаем предсказание для i-го квантиля
            errors = target - preds[:, i:i+1]
            q_loss = torch.max(q * errors, (q - 1) * errors)
            losses.append(torch.mean(q_loss))
        
        return torch.mean(torch.stack(losses))
