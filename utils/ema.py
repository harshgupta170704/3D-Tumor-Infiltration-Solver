"""
Exponential Moving Average (EMA) for model weights.
Improves model generalization and stability during training.
"""
import copy
import torch
import torch.nn as nn

class EMAModel(nn.Module):
    """
    Maintains an Exponential Moving Average of a model's weights.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999, device=None):
        super().__init__()
        self.decay = decay
        # Create a deep copy of the model for EMA
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        
        if device is not None:
            self.ema_model = self.ema_model.to(device)

        # Disable gradients for EMA model
        for param in self.ema_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update the EMA weights based on the current model weights.
        """
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            if param.requires_grad:
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def forward(self, *args, **kwargs):
        """
        Forward pass using EMA weights.
        """
        return self.ema_model(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        """
        Return the state dict of the EMA model.
        """
        return self.ema_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        """
        Load a state dict into the EMA model.
        """
        return self.ema_model.load_state_dict(state_dict, strict=strict)
