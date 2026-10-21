import torch

class ClassificationAccuracy(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_bool, target):
        if input_bool and target[0] > 0.5:
            return 1.0
        
        if not input_bool and target[0] < 0.5:
            return 1.0
        
        return 0.0