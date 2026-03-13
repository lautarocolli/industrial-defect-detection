import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViT_B_16_Classifier(nn.Module):
    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()

        self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        in_features = self.model.heads.head.in_features

        self.model.heads.head = nn.Linear(in_features,num_classes)

        self._freeze_layers()
    
    def _freeze_layers(self) -> None:
        
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.heads.parameters():
            param.requires_grad = True
    
    def print_trainable_layers(self) -> None:

        print(f"\n{'Layer':<20} {'Status'}")
        print("-" * 32)
        for name, module in self.model.named_children():
            trainable = any(p.requires_grad for p in module.parameters())
            status    = "trainable" if trainable else "frozen"
            print(f"{name:<20} {status}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def count_parameters(self) -> dict[str, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())

        return {
            "trainable": trainable,
            "frozen":    total - trainable,
            "total":     total,
        }
    
def build_vit(num_classes: int = 6) -> ViT_B_16_Classifier:
    model = ViT_B_16_Classifier(num_classes=num_classes)
    return model