import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import Resize

class ChestResNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2): 
        super().__init__()
        
        # 1. Load pre-trained ResNet-18
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 2. Modify input layer for grayscale images
        original_conv1 = self.base_model.conv1
        self.base_model.conv1 = nn.Conv2d(
            in_channels,
            original_conv1.out_channels, 
            kernel_size=original_conv1.kernel_size, 
            stride=original_conv1.stride, 
            padding=original_conv1.padding, 
            bias=original_conv1.bias
        )
        
        # 3. Modify final classification layer
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, 1 if num_classes == 2 else num_classes)

        # 4. Add resizing layer for proper input dimensions
        self.resize = Resize((224, 224), antialias=True)

    def forward(self, x):
        x = self.resize(x)
        x = self.base_model(x)
        return x


# --- Testing section ---
if __name__ == '__main__':
    NUM_CLASSES = 2
    IN_CHANNELS = 1
    
    print("--- Testing ChestResNet Model ---")
    
    model = ChestResNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    print("Model Architecture:")
    print(model)
    
    # Test with dummy input
    dummy_input = torch.randn(4, IN_CHANNELS, 28, 28)
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
