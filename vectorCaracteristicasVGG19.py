import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import numpy as np

class FeatureExtractorFC6(nn.Module):
    def __init__(self):
        super(FeatureExtractorFC6, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        # capa fc6
        self.fc6 = vgg.classifier[0]

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc6(x)
        return x

# Inicializar extractor y device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
feature_extractor_fc6 = FeatureExtractorFC6().to(device)
feature_extractor_fc6.eval()

# Transformaciones para una sola imagen
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

def extraer_fc6(img_path: str) -> np.ndarray:
    """Extrae características fc6 de una imagen usando VGG19."""
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = feature_extractor_fc6(x)
    return feats.cpu().numpy().flatten()