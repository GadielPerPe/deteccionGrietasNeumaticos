import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import densenet121, DenseNet121_Weights
from PIL import Image
import numpy as np

# Definir la clase para el extractor de características
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.densenet = densenet121(weights=DenseNet121_Weights.DEFAULT)

    def forward(self, x):
        # Pasar las imágenes a través de DenseNet hasta la capa de agregación
        x = self.densenet.features(x)
        x = nn.functional.relu(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x

# Inicializar el dispositivo y el extractor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
feature_extractor = FeatureExtractor().to(device)
feature_extractor.eval()  # modo evaluación

# Definir las transformaciones (para una sola imagen)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

def extraer_densenet(img_path: str) -> np.ndarray:
    """Extrae características de una imagen usando DenseNet121."""
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)  # añadir batch dimension
    with torch.no_grad():
        features = feature_extractor(x)
    return features.cpu().numpy().flatten()