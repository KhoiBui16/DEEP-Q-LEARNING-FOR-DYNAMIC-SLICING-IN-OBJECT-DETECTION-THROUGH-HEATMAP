import os
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

# Load an example image
script_dir = os.path.dirname(__file__)
image_path = os.path.join(script_dir, '../data/dog_background_1.jpg')

# class_idx = 243  # Golden Retriever dog class index
class_idx = None # Predicted class index with highest probability

type_cams = {
    0: "CAM",
    1: "Grad-CAM",
    2: "Grad-CAM++",
    3: "Score-CAM",
    4: "Layer-CAM",
    5: "XGrad-CAM",
    6: "Augmented-GradCAM",
    7: "Gridging-gap-WSOL",
    9: "F-CAM",
    9: "GAN-CAM"
}

# Preprocess the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

WEIGHTS = ResNet18_Weights.IMAGENET1K_V1  # Specify the weights
MODEL = resnet18(weights=WEIGHTS)  # Use the weights parameter
TARGET_LAYER = dict(MODEL.named_modules())['layer4']