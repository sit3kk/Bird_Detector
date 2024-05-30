import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import matplotlib.pyplot as plt


transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def prediction_ResNet(image_path):

    model_path = "saved_models/ResNet/model.pth"
    model = models.resnet50()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    input_image = load_image(image_path)
    with torch.no_grad():
        output = model(input_image)
        _, predicted = torch.max(output, 1)
    return "bird" if predicted == 0 else "nonbird"
