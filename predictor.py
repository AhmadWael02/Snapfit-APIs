import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Load the model and weights only once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your class names (adjust based on your dataset)
class_names = ['jeans', 'trousers', 'shorts', 'skirts','track pants']

# Define your model architecture
model = models.resnet101(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))  # Replace with correct output size

# Load your trained weights
# model.load_state_dict(torch.load("Bottomwear_Classifier.pth", map_location=device))
# model.eval()
# model.to(device)

# Define image transformations (match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet means
                         [0.229, 0.224, 0.225])  # ImageNet stds
])

def predict_bottom_type(image: Image.Image) -> str:
    """Predicts the class of the given image."""
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
    return predicted_class
