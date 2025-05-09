import streamlit as st
import torch
import torchvision.transforms.v2 as transforms
from torchvision.models import vgg16
from torch import nn
from PIL import Image
import os

# Define the class labels (should match your training labels)
DATA_LABELS = ["freshapples", "freshbanana", "freshoranges", "rottenapples", "rottenbanana", "rottenoranges"]

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FruitClassifier(nn.Module):
    def __init__(self):
        super(FruitClassifier, self).__init__()
        N_CLASSES = 6
        vgg_model = vgg16(weights=None)  # Initialize without pre-trained weights
        vgg_model.requires_grad_(False)  # Freeze base model
        self.model = nn.Sequential(
            vgg_model.features,
            vgg_model.avgpool,
            nn.Flatten(),
            vgg_model.classifier[0:3],
            nn.Linear(4096, 500),
            nn.ReLU(),
            nn.Linear(500, N_CLASSES)
        )

    def forward(self, x):
        return self.model(x)

# Load the trained model
@st.cache_resource
def load_model():
    model_path = os.path.join("model", "fruit_classification_model.pth")
    model = torch.load(model_path, map_location=DEVICE,weights_only=False)
    model.eval()
    return model

model = load_model()

# Define image preprocessing (should match your training pipeline)
def preprocess_image(image):
    pre_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])
    img = image.convert("RGB")
    img_tensor = pre_trans(img).unsqueeze(0).to(DEVICE)
    return img_tensor

# Streamlit UI

st.title("Fruit Quality Classifier")
st.write("Upload an image of a fruit to classify it as fresh or rotten.")

# User notice about supported fruits
st.info(
    """
    **Notice:**
    This model is trained to classify only apples, bananas, and oranges as either fresh or rotten. 
    Please upload clear images of these fruits only. Images of other fruits or objects may not be recognized correctly.
    """
)

uploaded_file = st.file_uploader("Choose a fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption=f"{uploaded_file.name}", use_container_width=False)
    st.write("")
    status = st.status("Classifying...", expanded=True)
    with status:
        img_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            predicted_class = DATA_LABELS[predicted.item()]
    # Update status to indicate classification is done
    status.success("Classification complete!", icon="✅")

    # Display the predicted class
    # Format the class name for display
    display_class = predicted_class.replace('fresh', 'Fresh ').replace('rotten', 'Rotten ').title()
    if 'fresh' in predicted_class:
        st.badge("Fresh", icon="✅", color="green")
        st.success(f"Prediction: {display_class}")
    else:
        st.badge("Rotten", icon="❌", color="red")
        st.error(f"Prediction: {display_class}")
