import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


class ConvNet(nn.Module):
  def __init__(self, num_classes=38):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),  
            nn.Linear(256 * 32 * 32, 512),  
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  
            nn.Linear(512, num_classes),
        )

  def forward(self, x):
      x = self.features(x)
      x = self.classifier(x)
      return x

model = ConvNet()
model.load_state_dict(torch.load('cnn.pth', map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])


st.title("Plant Disease Detection using Pytorch")
st.write("Upload an image of a leaf to detect its health status.")

classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',\
            'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',\
            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',\
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',\
            'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',\
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',\
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',\
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', \
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',\
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',\
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',\
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',\
            'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',\
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0) 

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted_class = torch.max(output, 1)
        st.write(f"Predicted Class: {classes[predicted_class.item()]}")
