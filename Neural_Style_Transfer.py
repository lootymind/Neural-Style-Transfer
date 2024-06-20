import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import save_image
import streamlit as st

# Define the model and helper functions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.req_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.req_features:
                features.append(x)
        return features

def calc_content_loss(gen_feat, orig_feat):
    content_l = torch.mean((gen_feat - orig_feat) ** 2)
    return content_l

def calc_style_loss(gen, style):
    batch_size, channel, height, width = gen.shape
    G = torch.mm(gen.view(channel, height * width), gen.view(channel, height * width).t())
    A = torch.mm(style.view(channel, height * width), style.view(channel, height * width).t())
    style_l = torch.mean((G - A) ** 2)
    return style_l

alpha = 25
beta = 120

def calculate_loss(gen_features, orig_features, style_features):
    style_loss = content_loss = 0
    for gen, cont, style in zip(gen_features, orig_features, style_features):
        content_loss += calc_content_loss(gen, cont)
        style_loss += calc_style_loss(gen, style)
    total_loss = alpha * content_loss + beta * style_loss
    return total_loss

def train_model(epochs, generated_image, original_image, style_image, model):
    optimizer = optim.Adam([generated_image], lr=0.003)
    start_epoch = st.session_state.get('epoch', 0)
    for e in range(start_epoch, epochs):
        gen_features = model(generated_image)
        orig_features = model(original_image)
        style_features = model(style_image)
        total_loss = calculate_loss(gen_features, orig_features, style_features)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        st.write(f"Epoch [{e}/{epochs}] - Loss: {total_loss.item()}")
        st.session_state.epoch = e + 1  # Update Streamlit session state
        print(f"Epoch [{e}/{epochs}] - Loss: {total_loss.item()}")  # For debugging
        print(f"Epoch {e} completed")  # For debugging

# Initialize the model
model = VGG().to(device).eval()

# Streamlit UI

st.title("Image Style Transfer")

style_image_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])
content_image_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])

if style_image_file and content_image_file:
    style_image = Image.open(style_image_file).convert('RGB')
    content_image = Image.open(content_image_file).convert('RGB')

    st.image(style_image, caption="Style Image", use_column_width=True)
    st.image(content_image, caption="Content Image", use_column_width=True)

    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])

    style_image_tensor = transform(style_image).unsqueeze(0).to(device, torch.float)
    content_image_tensor = transform(content_image).unsqueeze(0).to(device, torch.float)
    generated_image_tensor = content_image_tensor.clone().requires_grad_(True)

    if st.button("Start Training"):
        st.write("Training started...")
        print("Training started...")  # For debugging
        train_model(100, generated_image_tensor, content_image_tensor, style_image_tensor, model)
        st.write("Training completed!")
        print("Training completed!")  # For debugging
        
        generated_image = generated_image_tensor.clone().detach().cpu().squeeze(0)
        save_image(generated_image, "generated_image.png")
        
        # Convert the tensor to a PIL image for display in Streamlit
        generated_image_pil = transforms.ToPILImage()(generated_image)
        
        st.image(generated_image_pil, caption="Generated Image", use_column_width=True)
        
        if st.button("Download Generated Image"):
            with open("generated_image.png", "rb") as file:
                btn = st.download_button(
                    label="Download",
                    data=file,
                    file_name="generated_image.png",
                    mime="image/png"
                )

