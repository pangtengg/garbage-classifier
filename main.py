import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F

# Load a pretrained ResNet model
resnet_model = models.resnet50(weights='IMAGENET1K_V1')
num_classes = 12  # Set this to the number of classes in your garbage classification dataset

# Modify the last layer to fit your number of classes
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, num_classes)

# # Load the model weights if you have trained it
# model_path = r'garbage_classifier.pt'  # Use raw string for Windows path
# resnet_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Ensure model loads correctly
# resnet_model.eval()

resnet_model = torch.load('garbage_classifier-accuracy-94.88.pt', map_location=torch.device('cpu'), weights_only=False)
resnet_model.eval()

# Define transformations for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match the input size of ResNet
    transforms.ToTensor(),
])

class_names = ['Battery', 'Biological', 'Brown-glass', 'Cardboard', 'Clothes', 
                   'Green-glass', 'Metal', 'Paper', 'Plastic', 'Shoes', 'Trash', 'White-glass']
    
recycling_info = {
    'Battery': ('Recycled into new batteries or metal components', 'Orange (E-waste)'),
    'Biological': ('Composted into organic fertilizer', 'Brown (Food waste)'),
    'Brown-glass': ('Melted and reformed into new glass bottles', 'Blue (Recyclables)'),
    'Cardboard': ('Recycled into new paper products', 'Blue (Recyclables)'),
    'Clothes': ('Repurposed or recycled into fabric materials', 'Blue (Recyclables)'),
    'Green-glass': ('Melted and reformed into new glass bottles', 'Blue (Recyclables)'),
    'Metal': ('Recycled into new metal products', 'Blue (Recyclables)'),
    'Paper': ('Recycled into new paper products', 'Blue (Recyclables)'),
    'Plastic': ('Recycled into plastic pellets for new products', 'Blue (Recyclables)'),
    'Shoes': ('Repurposed or shredded for material reuse', 'General Waste (Black)'),
    'Trash': ('Disposed in landfills or incinerated', 'General Waste (Black)'),
    'White-glass': ('Melted and reformed into new glass bottles', 'Blue (Recyclables)'),
}

# Streamlit UI
st.title('Welcome to Garbage Classifier!')
st.markdown("""This model helps in classifying household waste into recyclable categories, aiding in effective waste management. The 12 available classes are: Battery, Biological, Brown Glass, Cardboard, Clothes, Green Glass, Metal, Paper, Plastic, Shoes, Trash, and White Glass.""")
 
uploaded_file = st.file_uploader("Upload an image of garbage", type=["jpg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file).convert("RGB")  # Convert to RGB
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform the prediction
    with torch.no_grad():
        output = resnet_model(input_tensor)
        _, predicted = torch.max(output, 1)

         # Calculate softmax probabilities
        probabilities = F.softmax(output, dim=1)
        predicted_prob = probabilities[0][predicted].item() * 100  # Convert to percentage


    # Display the predicted class
    predicted_class = class_names[predicted.item()]
    recycling_info, bin_color = recycling_info[predicted_class]

    st.write(f"**Predicted class   :** {predicted_class}")
    st.write(f"**Recycling Process :** {recycling_info}")
    st.write(f"**Bin Color         :** {bin_color}")
    st.write(f"**Predicted accuracy:** {predicted_prob:.2f}%")  