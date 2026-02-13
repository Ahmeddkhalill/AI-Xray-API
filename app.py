import torch
import torchvision.transforms as T
from PIL import Image
import gradio as gr
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

# Load model architecture
weights = MobileNet_V3_Large_Weights.DEFAULT
model = mobilenet_v3_large(weights=weights)
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 5)

# Load trained weights
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

# Transform
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

class_names = {
    0: "Bacterial Pneumonia",
    1: "Corona Virus Disease",
    2: "Normal",
    3: "Tuberculosis",
    4: "Viral Pneumonia"
}

def predict(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = float(probs[0][pred]) * 100

    return f"{class_names[pred]} ({confidence:.2f}%)"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Chest X-Ray Classifier",
    description="Upload a chest X-ray image to get prediction."
)

demo.launch()