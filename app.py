from flask import Flask, request, jsonify
import torch
import os
import torchvision.transforms as T
from PIL import Image
import io
import base64
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torch.serialization import safe_globals
import torchvision

app = Flask(__name__)
weights = MobileNet_V3_Large_Weights.DEFAULT
model = mobilenet_v3_large(weights=weights)

# Replace classifier for 5 classes
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 5)
# Load model
path = 'best_model_full (1).pth'

try:
    with safe_globals([torchvision.models.mobilenetv3.MobileNetV3]):
        model = torch.load(path, weights_only=False, map_location=torch.device('cpu'))
    model.eval()
    model_loaded = True
except Exception as e:
    print(f"Warning: Could not load model from {path}. Error: {e}")
    print("Using dummy predictions for demo purposes.")
    model_loaded = False
# Transform

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class_names = {
    0: "Bacterial Pneumonia",
    1: "Corona Virus Disease",
    2: "Normal",
    3: "Tuberculosis",
    4: "Viral Pneumonia"
}
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = float(probs[0][pred_class]) * 100

        return jsonify({
            'class': class_names[pred_class],
            'confidence': round(confidence, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)