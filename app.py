import os
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.cm as cm

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Retinal OCT Classifier",
    page_icon="👁️",
    layout="wide"
)

# ---------------- DESIGN ----------------
st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #f4f7fb;
    color: #172b4d;
}

/* Text visibility */
h1, h2, h3, p, label, .stMarkdown {
    color: #172b4d !important;
}

/* Main heading */
h1 {
    font-weight: 800;
}

/* General buttons */
.stButton > button {
    background-color: #1769aa !important;
    color: white !important;
    border: none !important;
    border-radius: 8px;
    padding: 10px 22px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #0d477a !important;
    color: white !important;
}

/* Entire upload box */
[data-testid="stFileUploader"] {
    background-color: white !important;
    border: 2px solid red !important;
    border-radius: 10px;
    padding: 20px;
}

/* Upload section */
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
    background-color: white !important;
    border: none !important;
}

/* Upload text */
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] label {
    color: #172b4d !important;
}

/* Upload button - WHITE */
[data-testid="stFileUploader"] button {
    background-color: white !important;
    color: #172b4d !important;
    border: 1px solid #1769aa !important;
    border-radius: 7px !important;
    font-weight: 600;
}

/* Upload button hover */
[data-testid="stFileUploader"] button:hover {
    background-color: #f0f4f8 !important;
    color: #172b4d !important;
}

/* Upload icon */
[data-testid="stFileUploader"] button svg {
    color: #172b4d !important;
    fill: none !important;
}

/* Prediction metric cards */
[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #dce5ef;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 8px;
}

/* Dividers */
hr {
    border-color: #dce5ef;
}

</style>
""", unsafe_allow_html=True)

# ---------------- MODEL DEFINITIONS ----------------

class CustomCNN(nn.Module):

    def __init__(self, num_classes=4, dropout=0.5):
        super().__init__()

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(name, num_classes=4, freeze_backbone=False, pretrained=True):

    if name == "custom_cnn":
        return CustomCNN(num_classes)

    weights = "DEFAULT" if pretrained else None

    if name == "resnet18":
        model = models.resnet18(weights=weights)
    elif name == "vgg16":
        model = models.vgg16(weights=weights)
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights)
    elif name == "densenet121":
        model = models.densenet121(weights=weights)
    else:
        raise ValueError("Unknown model: " + name)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    if name == "resnet18":
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, num_classes)
        )

    elif name == "vgg16":
        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features, num_classes
        )

    else:
        in_f = (
            model.classifier[1].in_features
            if name == "mobilenet_v2"
            else model.classifier.in_features
        )

        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_f, num_classes)
        )

    return model


# ---------------- GRAD-CAM ----------------

def get_target_layer(model, model_name):

    if model_name == "custom_cnn":
        return model.features[-1]
    if model_name == "resnet18":
        return model.layer4
    if model_name == "vgg16":
        return model.features[29]
    if model_name == "mobilenet_v2":
        return model.features[-1]
    if model_name == "densenet121":
        return model.features

    raise ValueError("No Grad-CAM layer for " + model_name)


class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.activations = None
        self.gradients = None
        self.handle = target_layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        out = output.clone()
        self.activations = out.detach().clone()
        out.register_hook(
            lambda grad: setattr(self, "gradients", grad.detach())
        )
        return out

    def remove(self):
        self.handle.remove()

    def __call__(self, input_tensor, class_idx=None):

        x = input_tensor.clone().requires_grad_(True)

        with torch.enable_grad():
            scores = self.model(x)

            if class_idx is None:
                class_idx = int(scores.argmax(1))

            self.model.zero_grad()
            scores[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        cam = F.relu(
            (weights * self.activations).sum(dim=1, keepdim=True)
        )

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False
        )[0, 0]

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy(), class_idx


def overlay_heatmap(image_rgb, heatmap, alpha=0.45):
    colored = cm.jet(heatmap)[..., :3]
    return np.clip(
        (1 - alpha) * image_rgb + alpha * colored, 0, 1
    )


# ---------------- APP SETTINGS ----------------

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "best_model.pt"
)

IMAGE_SIZE = 224
MAX_FILE_MB = 10

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

eval_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)


@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    ckpt = torch.load(MODEL_PATH, map_location="cpu")

    model = build_model(
        ckpt["model_name"],
        pretrained=False
    )

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return model, ckpt["class_names"], ckpt["model_name"]


# ---------------- MAIN PAGE ----------------

st.title("👁️ Retinal OCT Disease Detection")

st.markdown(
    "### Deep Learning-Based Retinal Disease Detection and Classification"
)

st.warning(
    "**Disclaimer:** This is an academic prototype designed for learning and research, not for real-world medical diagnosis "
    
)

loaded = load_model()

if loaded is None:
    st.error(
        "`best_model.pt` was not found. Place the trained model "
        "in the same folder as this application."
    )
    st.stop()

model, class_names, model_name = loaded

st.success(f"Model loaded successfully: {model_name}")

st.caption(f"Available classes: {', '.join(class_names)}")

st.markdown("---")

st.subheader("📤 Upload OCT Image")

uploaded = st.file_uploader(
    "Choose a retinal OCT image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded is None:
    st.info("Please upload an OCT image to begin analysis.")
    st.stop()


# ---------------- IMAGE VALIDATION ----------------

if uploaded.size > MAX_FILE_MB * 1024 * 1024:
    st.error(f"File exceeds {MAX_FILE_MB} MB.")
    st.stop()

try:
    image = Image.open(uploaded).convert("RGB")

except (UnidentifiedImageError, OSError):
    st.error("Invalid image. Please upload a valid JPG or PNG.")
    st.stop()

if min(image.size) < 64:
    st.error("Image is too small. Minimum size is 64 pixels.")
    st.stop()


# ---------------- PREDICTION ----------------

st.markdown("---")

st.subheader("🔍 Image Analysis")

tensor = eval_tf(image).unsqueeze(0)

with torch.no_grad():
    probs = torch.softmax(model(tensor), dim=1)[0].numpy()

pred_idx = int(np.argmax(probs))
confidence = float(probs[pred_idx]) * 100


# ---------------- DISPLAY RESULTS ----------------

col1, col2, col3 = st.columns(3)

with col1:

    st.subheader("📷 Uploaded Image")
    st.image(image, width="stretch")

with col2:

    st.subheader("🧠 Prediction")

    st.metric("Predicted Class", class_names[pred_idx])
    st.metric("Confidence", f"{confidence:.1f}%")

    if confidence < 60:
        st.info("Low confidence. Interpret this result cautiously.")

    st.markdown("**Class Probabilities**")

    st.bar_chart({
        c: float(p)
        for c, p in zip(class_names, probs)
    })

with col3:

    st.subheader("🔥 Grad-CAM")

    try:

        cam_tool = GradCAM(
            model,
            get_target_layer(model, model_name)
        )

        try:
            heatmap, _ = cam_tool(tensor, pred_idx)
        finally:
            cam_tool.remove()

        base = (
            denormalize(tensor[0])
            .permute(1, 2, 0)
            .numpy()
        )

        st.image(
            overlay_heatmap(base, heatmap),
            width="stretch",
            caption="Red areas indicate stronger model attention"
        )

    except Exception as err:
        st.info(f"Grad-CAM unavailable: {err}")


# ---------------- FOOTER ----------------

st.markdown("---")

st.caption(
    "Deep Learning-Based Retinal Disease Detection and Classification | AI/ML Educational Project"
)
