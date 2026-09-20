import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms


# ============================================================
# 1. Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


# ============================================================
# 2. Siamese Network
# ============================================================

class SiameseNetwork(nn.Module):

    def __init__(self, freeze_backbone=True):
        super().__init__()

        self.encoder = InceptionResnetV1(
            pretrained="vggface2"
        )

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, image1, image2):

        embedding1 = self.encoder(image1)
        embedding2 = self.encoder(image2)

        embedding1 = F.normalize(
            embedding1,
            p=2,
            dim=1
        )

        embedding2 = F.normalize(
            embedding2,
            p=2,
            dim=1
        )

        difference = torch.abs(
            embedding1 - embedding2
        )

        product = embedding1 * embedding2

        combined = torch.cat(
            [difference, product],
            dim=1
        )

        output = self.classifier(combined)

        return output


# ============================================================
# 3. Load trained model
# ============================================================

model = SiameseNetwork()

model.load_state_dict(
    torch.load(
        "model/best_siamese_model_final.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

print("RelationAI model loaded successfully!")


# ============================================================
# 4. Inference preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ============================================================
# 5. FastAPI application
# ============================================================

app = FastAPI(
    title="RelationAI API",
    description="AI-powered relationship prediction API",
    version="1.0.0"
)


# ============================================================
# 6. CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 7. Home endpoint
# ============================================================

@app.get("/")
def home():

    return {
        "message": "RelationAI API is running",
        "model": "SiameseNetwork",
        "version": "1.0.0"
    }


# ============================================================
# 8. Prediction endpoint
# ============================================================

@app.post("/predict")
async def predict(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Read uploaded images
    # --------------------------------------------------------

    image1_bytes = await image1.read()
    image2_bytes = await image2.read()

    img1 = Image.open(
        io.BytesIO(image1_bytes)
    ).convert("RGB")

    img2 = Image.open(
        io.BytesIO(image2_bytes)
    ).convert("RGB")


    # --------------------------------------------------------
    # Apply preprocessing
    # --------------------------------------------------------

    tensor1 = transform(img1)
    tensor2 = transform(img2)

    # Add batch dimension
    tensor1 = tensor1.unsqueeze(0).to(device)
    tensor2 = tensor2.unsqueeze(0).to(device)


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            tensor1,
            tensor2
        )

        probability = torch.sigmoid(
            output
        ).item()


    # --------------------------------------------------------
    # Convert probability to class
    # --------------------------------------------------------

    # Assuming:
    # 1 = Related
    # 0 = Not Related

    if probability >= 0.5:

        prediction = "Related"
        confidence = probability

    else:

        prediction = "Not Related"
        confidence = 1 - probability


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------
    
    return {
        
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "probability_related": round(probability, 4)
    }