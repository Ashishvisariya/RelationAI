import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1


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

        return self.classifier(combined)


# Device
device = torch.device("cpu")


# Create model
model = SiameseNetwork()


# Load your trained weights
model.load_state_dict(
    torch.load(
        "model/best_siamese_model_final.pth",
        map_location=device
    )
)


# Evaluation modepython test_model.py
model.to(device)
model.eval()


print("✅ RelationAI model loaded successfully!")
print("✅ Model is ready for inference!")