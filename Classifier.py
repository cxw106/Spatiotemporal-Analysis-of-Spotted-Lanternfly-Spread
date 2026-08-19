import os
import torch
import torch.nn as nn
from PIL import Image
import open_clip
import csv


# 1. load
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, _, preprocess_val = open_clip.create_model_and_transforms(
    "hf-hub:imageomics/bioclip"
)
model = model.visual.to(device)

# 冻结 backbone
for param in model.parameters():
    param.requires_grad = False

with torch.no_grad():
    dummy = torch.zeros(1, 3, 224, 224).to(device)
    feat_dim = model(dummy).shape[-1]

# Load classification header
classifier = nn.Linear(feat_dim, 2).to(device)
checkpoint = torch.load("best_classifier.pt", map_location=device)
classifier.load_state_dict(checkpoint["classifier_state"])

class BioCLIPClassifier(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone   = backbone
        self.classifier = classifier

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return self.classifier(features)

clf_model = BioCLIPClassifier(model, classifier).to(device)
clf_model.eval()



# 2. process image and create dataset
IMAGE_DIR   = " "   # your image address
OUTPUT_CSV  = " "    #SLF occurrence data
CONFIDENCE_THRESHOLD = 0.8         # Confidence threshold

results = []

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
]

print(f"the number of image: {len(image_files)} ")

for i, fname in enumerate(image_files):
    img_path = os.path.join(IMAGE_DIR, fname)

    try:
        img = Image.open(img_path).convert("RGB")
        tensor = preprocess_val(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = clf_model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]
            pred   = probs.argmax().item()
            conf   = probs[pred].item()

        label = "SLF" if pred == 1 else "Not-SLF"

        # > threshold
        if conf < CONFIDENCE_THRESHOLD:
            label = "Uncertain"

        results.append({
            "filename":   fname,
            "prediction": label,
            "confidence": round(conf, 4),
            "slf_prob":   round(probs[1].item(), 4),
        })

        if (i + 1) % 50 == 0:
            print(f" rate of progress: {i+1}/{len(image_files)}")

    except Exception as e:
        print(f" jump: {fname}: {e}")


# 3. 保存结果
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "prediction", "confidence", "slf_prob"])
    writer.writeheader()
    writer.writerows(results)

# 统计
slf_count       = sum(1 for r in results if r["prediction"] == "SLF")
not_slf_count   = sum(1 for r in results if r["prediction"] == "Not-SLF")
uncertain_count = sum(1 for r in results if r["prediction"] == "Uncertain")

print(f"\nComplete")
print(f"  SLF:      {slf_count} ")
print(f"  Not-SLF:  {not_slf_count} ")
print(f"  Uncertain:{uncertain_count} ")
print(f"Saved as {OUTPUT_CSV}")