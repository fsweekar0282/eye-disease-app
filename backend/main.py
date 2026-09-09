from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
from torchvision import transforms
from model_def import EyeDiseaseModel
from PIL import Image
import io
import os

app = FastAPI(title="Eye Disease Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "eye_disease_model_v4.pth")

try:
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model = EyeDiseaseModel(num_classes=6)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

CLASSES = [
    "Cataract",
    "Diabetic_Retinopathy",
    "Glaucoma",
    "Keratoconus",
    "Normal",
    "Uveitis",
]

DISEASE_DESCRIPTIONS = {
    "Cataract": "Clouding of the eye's natural lens. Common and treatable with surgery.",
    "Diabetic_Retinopathy": "Damage to retinal blood vessels caused by diabetes.",
    "Glaucoma": "Damage to the optic nerve, often caused by high eye pressure.",
    "Keratoconus": "Progressive thinning of the cornea causing it to bulge outward.",
    "Normal": "No signs of disease detected in this image.",
    "Uveitis": "Inflammation of the uvea (middle layer of the eye).",
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    pred_idx = int(probs.argmax().item())
    predicted_class = CLASSES[pred_idx]
    confidence = round(probs[pred_idx].item() * 100, 1)

    all_scores = {
        cls: round(probs[i].item() * 100, 1)
        for i, cls in enumerate(CLASSES)
    }

    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "description": DISEASE_DESCRIPTIONS[predicted_class],
        "all_scores": all_scores,
        "is_disease": predicted_class != "Normal",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
