# Eye Disease Classifier

### 🔗 [Try it live →](https://eye-disease-app.vercel.app)

Full-stack web app that classifies clinical eye images into six categories using an EfficientNet-B3 model.

## Overview

| | |
|---|---|
| Frontend | React + Vite, deployed on Vercel |
| Backend | FastAPI + PyTorch, containerized and deployed on Google Cloud Run |
| Model | EfficientNet-B3 (timm) with a custom classification head |
| Classes | Cataract, Diabetic Retinopathy, Glaucoma, Keratoconus, Normal, Uveitis |

Upload an eye image or click a sample, and the app returns a predicted class with confidence scores across all six categories.

## Reconstructing the model architecture

The pretrained weights came from a public Kaggle dataset with no accompanying code. Two files were published: a pickled full model and a raw `state_dict`.

The pickled model was unusable — `torch.load` needs the original class definition to unpickle an `nn.Module`, and that class was never published:

```
AttributeError: Can't get attribute 'EyeDiseaseModel'
```

The `state_dict` had no such dependency, but loading it required knowing the exact architecture. I reconstructed it from the 586 parameter keys:

- Keys like `backbone.conv_stem`, `backbone.bn1`, and `backbone.blocks.0.0.conv_dw` follow **timm** naming rather than torchvision's, identifying the backbone library.
- `backbone.conv_head` with shape `(1536, 384, 1, 1)` and the absence of any `backbone.classifier` keys indicated the backbone was created with `num_classes=0` — a pure feature extractor emitting 1536 features.
- The head was an `nn.Sequential`, and the gaps in its indices revealed the paramless layers. `head.0` carried `weight`, `bias`, and running statistics of shape `(1536,)` — a `BatchNorm1d`. `head.2` had shape `(512, 1536)` — a `Linear`. `head.4` was another `BatchNorm1d` at `(512,)`, and `head.6` was `(6, 512)`, the output layer. Indices 1, 3, and 5 held no parameters, corresponding to Dropout, ReLU, and Dropout.

The resulting definition in `model_def.py` loads the weights with `strict=True` and zero key mismatches.

## Running locally

Backend:

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Download `eye_disease_model_v4.pth` from the Kaggle dataset into `backend/models/`, then:

```bash
uvicorn main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

The frontend expects the API at `http://localhost:8000` by default; override with `VITE_API_URL`.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service status and whether the model loaded |
| `/predict` | POST | Multipart image upload; returns predicted class, confidence, and all six scores |

Interactive docs at `/docs`.

## Limitations

This is a demonstration project, not a diagnostic tool.

The model expects clinical eye images. The retinal conditions were trained on fundus photographs, captured with a specialized camera through a dilated pupil. Photos taken on a phone capture the exterior of the eye and fall outside the training distribution entirely.

There is also no "healthy exterior eye" class — `Normal` appears among the fundus classes. An external photo therefore has no correct label available to it, and the model will assign it a disease class with apparent confidence. This is a structural limitation, not a tuning issue.

For anything concerning your eyes, see an optometrist.
