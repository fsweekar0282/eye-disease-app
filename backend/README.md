# Eye Disease Classifier - Backend

FastAPI service that classifies eye images into six categories using an EfficientNet-B3 model.

## Classes

Cataract, Diabetic Retinopathy, Glaucoma, Keratoconus, Normal, Uveitis

## Setup

1. Create and activate a virtual environment (python -m venv venv, then source venv/bin/activate)
2. Install dependencies: pip install -r requirements.txt
3. Download eye_disease_model_v4.pth from the Kaggle dataset and place it in backend/models/
4. Run the server: uvicorn main:app --reload --port 8000

## Endpoints

- GET /health - service and model status
- POST /predict - multipart image upload, returns predicted class with confidence scores

Interactive docs available at http://localhost:8000/docs

## Note

This is a demonstration project, not a diagnostic tool. The model expects clinical eye images (fundus photographs for retinal conditions, anterior segment photos for corneal conditions) and should not be used for medical decisions.

## Model

Architecture reconstructed from the published state_dict: a timm efficientnet_b3 backbone (1536 features) with a custom classification head. See model_def.py
