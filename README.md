# KBioX AI — Render Free Deployment

This package keeps the application code on GitHub and downloads the three large trained ML models during the Render build. The models therefore do **not** need to be committed to GitHub.

## Files
- `main.py` — KBioX AI FastAPI application
- `download_models.py` — downloads the real trained models during build
- `render.yaml` — Render service configuration
- `requirements.txt` — Python dependencies
- `.python-version` — Python runtime version
- `models/` — populated automatically during Render build

## Model hosting
Host the three `.pkl` files in a public Hugging Face repository (or another stable HTTPS file host), then set these Render Environment Variables: 

- `MODEL_HIV_URL`
- `MODEL_TB_URL`
- `MODEL_MALARIA_URL`
- `OPENROUTER_API_KEY` (optional, server-side only)

The model URLs are only used during the Render build. Visitors never upload or enter model files.

## Render
Build Command:
`pip install -r requirements.txt && python download_models.py`

Start Command:
`uvicorn main:app --host 0.0.0.0 --port $PORT`

Plan: **Free**

Note: Render Free web services can spin down after inactivity and have monthly usage limits.


## Model hosting
The three trained model files are hosted in the public Hugging Face repository `kharanp/KBioX-AI-Models`. Render downloads them during the build into the local `models/` directory. GitHub does not need to contain the large `.pkl` files.


Render fix: the app uses its writable project directory instead of /content, pins Python 3.13.5/scikit-learn 1.6.1 for the supplied models, and includes xgboost for the TB model bundle.
