import os
from pathlib import Path
import requests

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HF_BASE = "https://huggingface.co/kharanp/KBioX-AI-Models/resolve/main"

# Public model URLs are built in so Render can download the real trained models
# during the build. Environment variables can still override them if needed.
MODELS = {
    "HIV_RF_Model.pkl": os.environ.get(
        "MODEL_HIV_URL", f"{HF_BASE}/HIV_RF_Model.pkl"
    ).strip(),
    "Tuberculosis.pkl": os.environ.get(
        "MODEL_TB_URL", f"{HF_BASE}/Tuberculosis.pkl"
    ).strip(),
    "Malaria_RF_Model.pkl": os.environ.get(
        "MODEL_MALARIA_URL", f"{HF_BASE}/Malaria_RF_Model.pkl"
    ).strip(),
}


def download(name, url):
    if not url:
        print(f"[MODEL] {name}: URL not configured; leaving file absent.")
        return
    dest = MODEL_DIR / name
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[MODEL] Downloading {name}...")
    with requests.get(url, stream=True, timeout=(30, 600), allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", "0") or 0)
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"[MODEL] {name}: {done / total * 100:.1f}%", end="\r")
    tmp.replace(dest)
    print(f"\n[MODEL] Ready: {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")

for name, url in MODELS.items():
    download(name, url)
