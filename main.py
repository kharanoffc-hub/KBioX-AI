# ================================================================
# KBioX AI v2.7 — ONE-CELL GOOGLE COLAB VERSION
# AI Molecular Bioactivity / Cheminformatics Dashboard
# ================================================================
# Paste/run this ENTIRE cell in Google Colab.
# Optional model files:
#   /content/HIV_RF_Model.pkl
#   /content/Tuberculosis.pkl
#   /content/Malaria_RF_Model.pkl
#
# Optional OpenRouter:
#   The script asks for the key once at startup. Public users do not need one.
# ================================================================

import sys, subprocess, os, io, math, json, warnings, threading, time, socket, re
from pathlib import Path

# ---------- Install only missing packages ----------
packages = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "reportlab": "reportlab",
    "requests": "requests",
    "joblib": "joblib",
    "scikit-learn": "sklearn",
    "rdkit": "rdkit",
}
for pip_name, import_name in packages.items():
    try:
        __import__(import_name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])

warnings.filterwarnings("ignore")

import numpy as np
import requests
import re
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, Draw, AllChem
from rdkit.Chem import rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image as PDFImage, ListFlowable, ListItem

# ================================================================
# 1. DIRECTORIES / CONFIG
# ================================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
TEMP_DIR = BASE_DIR / "temp"

for d in (MODEL_DIR, REPORT_DIR, TEMP_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Server-side AI configuration
# The public website NEVER asks visitors for this key.
# In Colab, enter it once at startup. Type SKIP (or press Enter) to continue
# without AI. The key remains only in this backend runtime.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()

if not OPENROUTER_API_KEY and sys.stdin.isatty():
    print("\n================ KBioX AI CONFIGURATION ================")
    print("Your API key is optional.")
    print("Enter your OpenRouter key, or type SKIP and press Enter to continue.")
    print("Public visitors will NEVER be asked for the key.\n")
    try:
        OPENROUTER_API_KEY = input("OpenRouter API key (or SKIP): ").strip()
    except (EOFError, KeyboardInterrupt):
        OPENROUTER_API_KEY = ""
    if OPENROUTER_API_KEY.upper() in {"SKIP", "NO", "NONE"}:
        OPENROUTER_API_KEY = ""

if OPENROUTER_API_KEY:
    print("✓ AI API key loaded. Scientific Interpretation + Copilot will use OpenRouter.")
else:
    print("✓ Continuing without AI API. Scientific Interpretation will show API NOT LINKED.")

OPENROUTER_MODEL = "openai/gpt-4o-mini"

PORT = int(os.environ.get("PORT", "8000"))

# ================================================================
# 2. RDKit FEATURE ENGINE
# ================================================================
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=512
)

def descriptor_vector(mol):
    return np.asarray([
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        rdMolDescriptors.CalcNumHeavyAtoms(mol),
    ], dtype=float)

def fingerprint_vector(mol):
    return np.asarray(
        MORGAN_GENERATOR.GetFingerprint(mol),
        dtype=float
    )

def all_features(mol):
    return np.concatenate([descriptor_vector(mol), fingerprint_vector(mol)])

def adapt_features_to_model(mol, model):
    """
    Automatically adapts features to the number expected by the saved model.

    Supported common cases:
      10   -> descriptors only
      512  -> Morgan fingerprint only
      522  -> descriptors + Morgan fingerprint

    If the model has another feature count, a clear error is returned.
    """
    expected = getattr(model, "n_features_in_", None)

    desc = descriptor_vector(mol)
    fp = fingerprint_vector(mol)
    combined = np.concatenate([desc, fp])

    if expected is None:
        return combined.reshape(1, -1)

    expected = int(expected)

    if expected == len(desc):
        return desc.reshape(1, -1)

    if expected == len(fp):
        return fp.reshape(1, -1)

    if expected == len(combined):
        return combined.reshape(1, -1)

    # Some sklearn pipelines expose the dimension through the first step.
    try:
        first = model[0]
        inner_expected = getattr(first, "n_features_in_", None)
        if inner_expected is not None:
            inner_expected = int(inner_expected)
            if inner_expected == len(desc):
                return desc.reshape(1, -1)
            if inner_expected == len(fp):
                return fp.reshape(1, -1)
            if inner_expected == len(combined):
                return combined.reshape(1, -1)
    except Exception:
        pass

    raise ValueError(
        f"Saved model expects {expected} features, but this app provides "
        f"10 descriptors, 512 Morgan bits, or 522 combined features."
    )

# ================================================================
# 3. MODEL LOADING
# ================================================================
def load_first_existing(paths):
    for path in paths:
        p = Path(path)
        if p.exists():
            try:
                return joblib.load(p), str(p)
            except Exception as e:
                print(f"[MODEL] Could not load {p}: {e}")
    return None, None

HIV_PACKAGE, HIV_PATH = load_first_existing([
    str(MODEL_DIR / "HIV_RF_Model.pkl")
])

TB_PACKAGE, TB_PATH = load_first_existing([
    str(MODEL_DIR / "Tuberculosis.pkl")
])

MALARIA_PACKAGE, MALARIA_PATH = load_first_existing([
    str(MODEL_DIR / "Malaria_RF_Model.pkl")
])

MODELS = {
    "HIV": HIV_PACKAGE,
    "TB": TB_PACKAGE,
    "Malaria": MALARIA_PACKAGE
}

MODEL_PATHS = {
    "HIV": HIV_PATH,
    "TB": TB_PATH,
    "Malaria": MALARIA_PATH
}

def unwrap_model(package):
    if package is None:
        return None

    if hasattr(package, "predict_proba") or hasattr(package, "predict"):
        return package

    if isinstance(package, dict):
        for key in (
            "model", "rf_model", "classifier",
            "estimator", "pipeline", "best_model"
        ):
            obj = package.get(key)
            if hasattr(obj, "predict_proba") or hasattr(obj, "predict"):
                return obj

    if isinstance(package, (tuple, list)):
        for obj in package:
            if hasattr(obj, "predict_proba") or hasattr(obj, "predict"):
                return obj

    return None

# ================================================================
# 4. CHEMICAL UTILITIES
# ================================================================
PRESETS = {
    "CC(=O)Oc1ccccc1C(=O)O": "Acetylsalicylic Acid (Aspirin)",
    "CC(OCP(=O)(O)O)Cn1cnc2c(N)ncnc12": "Tenofovir",
    "Nc1ccn([C@H]2CS[C@H](CO)O2)c(=O)n1": "Lamivudine (3TC)",
    "NC(=O)c1ccncc1": "Isoniazid",
    "CC1CCC2C(C)C(=O)OC3OC4(C)CCC1C23OO4": "Artemisinin",
    "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12": "Chloroquine",
}

def fetch_compound_name(smiles):
    if smiles in PRESETS:
        return PRESETS[smiles]

    try:
        encoded = requests.utils.quote(smiles, safe="")
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
            f"smiles/{encoded}/property/Title,IUPACName/JSON"
        )
        r = requests.get(url, timeout=5)
        if r.ok:
            props = r.json().get("PropertyTable", {}).get("Properties", [])
            if props:
                return props[0].get("Title") or props[0].get("IUPACName") or "Unknown compound"
    except Exception:
        pass

    return "Novel / Unresolved Compound"

def make_2d_svg(mol):
    d2d = rdMolDraw2D.MolDraw2DSVG(500, 300)
    opts = d2d.drawOptions()
    opts.clearBackground = True
    opts.bondLineWidth = 2.0

    # RDKit versions differ slightly in drawOptions APIs,
    # so only use universally supported drawing operations.
    try:
        d2d.DrawMolecule(mol)
    except Exception:
        Chem.rdDepictor.Compute2DCoords(mol)
        d2d.DrawMolecule(mol)

    d2d.FinishDrawing()
    return d2d.GetDrawingText()

def get_geometry(mol):
    mol3d = Chem.AddHs(Chem.Mol(mol))
    embedded = False

    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        code = AllChem.EmbedMolecule(mol3d, params)
        embedded = (code == 0)
    except Exception:
        try:
            code = AllChem.EmbedMolecule(mol3d, randomSeed=42)
            embedded = (code == 0)
        except Exception:
            embedded = False

    if embedded:
        try:
            AllChem.UFFOptimizeMolecule(mol3d, maxIters=100)
        except Exception:
            pass
    else:
        # Fallback: create 2D coordinates and use z=0.
        try:
            AllChem.Compute2DCoords(mol3d)
        except Exception:
            pass

    conf = mol3d.GetConformer()

    element_colors = {
        "C": "#38bdf8",
        "O": "#ef4444",
        "N": "#818cf8",
        "S": "#eab308",
        "P": "#fbbf24",
        "CL": "#34d399",
        "F": "#22d3ee",
        "BR": "#f87171",
        "I": "#c084fc",
        "H": "#94a3b8",
    }

    atoms = []
    for i, atom in enumerate(mol3d.GetAtoms()):
        p = conf.GetAtomPosition(i)
        sym = atom.GetSymbol().upper()
        atoms.append({
            "element": atom.GetSymbol(),
            "x": float(p.x * 24),
            "y": float(p.y * 24),
            "z": float(p.z * 24),
            "color": element_colors.get(sym, "#cbd5e1"),
            "size": 3.5 if sym == "H" else 7.5
        })

    bonds = []
    for b in mol3d.GetBonds():
        bonds.append([
            int(b.GetBeginAtomIdx()),
            int(b.GetEndAtomIdx())
        ])

    return atoms, bonds, make_2d_svg(mol)

def calculate_properties(mol):
    mw = float(round(Descriptors.MolWt(mol), 2))
    logp = float(round(Crippen.MolLogP(mol), 2))
    tpsa = float(round(rdMolDescriptors.CalcTPSA(mol), 2))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    rotb = int(Lipinski.NumRotatableBonds(mol))
    qed = float(round(QED.qed(mol), 3))

    violations = int(sum([
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10
    ]))

    return {
        "mw": mw,
        "logp": logp,
        "hbd": hbd,
        "hba": hba,
        "tpsa": tpsa,
        "rotb": rotb,
        "qed": qed,
        "heavyAtoms": int(Lipinski.HeavyAtomCount(mol)),
        "violations": violations
    }

# ================================================================
# 5. SAFE PREDICTION ENGINE
# ================================================================
def heuristic_prediction(props):
    """
    Used ONLY when no compatible trained model is available.
    It is clearly labelled as heuristic in the API response.
    """
    favorable = (
        props["mw"] <= 460
        and 1.0 <= props["logp"] <= 4.2
        and props["violations"] == 0
    )

    if favorable:
        return True, 72.0, 6.7, "heuristic"
    return False, 31.0, 4.2, "heuristic"

def run_prediction(mol, target):
    target_key = (
        "TB" if "TB" in target.upper()
        else "MALARIA" if "MALARIA" in target.upper()
        else "HIV"
    )

    package = MODELS.get(target_key)
    model = unwrap_model(package)
    props = calculate_properties(mol)

    if model is None:
        active, confidence, pic50, source = heuristic_prediction(props)
        return active, confidence, pic50, source, None

    try:
        X = adapt_features_to_model(mol, model)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[0]
            if len(probabilities) == 2:
                probability = float(probabilities[1])
            else:
                probability = float(np.max(probabilities))
        else:
            pred = model.predict(X)[0]
            probability = float(pred)

        probability = max(0.0, min(1.0, probability))
        active = probability >= 0.5
        confidence = round(probability * 100, 1)

        # IMPORTANT:
        # This pIC50 is a UI estimate unless the saved model was actually
        # trained to predict pIC50. It must not be interpreted as measured pIC50.
        pic50 = round(4.0 + probability * 4.5, 2)

        return active, confidence, pic50, "trained_model", getattr(model, "n_features_in_", None)

    except Exception as e:
        active, confidence, pic50, source = heuristic_prediction(props)
        return active, confidence, pic50, "heuristic_fallback", str(e)

def make_shap_like_attribution(props, active):
    # These are descriptor-level explanatory heuristics, NOT TreeSHAP values.
    # True SHAP requires running SHAP against the actual trained model.
    return [
        {
            "fragment": "Aromatic / hydrophobic contribution",
            "value": round(0.30 if active else -0.16, 2)
        },
        {
            "fragment": "H-bond acceptor contribution",
            "value": round(0.22 if props["hba"] >= 2 else -0.16, 2)
        },
        {
            "fragment": "Calculated LogP",
            "value": round(
                0.16 if 0.5 <= props["logp"] <= 3.8 else -0.24, 2
            )
        },
        {
            "fragment": "TPSA contribution",
            "value": round(
                0.10 if props["tpsa"] <= 110 else -0.18, 2
            )
        },
        {
            "fragment": "Rotatable-bond flexibility",
            "value": round(
                0.08 if props["rotb"] <= 5 else -0.12, 2
            )
        }
    ]

# ================================================================
# 6. FASTAPI
# ================================================================
app = FastAPI(title="KBioX AI", version="2.7")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ScreeningRequest(BaseModel):
    smiles: str
    target: str

class CopilotRequest(BaseModel):
    query: str
    target: str
    smiles: str
    history: list = []

class PDFRequest(BaseModel):
    target: str
    smiles: str
    name: str
    properties: dict
    prediction: dict
    ai_summary: str
    ai_interpretation: str = ""
    ai_keypoints: list = []

@app.get("/api/health")
def health():
    return {
        "status": "online",
        "version": "2.7",
        "models": {
            k: bool(unwrap_model(v))
            for k, v in MODELS.items()
        },
        "aiApiLinked": bool(OPENROUTER_API_KEY and OPENROUTER_API_KEY != "YOUR_OPENROUTER_API_KEY")
    }

def build_local_scientific_summary(target, name, active, confidence, pic50, source, props, attribution):
    """Useful non-LLM summary using only values actually computed by KBioX AI."""
    activity = "predicted active" if active else "predicted inactive / low activity"
    lipinski = "passes the implemented Lipinski screen" if props.get("violations", 0) == 0 else f"has {props.get('violations')} Lipinski violation(s)"
    attr = sorted(attribution or [], key=lambda x: abs(float(x.get("value", x.get("shap", 0)) or 0)), reverse=True)
    top = attr[0].get("fragment", "the calculated descriptor profile") if attr else "the calculated descriptor profile"
    return (
        f"The screening model classified **{name}** as **{activity} against {target}** with a confidence of **{confidence:.1f}%**; the prediction source is **{source}**. "
        f"The estimated pIC50 is **{pic50}**, which is a computational estimate rather than an experimental measurement. "
        f"Its calculated profile is MW **{props.get('mw')} g/mol**, LogP **{props.get('logp')}**, TPSA **{props.get('tpsa')} Å²**, HBD/HBA **{props.get('hbd')}/{props.get('hba')}**, and QED **{props.get('qed')}**; the structure {lipinski}. "
        f"The strongest descriptor-level contribution in the current explanation is **{top}**. These findings support prioritization for further investigation, but they do not establish target binding or biological efficacy without experimental validation."
    )

def generate_ai_interpretation(target, smiles, name, active, confidence, pic50, source, props, attribution):
    """Generate short + detailed scientific interpretation when an LLM API is linked."""
    not_linked = (
        not OPENROUTER_API_KEY
        or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY"
    )
    if not_linked:
        msg = "AI interpretation API not linked — add your OpenRouter API key to enable AI-generated scientific interpretation."
        local = build_local_scientific_summary(target, name, active, confidence, pic50, source, props, attribution)
        keypoints = [
            f"Prediction: **{('active' if active else 'inactive / low activity')}** against **{target}** at **{confidence:.1f}%** confidence; source: **{source}**.",
            f"Calculated profile: **MW {props.get('mw')} g/mol**, **LogP {props.get('logp')}**, **TPSA {props.get('tpsa')} Å²**, with **QED {props.get('qed')}**.",
            f"The current descriptor explanation is led by **{(attribution or [{}])[0].get('fragment', 'descriptor-level features')}**; these are explanatory heuristics, not validated SHAP values.",
            "*Experimental validation is required* before biological activity, mechanism, safety, or therapeutic relevance can be concluded."
        ]
        return local, keypoints, msg + "\n\n" + local

    attr_text = "; ".join(
        f"{x.get('fragment','feature')}: {x.get('value', x.get('shap','N/A'))}"
        for x in (attribution or [])
    )
    prompt = f"""
You are a senior computational medicinal chemist and scientific writer.
Prepare a scientifically cautious interpretation of an in-silico molecular screening result.
Do not invent docking scores, binding constants, literature evidence, experimental findings,
mechanisms, toxicity results, or validated ADMET endpoints that are not supplied.
Clearly distinguish calculated descriptors, model predictions, heuristics, and limitations.

Target: {target}
Compound: {name}
SMILES: {smiles}
Classification: {'predicted active' if active else 'predicted inactive/low activity'}
Confidence: {confidence}%
Estimated pIC50: {pic50}
Prediction source: {source}
Properties: {json.dumps(props, ensure_ascii=False)}
Descriptor-level attribution: {attr_text}

Return exactly three labeled parts in Markdown:
SHORT:
Write a useful scientific summary of 5-7 sentences for the main website. It must interpret the actual values supplied above rather than merely restating them. Explain the prediction/classification and confidence, what the estimated pIC50 can and cannot mean, the most relevant physicochemical findings, the leading explanation signal, and what the combined evidence means for prioritizing this molecule for further study. Explicitly distinguish model evidence from experimental evidence. Do not use generic filler such as “recent screening suggests” without explaining the scientific significance of the result.

KEYPOINTS:
Write exactly 4 useful scientific key points as bullet points. Each bullet must begin with - and contain one substantive finding or implication using the supplied values. Avoid repeating the same sentence in different words.

DETAILED:
Write a substantially fuller postgraduate-level scientific interpretation (about 500-700 words) for a computational screening dossier/PDF. Format it as clearly separated numbered sections, for example:
1. Overall computational finding
2. Physicochemical interpretation
3. Explainability / descriptor interpretation
4. Target-related implications
5. Limitations and experimental validation
Put each numbered section on its own line with a blank line between sections. Use bullet points where useful. Use **bold** for important findings and *italicize* important caveats. Do not put all sections into one paragraph.
Discuss the prediction in relation to the calculated physicochemical profile, explainability signals, plausible implications for the selected target, and limitations/need for experimental validation.
Do not overstate causality or efficacy. Do not invent evidence.
"""
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": "You are a senior computational medicinal chemist and scientific manuscript reviewer."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1600,
            "temperature": 0.2
        }
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=45)
        if not r.ok:
            msg = f"AI interpretation unavailable (OpenRouter HTTP {r.status_code}). Check the API key and model availability."
            return msg, [], msg
        content = str(r.json()["choices"][0]["message"]["content"]).strip()
        short = content
        keypoints = []
        detailed = content
        if "SHORT:" in content:
            remainder = content.split("SHORT:", 1)[1]
            if "KEYPOINTS:" in remainder:
                short, remainder = remainder.split("KEYPOINTS:", 1)
                if "DETAILED:" in remainder:
                    kp_text, detailed = remainder.split("DETAILED:", 1)
                else:
                    kp_text = remainder
                keypoints = [ln.strip()[2:].strip() for ln in kp_text.splitlines() if ln.strip().startswith("-")]
            elif "DETAILED:" in remainder:
                short, detailed = remainder.split("DETAILED:", 1)
        short = short.strip()
        detailed = detailed.strip()
        return short, keypoints[:6], detailed
    except Exception as e:
        msg = f"AI interpretation unavailable: {str(e)}"
        return msg, [], msg


@app.post("/api/predict")
def predict_endpoint(req: ScreeningRequest):
    smiles = req.smiles.strip()

    if not smiles:
        raise HTTPException(400, "SMILES input is empty.")

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise HTTPException(
            400,
            "Invalid SMILES. RDKit could not parse or sanitize the structure."
        )

    canonical = Chem.MolToSmiles(mol, canonical=True)
    name = fetch_compound_name(canonical)

    props = calculate_properties(mol)
    atoms, bonds, svg2d = get_geometry(mol)

    active, confidence, pic50, source, model_info = run_prediction(
        mol, req.target
    )

    attribution = make_shap_like_attribution(props, active)

    ai_summary, ai_keypoints, ai_interpretation = generate_ai_interpretation(
        req.target, canonical, name, active, confidence, pic50, source, props, attribution
    )

    return {
        "isValid": True,
        "canonicalSmiles": canonical,
        "compoundName": name,
        "isActive": active,
        "confidence": confidence,
        "pic50": pic50,
        "predictionSource": source,
        "modelFeatures": model_info,
        "properties": props,
        "shapFeatures": attribution,
        "geometry": {
            "atoms": atoms,
            "bonds": bonds,
            "svg2d": svg2d
        },
        "aiSummary": ai_summary,
        "aiKeyPoints": ai_keypoints,
        "aiInterpretation": ai_interpretation
    }

@app.post("/api/copilot")
def copilot_endpoint(req: CopilotRequest):
    # Offline fallback
    if (
        not OPENROUTER_API_KEY
        or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY"
    ):
        q = req.query.lower()

        if "mechanism" in q or "binding" in q:
            return {
                "response":
                f"<strong>Binding assessment:</strong> "
                f"for {req.target}, binding should be interpreted from "
                f"target-specific structural evidence such as docking pose, "
                f"active-site contacts, hydrogen bonds, hydrophobic contacts "
                f"and experimentally validated activity."
            }

        if "isostere" in q or "substitut" in q:
            return {
                "response":
                "<strong>SAR assessment:</strong> consider controlled "
                "heteroatom substitution, polarity tuning and steric "
                "modification while monitoring potency, permeability and "
                "metabolic stability."
            }

        if "toxic" in q or "metabolic" in q:
            return {
                "response":
                "<strong>ADMET assessment:</strong> inspect reactive "
                "functional groups, CYP liabilities, excessive lipophilicity, "
                "high polarity and structural alerts using dedicated "
                "ADMET/toxicophore tools before drawing conclusions."
            }

        return {
            "response":
            f"<strong>Cheminformatics assessment:</strong> "
            f"<code>{req.smiles}</code> is being considered against "
            f"<em>{req.target}</em>. Add an OpenRouter API key for "
            f"LLM-assisted scientific reasoning."
        }

    prompt = f"""
You are KBioX AI Research AI Copilot, a professional computational medicinal chemistry assistant.
Answer the user's current scientific question using the supplied molecular context.

Target: {req.target}
SMILES: {req.smiles}
Research question: {req.query}

Write like a real scientific AI assistant, not like a generic chatbot.
- Start with a direct 1-2 sentence answer.
- Then provide the most useful scientific reasoning.
- Use Markdown formatting: **bold** for important findings, *italics* for caveats, numbered points for mechanisms/workflows, and bullet points for supporting observations.
- Use short paragraphs; never produce one large wall of text.
- When discussing a mechanism, distinguish established biology from hypothesis/inference.
- When discussing SAR, give actionable but scientifically cautious considerations.
- When discussing ADMET, clearly identify what is not actually predicted by this application.
- If the supplied information is insufficient, explicitly say what evidence would be needed.
- Do not invent experimental results, docking scores, binding constants, citations, literature findings, or validated ADMET endpoints that were not supplied.
- Do not claim that heuristic model output proves biological activity.
- Do not repeat the user's question.
- End with a short **Takeaway** when the answer is more than a few sentences.
"""

    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        messages = [
            {
                "role": "system",
                "content": "You are KBioX AI Research AI Copilot, a senior computational medicinal chemist. Be scientifically cautious, structured, useful, and transparent about uncertainty."
            }
        ]
        # Preserve a small amount of recent conversation context so Copilot behaves like a real assistant.
        for item in (req.history or [])[-6:]:
            if isinstance(item, dict) and item.get("role") in {"user", "assistant"} and item.get("content"):
                messages.append({"role": item["role"], "content": str(item["content"])[:3000]})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": messages,
            "max_tokens": 800,
            "temperature": 0.25
        }

        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        if not r.ok:
            return {
                "response":
                f"OpenRouter returned HTTP {r.status_code}. "
                "Check the API key and model availability."
            }

        data = r.json()
        return {
            "response":
            data["choices"][0]["message"]["content"]
        }

    except Exception as e:
        return {
            "response": f"AI Copilot error: {str(e)}"
        }

def markdown_to_pdf_flowables(text, body_style, heading_style, small_style):
    """Convert simple Markdown/numbered scientific prose into readable PDF blocks."""
    if not text:
        return [Paragraph("No scientific interpretation available.", body_style)]
    lines = str(text).replace("\r", "").split("\n")
    out = []
    para = []

    def inline_md(x):
        x = str(x)
        x = x.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        x = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", x)
        x = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", x)
        return x

    def flush():
        nonlocal para
        if para:
            txt = " ".join(x.strip() for x in para if x.strip())
            if txt:
                out.append(Paragraph(inline_md(txt), body_style))
                out.append(Spacer(1, 2.2*mm))
            para = []

    for raw in lines:
        line = raw.strip()
        if not line:
            flush()
            continue
        m = re.match(r"^(#{1,3})\s+(.+)$", line)
        if m:
            flush()
            out.append(Paragraph(inline_md(m.group(2)), heading_style))
            continue
        m = re.match(r"^(\d+)\.\s+(.+)$", line)
        if m:
            flush()
            out.append(Paragraph(f"<b>{m.group(1)}.</b> {inline_md(m.group(2))}", body_style))
            out.append(Spacer(1, 1.5*mm))
            continue
        m = re.match(r"^[-•]\s+(.+)$", line)
        if m:
            flush()
            out.append(Paragraph(f"• {inline_md(m.group(1))}", body_style))
            out.append(Spacer(1, 1*mm))
            continue
        # Handle AI output where numbered sections accidentally arrive on one line.
        parts = re.split(r"(?=\s+\d+\.\s+[A-Z])", line)
        if len(parts) > 1:
            flush()
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                m2 = re.match(r"^(\d+)\.\s+(.+)$", part)
                if m2:
                    out.append(Paragraph(f"<b>{m2.group(1)}.</b> {inline_md(m2.group(2))}", body_style))
                    out.append(Spacer(1, 1.5*mm))
                else:
                    out.append(Paragraph(inline_md(part), body_style))
                    out.append(Spacer(1, 2*mm))
        else:
            para.append(line)
    flush()
    return out or [Paragraph("No scientific interpretation available.", body_style)]

@app.post("/api/generate-pdf")
def create_pdf(req: PDFRequest):
    safe_target = re_safe(req.target)
    safe_name = re_safe(req.name)
    pdf_path = REPORT_DIR / f"KBioX AI_{safe_target}_{safe_name}_Complete_Dossier.pdf"

    mol = Chem.MolFromSmiles(req.smiles)
    if mol is None:
        raise HTTPException(400, "Cannot create PDF: invalid SMILES.")

    struct_path = TEMP_DIR / f"structure_{safe_target}_{safe_name}.png"
    try:
        Draw.MolToImage(mol, size=(1000, 650), kekulize=True).save(struct_path)
    except Exception as e:
        raise HTTPException(500, f"Could not render structure image: {e}")

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        rightMargin=14 * mm, leftMargin=14 * mm,
        topMargin=14 * mm, bottomMargin=14 * mm
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "KBTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=18, leading=22, textColor=colors.HexColor("#0f172a"),
        alignment=TA_CENTER, spaceAfter=5
    )
    subtitle_style = ParagraphStyle(
        "KBSub", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=8.5, leading=12, textColor=colors.HexColor("#64748b"),
        alignment=TA_CENTER, spaceAfter=8
    )
    heading_style = ParagraphStyle(
        "KBHeading", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=11.5, leading=14, textColor=colors.HexColor("#1d4ed8"),
        spaceBefore=8, spaceAfter=4
    )
    body_style = ParagraphStyle(
        "KBBody", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=8.5, leading=12, textColor=colors.HexColor("#334155")
    )
    small_style = ParagraphStyle(
        "KBSmall", parent=body_style, fontSize=7.5, leading=10,
        textColor=colors.HexColor("#64748b")
    )
    mono_style = ParagraphStyle(
        "KBMono", parent=body_style, fontName="Courier",
        fontSize=7.2, leading=10, textColor=colors.HexColor("#334155")
    )

    p = req.properties or {}
    pred = req.prediction or {}
    geometry = pred.get("geometry", {}) or {}
    attribution = pred.get("shapFeatures", []) or []
    source = pred.get("predictionSource", "N/A")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    def safe(v):
        return str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def table(data, widths):
        return Table(data, colWidths=widths, repeatRows=0, style=[
            ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#cbd5e1")),
            ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#f8fafc")),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("FONTSIZE", (0,0), (-1,-1), 7.8),
            ("LEFTPADDING", (0,0), (-1,-1), 5),
            ("RIGHTPADDING", (0,0), (-1,-1), 5),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ])

    story = [
        Paragraph("KBioX AI — Complete Computational Screening Dossier", title_style),
        Paragraph(
            f"Explainable Molecular Bioactivity Platform • Generated {safe(timestamp)}",
            subtitle_style
        ),
        table([
            ["Compound", safe(req.name)],
            ["Target", safe(req.target)],
            ["Canonical SMILES", Paragraph(safe(req.smiles), mono_style)],
            ["Prediction source", safe(source)],
        ], [42*mm, 132*mm]),
        Spacer(1, 4*mm),

        Paragraph("1. Molecular Structure & Identity", heading_style),
        Paragraph(
            "The structure below is generated from the submitted SMILES using RDKit. "
            "The 3D view used by the web interface is an RDKit conformer; the PDF includes "
            "the 2D chemical structure for reproducibility and review.", body_style
        ),
        Spacer(1, 2*mm),
        PDFImage(str(struct_path), width=115*mm, height=75*mm),
        Spacer(1, 2*mm),
        table([
            ["Atom count", safe(len(mol.GetAtoms())), "Bond count", safe(len(mol.GetBonds()))],
            ["Canonical SMILES", Paragraph(safe(req.smiles), mono_style), "", ""],
        ], [32*mm, 52*mm, 32*mm, 58*mm]),

        Paragraph("2. Bioactivity Prediction", heading_style),
        table([
            ["Target", safe(req.target), "Classification", "PREDICTED ACTIVE" if pred.get("isActive") else "INACTIVE / LOW"],
            ["Confidence", f"{safe(pred.get('confidence','N/A'))}%", "Estimated pIC50*", safe(pred.get("pic50","N/A"))],
            ["Model features", safe(pred.get("modelFeatures","N/A")), "Prediction source", safe(source)],
        ], [32*mm, 45*mm, 42*mm, 65*mm]),
        Paragraph(
            "* The displayed pIC50 is an estimated UI value unless the supplied estimator was explicitly trained as a pIC50 regression model. It is not a measured experimental concentration.",
            small_style
        ),

        Paragraph("3. Physicochemical Properties", heading_style),
        table([
            ["Molecular weight", safe(p.get("mw","N/A")), "LogP", safe(p.get("logp","N/A"))],
            ["TPSA", safe(p.get("tpsa","N/A")) + " Å²", "QED", safe(p.get("qed","N/A"))],
            ["H-bond donors", safe(p.get("hbd","N/A")), "H-bond acceptors", safe(p.get("hba","N/A"))],
            ["Rotatable bonds", safe(p.get("rotb","N/A")), "Heavy atoms", safe(p.get("heavyAtoms","N/A"))],
            ["Lipinski violations", safe(p.get("violations","N/A")), "Lipinski status", "PASS" if p.get("violations", 99) == 0 else "ALERT"],
        ], [43*mm, 44*mm, 43*mm, 54*mm]),

        Paragraph("4. Explainability / Attribution Results", heading_style),
        Paragraph(
            "The web application reports these as descriptor-level explanatory contributions. "
            "They should not be called true SHAP values unless a SHAP explainer was run against the loaded estimator.",
            body_style
        ),
        Spacer(1, 2*mm),
        table(
            [["Feature / fragment", "Contribution"]] + [
                [safe(x.get("fragment", "Feature")), safe(x.get("value", x.get("shap", "N/A")))]
                for x in attribution
            ],
            [130*mm, 54*mm]
        ),

        Paragraph("5. AI Scientific Key Points", heading_style),
        ListFlowable([ListItem(Paragraph(f"<b>{safe(str(x))}</b>", body_style), leftIndent=12) for x in (req.ai_keypoints or pred.get("aiKeyPoints", []))], bulletType="bullet", leftIndent=14) if (req.ai_keypoints or pred.get("aiKeyPoints", [])) else Paragraph("AI key points unavailable.", body_style),

        Paragraph("6. Scientific Interpretation", heading_style),
        *markdown_to_pdf_flowables(
            req.ai_interpretation or pred.get("aiInterpretation", req.ai_summary or pred.get("aiSummary", "No summary available.")),
            body_style, heading_style, small_style
        ),

        Paragraph("7. Model / Runtime Information", heading_style),
        table([
            ["Platform version", "KBioX AI v2.8 PRO"],
            ["Chemistry engine", "RDKit"],
            ["Target selected", safe(req.target)],
            ["Loaded estimator", safe(source)],
            ["Geometry atoms returned", safe(len(geometry.get("atoms", [])))],
            ["Geometry bonds returned", safe(len(geometry.get("bonds", [])))],
        ], [55*mm, 119*mm]),

        Paragraph("8. Full Computational Result Summary", heading_style),
        ListFlowable([
            ListItem(Paragraph(safe(f"{req.name} was classified as {'PREDICTED ACTIVE' if pred.get('isActive') else 'INACTIVE / LOW PREDICTED ACTIVITY'} against {req.target} with {pred.get('confidence','N/A')}% confidence using a {source} prediction source."), body_style), leftIndent=12),
            ListItem(Paragraph(safe(f"The calculated profile is MW {p.get('mw','N/A')} g/mol, LogP {p.get('logp','N/A')}, TPSA {p.get('tpsa','N/A')} Å², HBD/HBA {p.get('hbd','N/A')}/{p.get('hba','N/A')}, QED {p.get('qed','N/A')}, with {p.get('violations','N/A')} Lipinski violation(s)."), body_style), leftIndent=12),
            ListItem(Paragraph(safe(f"The estimated pIC50 of {pred.get('pic50','N/A')} is computational and must not be treated as a measured experimental potency value."), body_style), leftIndent=12),
            ListItem(Paragraph(safe("Overall, the result is a prioritization signal for further target-specific computational analysis and experimental validation, not proof of biological efficacy."), body_style), leftIndent=12),
        ], bulletType="bullet", leftIndent=14),

        Paragraph("9. Scientific & Regulatory Disclaimer", heading_style),
        Paragraph(
            "KBioX AI is intended for academic hypothesis generation, computational lead screening and educational research. "
            "In silico predictions do not establish experimental efficacy, binding affinity, pharmacokinetics, toxicity, safety or clinical utility. "
            "Model outputs may be influenced by training-data coverage, feature representation and model uncertainty. "
            "Wet-lab validation and appropriate experimental controls are required before biological or translational conclusions are made.",
            body_style
        ),
    ]

    doc.build(story)
    return FileResponse(str(pdf_path), media_type="application/pdf", filename=pdf_path.name)

def re_safe(s):
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))[:80] or "Report"

# ================================================================
# 7. FRONTEND
# ================================================================
HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
<title>KBioX AI | Explainable Molecular Bioactivity</title>
<script src="https://cdn.tailwindcss.com"></script>
<script>
tailwind.config={darkMode:'class',theme:{extend:{fontFamily:{sans:['Plus Jakarta Sans','Inter','system-ui','sans-serif'],mono:['JetBrains Mono','monospace']},colors:{brand:{500:'#3b82f6',600:'#2563eb'},bio:{400:'#34d399',500:'#10b981'}}}}}
</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.6.0/css/all.min.css">
<style>
*{box-sizing:border-box}html{scroll-behavior:smooth;touch-action:manipulation;-webkit-text-size-adjust:100%}body{background:#070b13;color:#e2e8f0;overflow-x:hidden;touch-action:manipulation}button,a{touch-action:manipulation}#molCanvas,#svg2d,.viewer-toggle{touch-action:none;-webkit-user-select:none;user-select:none}#svg2d{overflow:hidden}#svg2d svg{width:100% !important;height:100% !important;max-width:100%;max-height:100%;display:block;object-fit:contain}
body:before{content:"";position:fixed;inset:0;pointer-events:none;background:radial-gradient(circle at 10% 5%,rgba(37,99,235,.12),transparent 32%),radial-gradient(circle at 90% 18%,rgba(16,185,129,.07),transparent 30%);z-index:-2}
.hero-grid{grid-template-columns:minmax(0,1.45fr) minmax(320px,.75fr)}
@media(min-width:768px) and (max-width:1100px){.hero-grid{grid-template-columns:minmax(0,1.15fr) minmax(300px,.85fr)!important}}
.grid-bg{position:fixed;inset:0;z-index:-1;opacity:.16;background-image:linear-gradient(rgba(148,163,184,.07) 1px,transparent 1px),linear-gradient(90deg,rgba(148,163,184,.07) 1px,transparent 1px);background-size:42px 42px;mask-image:linear-gradient(to bottom,#000,transparent 80%)}
.glass{background:rgba(15,23,42,.74);backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);border:1px solid rgba(255,255,255,.075);box-shadow:0 18px 55px rgba(0,0,0,.18)}
.glass-hover{transition:.25s cubic-bezier(.16,1,.3,1)}.glass-hover:hover{transform:translateY(-3px);border-color:rgba(59,130,246,.35);box-shadow:0 18px 45px rgba(37,99,235,.12)}
.iconbox{width:38px;height:38px;border-radius:12px;display:grid;place-items:center;background:rgba(59,130,246,.09);border:1px solid rgba(59,130,246,.2);color:#60a5fa;flex:none}
.pill{border:1px solid rgba(255,255,255,.08);background:rgba(15,23,42,.7);border-radius:999px;padding:5px 10px;font-size:10px;font-weight:700}
.navbtn{transition:.2s;border:1px solid transparent}.navbtn:hover{background:rgba(255,255,255,.05);color:#fff}.navbtn.active{background:#2563eb;color:#fff;box-shadow:0 7px 22px rgba(37,99,235,.25)}
.target{transition:.22s cubic-bezier(.16,1,.3,1)}.target:hover{transform:translateY(-2px)}.target.active{border-color:rgba(59,130,246,.8);background:linear-gradient(135deg,rgba(37,99,235,.18),rgba(15,23,42,.82));box-shadow:0 0 0 1px rgba(59,130,246,.18),0 16px 38px rgba(37,99,235,.11)}
.btn-main{position:relative;overflow:hidden}.btn-main:after{content:"";position:absolute;top:0;bottom:0;width:45%;left:-55%;background:linear-gradient(90deg,transparent,rgba(255,255,255,.18),transparent);transform:skewX(-18deg);animation:shine 3.2s infinite}@keyframes shine{0%,60%{left:-55%}100%{left:130%}}
.scan{position:absolute;left:0;right:0;top:0;height:2px;background:linear-gradient(90deg,transparent,#38bdf8,#34d399,transparent);box-shadow:0 0 16px #38bdf8;animation:scan 2.8s ease-in-out infinite;z-index:2}@keyframes scan{0%{transform:translateY(0);opacity:0}18%{opacity:1}80%{opacity:1}100%{transform:translateY(330px);opacity:0}}
.pulse{animation:pulse 2s infinite}@keyframes pulse{0%,100%{transform:scale(1);opacity:1}50%{transform:scale(1.45);opacity:.45}}
.float{animation:float 5s ease-in-out infinite}@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}
.reveal{animation:reveal .45s ease both}@keyframes reveal{from{opacity:0;transform:translateY(9px)}to{opacity:1;transform:none}}
.progress{background:linear-gradient(90deg,#3b82f6,#34d399);transition:width .5s ease}
.metric{background:rgba(2,6,23,.46);border:1px solid rgba(255,255,255,.055);transition:.2s}.metric:hover{border-color:rgba(59,130,246,.25);transform:translateY(-1px)}
.tab{transition:.2s}.tab.active{background:#2563eb;color:#fff;box-shadow:0 8px 20px rgba(37,99,235,.2)}
.toast{transition:.3s cubic-bezier(.16,1,.3,1)}
.ai-text h4{font-weight:800;color:#fff;margin:14px 0 7px}.ai-text p{margin:7px 0}.ai-text ul{list-style:disc;padding-left:20px;margin:7px 0}.ai-text ol{list-style:decimal;padding-left:22px;margin:7px 0}.ai-text li{margin:4px 0}.ai-text strong{font-weight:800;color:#fff}.ai-text em{font-style:italic;color:#cbd5e1}.ai-text code{font-family:"JetBrains Mono",monospace;color:#7dd3fc}.ai-summary-points{list-style:none;padding:0;margin:0;display:grid;gap:10px}.ai-summary-points li{position:relative;padding-left:18px}.ai-summary-points li:before{content:"";position:absolute;left:0;top:.55em;width:6px;height:6px;border-radius:50%;background:#8b5cf6;box-shadow:0 0 10px rgba(139,92,246,.5)}
@media(max-width:767px){html,body{touch-action:pan-y;overscroll-behavior-x:none}.mobile-scroll{overflow-x:auto;scrollbar-width:none}.mobile-scroll::-webkit-scrollbar{display:none}.hero-title{font-size:2rem;line-height:1.05}.section-pad{padding:1rem}.hide-mobile{display:none}#svg2d{touch-action:none;user-select:none;-webkit-user-select:none;-webkit-touch-callout:none}#svg2d svg{display:block;width:100%!important;height:100%!important;max-width:100%!important;max-height:100%!important;touch-action:none}}
/* Mobile browser Desktop-Site compatibility. */
@media (hover:none) and (pointer:coarse) and (max-width:1100px){.lg\:grid-cols-\[1fr_auto\]{grid-template-columns:minmax(0,1fr)!important}.lg\:col-span-5,.lg\:col-span-7{grid-column:span 12 / span 12!important}.lg\:flex{display:none!important}.lg\:grid-cols-12{grid-template-columns:repeat(1,minmax(0,1fr))!important}main{width:100%!important;max-width:100%!important}.hero-title{font-size:2.25rem!important;line-height:1.05!important}}
</style>
<style id="kbio-font-boost">
/* KBioX AI readability boost: preserve the existing design while making all UI text easier to read. */
body{font-size:17px!important}
body .text-\[9px\]{font-size:11px!important}
body .text-\[10px\]{font-size:12px!important}
body .text-\[11px\]{font-size:13px!important}
body .text-\[12px\]{font-size:14px!important}
body .text-xs{font-size:14px!important}
body .text-sm{font-size:16px!important}
body .text-base{font-size:17px!important}
body .text-lg{font-size:20px!important}
body .text-xl{font-size:22px!important}
body .text-2xl{font-size:26px!important}
body .text-3xl{font-size:32px!important}
body .text-4xl{font-size:38px!important}
.ai-text{font-size:16px!important;line-height:1.75!important}
.ai-text p,.ai-text li{font-size:16px!important;line-height:1.75!important}
.ai-summary-points{font-size:15px!important;line-height:1.7!important}
.ai-summary-points li{font-size:15px!important;line-height:1.7!important}
#aiKeypoints,#aiKeypoints *{font-size:15px!important;line-height:1.7!important}
#summary{font-size:15px!important;line-height:1.75!important}
#confidence{font-size:14px!important;line-height:1.5!important}
.metric>div:first-child{font-size:11px!important}
.metric>div:last-child{font-size:18px!important}
@media(max-width:767px){
  body{font-size:17px!important}
  .hero-title{font-size:2.25rem!important}
  #summary,.ai-summary-points,.ai-summary-points li,#aiKeypoints,#aiKeypoints *{font-size:15px!important}
  .ai-text,.ai-text p,.ai-text li{font-size:16px!important}
}
</style>
</head>
<body class="min-h-screen font-sans antialiased">
<div class="grid-bg"></div>

<header class="sticky top-0 z-50 glass border-b border-white/10 px-3 sm:px-5 lg:px-8 py-3">
 <div class="max-w-7xl mx-auto flex items-center justify-between gap-3">
  <a href="#top" class="flex items-center gap-3 min-w-0">
   <div class="relative w-10 h-10 rounded-xl bg-slate-950 border border-blue-500/25 shadow-lg shadow-blue-500/10 grid place-items-center">
    <i class="fa-solid fa-dna text-lg bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent"></i>
    <span class="absolute -bottom-1 -right-1 w-3 h-3 rounded-full bg-emerald-400 border-2 border-[#070b13] pulse"></span>
   </div>
   <div class="min-w-0">
    <div class="flex items-center gap-2"><span class="text-lg sm:text-xl font-extrabold tracking-tight">KBioX <span class="text-blue-500">AI</span></span><span class="pill text-blue-400 hidden sm:inline">v2.8 PRO</span></div>
    <p class="text-[10px] text-slate-500 truncate">Where Biology Meets Intelligence.</p>
   </div>
  </a>
  <nav class="hidden lg:flex items-center gap-1 bg-slate-950/65 p-1.5 rounded-xl border border-white/5 text-[11px] font-semibold">
   <a class="navbtn active px-3 py-2 rounded-lg" href="#studio"><i class="fa-solid fa-flask mr-1.5"></i>Studio</a>
   <a class="navbtn px-3 py-2 rounded-lg text-slate-400" href="#analytics"><i class="fa-solid fa-chart-column mr-1.5"></i>Analytics</a>
   <a class="navbtn px-3 py-2 rounded-lg text-slate-400" href="#copilot"><i class="fa-solid fa-brain mr-1.5 text-purple-400"></i>Copilot</a>
  </nav>
  <div class="flex items-center gap-2"><span id="serverPill" class="pill text-emerald-400"><span class="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5 pulse"></span>ONLINE</span><button onclick="downloadPdfReport()" class="p-2.5 rounded-xl bg-slate-900 border border-white/10 hover:border-blue-500/40 transition" title="Export full PDF"><i class="fa-solid fa-file-pdf text-red-400"></i></button></div>
 </div>
</header>

<main id="top" class="max-w-7xl mx-auto px-3 sm:px-5 lg:px-8 py-5 sm:py-7 space-y-5">

<section class="glass rounded-3xl overflow-hidden relative">
 <div class="absolute -top-24 -right-20 w-72 h-72 rounded-full bg-blue-600/10 blur-3xl"></div>
 <div class="hero-grid relative p-5 sm:p-7 lg:p-8 grid gap-6 items-center">
  <div>
   <div class="flex flex-wrap gap-2 mb-4"><span class="pill text-blue-400"><i class="fa-solid fa-microchip mr-1"></i>ML SCREENING</span><span class="pill text-emerald-400"><i class="fa-solid fa-atom mr-1"></i>RDKit</span><span class="pill text-purple-400"><i class="fa-solid fa-wand-magic-sparkles mr-1"></i>Explainable</span></div>
   <h1 class="hero-title text-3xl sm:text-5xl font-extrabold tracking-tight leading-tight">AI-Powered Molecular<br><span class="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-cyan-300 to-emerald-400">Bioactivity Intelligence</span></h1>
   <p class="text-xs sm:text-sm text-purple-300 font-semibold mt-2">Where Biology Meets Intelligence.</p>
   <p class="text-sm sm:text-base text-slate-400 max-w-2xl mt-4 leading-relaxed">A polished computational workspace for SMILES validation, disease-target screening, molecular visualization, descriptor analysis, explainability and full scientific reporting.</p>
   <div class="flex flex-wrap gap-3 mt-6 text-[11px] text-slate-500"><span><i class="fa-solid fa-circle-check text-emerald-400 mr-1"></i>Interactive</span><span><i class="fa-solid fa-mobile-screen text-blue-400 mr-1"></i>Mobile ready</span><span><i class="fa-solid fa-file-pdf text-red-400 mr-1"></i>Full dossier</span></div>
  </div>
  <div class="hidden sm:grid grid-cols-2 gap-3 w-full">
   <div class="metric rounded-2xl p-4"><i class="fa-solid fa-vial text-blue-400"></i><div class="text-2xl font-extrabold mt-2">03</div><div class="text-[10px] text-slate-500">Target models</div></div>
   <div class="metric rounded-2xl p-4"><i class="fa-solid fa-chart-simple text-emerald-400"></i><div class="text-2xl font-extrabold mt-2">04</div><div class="text-[10px] text-slate-500">Analytics views</div></div>
   <div class="metric rounded-2xl p-4"><i class="fa-solid fa-code-branch text-purple-400"></i><div class="text-2xl font-extrabold mt-2">RDKit</div><div class="text-[10px] text-slate-500">Chemistry engine</div></div>
   <div class="metric rounded-2xl p-4"><i class="fa-solid fa-file-lines text-amber-400"></i><div class="text-2xl font-extrabold mt-2">PDF</div><div class="text-[10px] text-slate-500">Full results</div></div>
  </div>
 </div>
</section>

<section id="studio" class="space-y-3">
 <div class="flex items-end justify-between gap-3"><div><p class="text-[10px] uppercase tracking-[.18em] text-blue-400 font-bold">01 / Target models</p><h2 class="text-lg sm:text-xl font-bold mt-1">Choose a screening target</h2></div><span id="modelStatus" class="text-[10px] text-slate-500 font-mono">Checking model status...</span></div>
 <div class="grid md:grid-cols-3 gap-3">
  <button id="target-HIV" onclick="selectTarget('HIV')" class="target active glass glass-hover rounded-2xl p-4 text-left"><div class="flex items-center gap-3"><div class="iconbox"><i class="fa-solid fa-virus"></i></div><div><div class="font-bold">HIV-1</div><div class="text-[10px] text-slate-500">Antiviral bioactivity</div></div><i class="fa-solid fa-chevron-right ml-auto text-[10px] text-slate-600"></i></div></button>
  <button id="target-TB" onclick="selectTarget('TB')" class="target glass glass-hover rounded-2xl p-4 text-left"><div class="flex items-center gap-3"><div class="iconbox text-amber-400 bg-amber-500/10 border-amber-500/20"><i class="fa-solid fa-bacterium"></i></div><div><div class="font-bold">Tuberculosis</div><div class="text-[10px] text-slate-500">M. tuberculosis screening</div></div><i class="fa-solid fa-chevron-right ml-auto text-[10px] text-slate-600"></i></div></button>
  <button id="target-Malaria" onclick="selectTarget('Malaria')" class="target glass glass-hover rounded-2xl p-4 text-left"><div class="flex items-center gap-3"><div class="iconbox text-rose-400 bg-rose-500/10 border-rose-500/20"><i class="fa-solid fa-mosquito"></i></div><div><div class="font-bold">Malaria</div><div class="text-[10px] text-slate-500">P. falciparum screening</div></div><i class="fa-solid fa-chevron-right ml-auto text-[10px] text-slate-600"></i></div></button>
 </div>
</section>

<section class="glass rounded-3xl p-4 sm:p-6" id="input-card">
 <div class="flex items-start justify-between gap-4"><div><p class="text-[10px] uppercase tracking-[.18em] text-blue-400 font-bold">02 / Molecular input</p><h2 class="text-lg font-bold mt-1">Molecular Studio</h2><p class="text-[11px] text-slate-500 mt-1">Paste SMILES, use a preset, or clean the structure before screening.</p></div><span id="validity" class="pill text-slate-500">READY</span></div>
 <div class="flex flex-wrap gap-2 mt-5">
  <button onclick="setSMILES('CC(=O)Oc1ccccc1C(=O)O','HIV')" class="pill hover:border-blue-500/40 hover:text-white transition"><i class="fa-solid fa-capsules mr-1"></i>Aspirin</button>
  <button onclick="setSMILES('CC(OCP(=O)(O)O)Cn1cnc2c(N)ncnc12','HIV')" class="pill hover:border-blue-500/40 hover:text-white transition"><i class="fa-solid fa-dna mr-1"></i>Tenofovir</button>
  <button onclick="setSMILES('Nc1ccn([C@H]2CS[C@H](CO)O2)c(=O)n1','HIV')" class="pill hover:border-blue-500/40 hover:text-white transition"><i class="fa-solid fa-flask-vial mr-1"></i>Lamivudine</button>
  <button onclick="setSMILES('NC(=O)c1ccncc1','TB')" class="pill hover:border-amber-500/40 hover:text-white transition"><i class="fa-solid fa-vial mr-1"></i>Isoniazid</button>
  <button onclick="setSMILES('CC1CCC2C(C)C(=O)OC3OC4(C)CCC1C23OO4','Malaria')" class="pill hover:border-rose-500/40 hover:text-white transition"><i class="fa-solid fa-leaf mr-1"></i>Artemisinin</button>
  <button onclick="setSMILES('CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12','Malaria')" class="pill hover:border-rose-500/40 hover:text-white transition"><i class="fa-solid fa-pills mr-1"></i>Chloroquine</button>
 </div>
 <div class="relative mt-4">
  <input id="smilesInput" value="CC(=O)Oc1ccccc1C(=O)O" spellcheck="false" class="w-full bg-slate-950/80 border border-slate-700 focus:border-blue-500 focus:ring-4 focus:ring-blue-500/10 rounded-2xl px-4 py-4 pr-32 font-mono text-sm text-blue-300 outline-none transition" placeholder="Enter SMILES...">
  <div class="absolute right-2 top-2 flex gap-1"><button onclick="canonicalize()" class="p-2 rounded-lg text-slate-500 hover:text-white hover:bg-slate-800" title="Validate / canonicalize"><i class="fa-solid fa-wand-magic-sparkles"></i></button><button onclick="copySMILES()" class="p-2 rounded-lg text-slate-500 hover:text-white hover:bg-slate-800" title="Copy"><i class="fa-regular fa-copy"></i></button><button onclick="clearSMILES()" class="p-2 rounded-lg text-slate-500 hover:text-red-400 hover:bg-slate-800" title="Clear"><i class="fa-solid fa-xmark"></i></button></div>
 </div>
 <div id="errorBox" class="hidden mt-3 rounded-xl p-3 text-xs text-red-300 bg-red-950/40 border border-red-500/20"><i class="fa-solid fa-circle-exclamation mr-2"></i><span></span></div>
 <button id="runBtn" onclick="runScreening()" class="btn-main mt-4 w-full py-4 rounded-2xl bg-gradient-to-r from-orange-500 via-amber-500 to-orange-600 hover:from-orange-600 hover:to-amber-600 font-extrabold text-sm shadow-xl shadow-orange-500/10 active:scale-[.995] transition"><span class="relative z-10"><i class="fa-solid fa-bolt-lightning mr-2"></i>RUN SCREENING & EXPLAINABILITY</span></button>
 <div id="pipeline" class="hidden mt-4 space-y-2"><div class="flex justify-between text-[10px] text-slate-500 font-mono"><span id="pipelineText"><i class="fa-solid fa-circle-notch fa-spin mr-1 text-blue-400"></i>Preparing...</span><span id="pipelinePct">Screening progress is 0 percent complete.</span></div><div class="h-2 bg-slate-950 rounded-full overflow-hidden"><div id="pipelineBar" class="progress h-full w-0"></div></div></div>
</section>

<section class="grid lg:grid-cols-12 gap-5">
 <div class="lg:col-span-5 glass rounded-3xl p-5 sm:p-6 reveal">
  <div class="flex items-center justify-between"><div class="flex items-center gap-2"><div class="iconbox"><i class="fa-solid fa-chart-pie"></i></div><div><p class="text-[10px] text-slate-500 uppercase tracking-widest">Prediction</p><h3 class="font-bold">Bioactivity result</h3></div></div><span id="targetBadge" class="pill text-blue-400">HIV</span></div>
  <div class="mt-6"><div class="text-[10px] uppercase tracking-widest text-slate-500">Compound</div><div id="compoundName" class="text-xl sm:text-2xl font-extrabold text-sky-400 mt-1 truncate">—</div></div>
  <div class="mt-5"><span id="verdictPill" class="pill text-slate-500">WAITING</span><h2 id="verdict" class="text-3xl font-extrabold mt-3">Run screening</h2><p id="summary" class="text-xs text-slate-400 leading-relaxed mt-2">Your full result will appear here.</p></div>
  <div id="aiKeypointsCard" class="hidden mt-5 rounded-2xl p-4 bg-purple-500/5 border border-purple-500/15"><div class="flex items-center gap-2 mb-3"><i class="fa-solid fa-sparkles text-purple-400"></i><h3 class="font-bold text-sm">AI Key Points</h3><span class="pill text-purple-400 ml-auto">AI</span></div><div id="aiKeypoints" class="space-y-2 text-[11px] text-slate-300 leading-relaxed"></div></div>
  <div class="mt-6"><div class="flex justify-between text-[10px] text-slate-500"><span>Confidence</span><b id="confidence" class="text-slate-200">Awaiting screening result.</b></div><div class="h-2.5 bg-slate-950 rounded-full mt-2 overflow-hidden"><div id="confidenceBar" class="progress h-full w-0"></div></div></div>
  <div class="grid grid-cols-3 gap-2 mt-6"><div class="metric rounded-xl p-3"><div class="text-[9px] text-slate-500">MW</div><div id="mw" class="font-mono font-bold mt-1">—</div></div><div class="metric rounded-xl p-3"><div class="text-[9px] text-slate-500">LogP</div><div id="logp" class="font-mono font-bold mt-1">—</div></div><div class="metric rounded-xl p-3"><div class="text-[9px] text-slate-500">pIC50*</div><div id="pic50" class="font-mono font-bold text-emerald-400 mt-1">—</div></div></div>
  <p class="text-[9px] text-slate-600 mt-3">*Estimated UI value unless the supplied model was explicitly trained for pIC50 regression.</p>
 </div>
 <div class="lg:col-span-7 glass rounded-3xl p-4 sm:p-5 reveal">
  <div class="flex items-center justify-between mb-3"><div class="flex gap-1 bg-slate-950/80 p-1 rounded-xl border border-white/5"><button id="v3" onclick="switchView('3d')" class="tab active px-3 py-1.5 rounded-lg text-[10px] font-bold">3D VIEW</button><button id="v2" onclick="switchView('2d')" class="tab px-3 py-1.5 rounded-lg text-[10px]">2D VIEW</button></div><span class="text-[9px] text-slate-600"><i class="fa-solid fa-hand-pointer mr-1"></i>Drag to rotate</span></div>
  <div class="relative h-[290px] sm:h-[360px] rounded-2xl bg-[#04070d] border border-white/5 overflow-hidden"><div class="scan"></div><canvas id="molCanvas" class="w-full h-full"></canvas><div id="svg2d" class="hidden w-full h-full items-center justify-center p-4"></div><div class="absolute bottom-3 left-3 flex gap-1.5"><span class="pill">C</span><span class="pill text-red-400">O</span><span class="pill text-indigo-400">N</span></div></div>
  <div class="flex justify-between mt-2 text-[9px] text-slate-600"><span>RDKit conformer visualization</span><span id="atomBondCount">The molecular model is awaiting analysis.</span></div>
 </div>
</section>

<section id="analytics" class="glass rounded-3xl p-4 sm:p-6">
 <div class="flex items-start justify-between gap-3"><div><p class="text-[10px] uppercase tracking-[.18em] text-emerald-400 font-bold">03 / Analytics</p><h2 class="text-lg font-bold mt-1">Interpretation workspace</h2></div><button onclick="downloadPdfReport()" class="pill text-blue-400 hover:text-white hover:border-blue-500/40"><i class="fa-solid fa-download mr-1"></i>Full PDF</button></div>
 <div class="mobile-scroll flex gap-1 bg-slate-950/50 rounded-xl p-1.5 mt-5 border border-white/5">
  <button class="tab active whitespace-nowrap px-3 py-2 rounded-lg text-[10px] font-bold" onclick="openTab('properties',this)"><i class="fa-solid fa-flask mr-1"></i>Properties</button>
  <button class="tab whitespace-nowrap px-3 py-2 rounded-lg text-[10px]" onclick="openTab('shap',this)"><i class="fa-solid fa-chart-column mr-1"></i>Attribution</button>
  <button class="tab whitespace-nowrap px-3 py-2 rounded-lg text-[10px]" onclick="openTab('radar',this)"><i class="fa-solid fa-bullseye mr-1"></i>Radar</button>
  <button class="tab whitespace-nowrap px-3 py-2 rounded-lg text-[10px]" onclick="openTab('dossier',this)"><i class="fa-solid fa-file-invoice mr-1"></i>Dossier</button>
 </div>
 <div id="properties" class="tabContent mt-5 reveal"><div class="flex items-center justify-between"><h3 class="font-bold text-sm">Physicochemical profile</h3><span id="lipinski" class="pill text-slate-500">—</span></div><div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2 mt-4"><div class="metric rounded-xl p-3"><i class="fa-solid fa-weight-hanging text-blue-400 text-xs"></i><div class="text-[9px] text-slate-500 mt-2">Molecular weight</div><strong id="pMW" class="block mt-1">—</strong></div><div class="metric rounded-xl p-3"><i class="fa-solid fa-droplet text-cyan-400 text-xs"></i><div class="text-[9px] text-slate-500 mt-2">LogP</div><strong id="pLogP" class="block mt-1">—</strong></div><div class="metric rounded-xl p-3"><i class="fa-solid fa-hand-holding-droplet text-purple-400 text-xs"></i><div class="text-[9px] text-slate-500 mt-2">HBD</div><strong id="pHBD" class="block mt-1">—</strong></div><div class="metric rounded-xl p-3"><i class="fa-solid fa-magnet text-emerald-400 text-xs"></i><div class="text-[9px] text-slate-500 mt-2">HBA</div><strong id="pHBA" class="block mt-1">—</strong></div><div class="metric rounded-xl p-3"><i class="fa-solid fa-braille text-amber-400 text-xs"></i><div class="text-[9px] text-slate-500 mt-2">TPSA</div><strong id="pTPSA" class="block mt-1">—</strong></div><div class="metric rounded-xl p-3"><i class="fa-solid fa-link text-rose-400 text-xs"></i><div class="text-[9px] text-slate-500 mt-2">Rotatable bonds</div><strong id="pRotB" class="block mt-1">—</strong></div></div><div class="grid sm:grid-cols-3 gap-2 mt-3"><div class="metric rounded-xl p-3"><span class="text-[9px] text-slate-500">QED</span><strong id="pQED" class="block text-emerald-400 mt-1">—</strong></div><div class="metric rounded-xl p-3"><span class="text-[9px] text-slate-500">Heavy atoms</span><strong id="pHeavy" class="block mt-1">—</strong></div><div class="metric rounded-xl p-3"><span class="text-[9px] text-slate-500">Lipinski violations</span><strong id="pViol" class="block mt-1">—</strong></div></div></div>
 <div id="shap" class="tabContent hidden mt-5 reveal"><div class="flex justify-between gap-3"><div><h3 class="font-bold text-sm">Explainability / attribution</h3><p class="text-[10px] text-slate-500 mt-1">Descriptor-level contributions. True SHAP is only claimed when SHAP is run against the loaded estimator.</p></div></div><div id="shapBox" class="space-y-2 mt-5"></div></div>
 <div id="radar" class="tabContent hidden mt-5 reveal"><div class="flex flex-col md:flex-row items-center justify-center gap-8"><div class="relative"><canvas id="radarCanvas" width="340" height="340"></canvas></div><div class="max-w-md space-y-3 text-[11px] text-slate-400"><div><i class="fa-solid fa-circle text-blue-400 mr-2"></i><b class="text-white">Current molecule:</b> normalized descriptors</div><div><i class="fa-solid fa-circle text-emerald-400/40 mr-2"></i><b class="text-white">Reference space:</b> qualitative drug-like envelope</div><p class="leading-relaxed">The radar is a visual descriptor profile and should not be interpreted as a validated pharmacokinetic prediction.</p></div></div></div>
 <div id="dossier" class="tabContent hidden mt-5 reveal"><div class="flex items-center justify-between gap-3"><div><h3 class="font-bold text-sm"><i class="fa-solid fa-file-invoice text-emerald-400 mr-2"></i>Comprehensive Screening Dossier</h3><p class="text-[10px] text-slate-500 mt-1">The PDF export contains the same available results plus structure and interpretation notes.</p></div><div class="flex gap-2"><button onclick="copyDossier()" class="pill"><i class="fa-regular fa-copy mr-1"></i>Copy</button><button onclick="downloadPdfReport()" class="pill text-blue-400"><i class="fa-solid fa-file-pdf mr-1"></i>PDF</button></div></div><div id="dossierText" class="mt-4 bg-slate-950/75 rounded-2xl p-4 sm:p-5 border border-white/5 font-mono text-[10px] sm:text-[11px] leading-relaxed text-slate-400 whitespace-pre-wrap max-h-[480px] overflow-auto"></div></div>
</section>

<section id="copilot" class="glass rounded-3xl p-4 sm:p-6">
 <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-white/5 pb-4"><div class="flex items-center gap-3"><div class="iconbox text-purple-400 bg-purple-500/10 border-purple-500/20"><i class="fa-solid fa-brain"></i></div><div><h2 class="font-bold">Research AI Copilot</h2><p class="text-[10px] text-slate-500">Optional OpenRouter-powered scientific assistant</p></div></div><span id="copilotStatus" class="pill text-emerald-400"><span class="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5 pulse"></span>READY</span></div>
 <div class="flex flex-wrap gap-2 mt-4"><button onclick="ask('Explain the molecular binding mechanism against the current target.')" class="pill hover:text-white"><i class="fa-solid fa-link mr-1"></i>Binding mechanism</button><button onclick="ask('Suggest bioisosteric substitutions and SAR strategies.')" class="pill hover:text-white"><i class="fa-solid fa-shuffle mr-1"></i>SAR / bioisosteres</button></div>
 <div id="chat" class="mt-4 bg-slate-950/80 rounded-2xl p-4 min-h-[130px] max-h-[300px] overflow-y-auto space-y-2 text-[11px]"><div class="bg-slate-900 rounded-xl p-3 text-slate-400"><i class="fa-solid fa-sparkles text-purple-400 mr-2"></i>Ask a scientific question about the current molecule and target.</div></div>
 <div class="relative mt-3"><input id="question" onkeydown="if(event.key==='Enter')sendQuestion()" class="w-full bg-slate-950 border border-slate-700 focus:border-purple-500 focus:ring-4 focus:ring-purple-500/10 rounded-xl px-4 py-3 pr-24 text-xs outline-none" placeholder="Ask a scientific question..."><button onclick="sendQuestion()" class="absolute right-2 top-1.5 px-3 py-2 rounded-lg bg-purple-600 hover:bg-purple-500 text-xs font-bold"><i class="fa-solid fa-arrow-up mr-1"></i>Query</button></div>
</section>

<section class="rounded-2xl p-4 bg-slate-950/45 border border-white/5 text-[11px] text-slate-500 leading-relaxed flex items-start gap-3"><i class="fa-solid fa-circle-exclamation text-amber-500 mt-0.5"></i><p><b class="text-slate-300">Scientific disclaimer:</b> KBioX AI is intended for academic hypothesis generation, computational screening and educational use. In silico predictions, including activity labels and estimated potency, are not experimental proof of efficacy, toxicity, safety or clinical utility.</p></section>
</main>

<footer class="border-t border-white/10 bg-[#05080e] py-7 px-4"><div class="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3 text-[10px] text-slate-600"><div><b class="text-slate-300">KBioX AI © 2026</b><span class="mx-2">•</span>Where Biology Meets Intelligence.<p class="mt-1 text-slate-500">Created, Developed and Designed by Kharan P, VIT</p></div><div class="flex gap-4"><a href="#studio" class="hover:text-white">Studio</a><a href="#analytics" class="hover:text-white">Analytics</a><a href="#copilot" class="hover:text-white">Copilot</a></div></div></footer>

<div id="toast" class="toast fixed bottom-5 left-1/2 -translate-x-1/2 translate-y-24 opacity-0 z-[100] glass rounded-xl px-4 py-3 text-xs shadow-2xl"><i id="toastIcon" class="fa-solid fa-circle-check text-emerald-400 mr-2"></i><span id="toastText">Done</span></div>

<script>
const state={target:'HIV',result:null,atoms:[],bonds:[],rotX:.25,rotY:.45,drag:false,lastX:0,lastY:0};
const $=id=>document.getElementById(id);
function toast(msg,type='ok'){const t=$('toast');$('toastText').textContent=msg;$('toastIcon').className='fa-solid '+(type==='err'?'fa-circle-exclamation text-red-400 mr-2':'fa-circle-check text-emerald-400 mr-2');t.classList.remove('translate-y-24','opacity-0');setTimeout(()=>t.classList.add('translate-y-24','opacity-0'),2800)}
function esc(s){return String(s).replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[m]))}
function selectTarget(t){state.target=t;document.querySelectorAll('.target').forEach(x=>x.classList.remove('active'));$('target-'+t).classList.add('active');$('targetBadge').textContent=t;runScreening()}
function setSMILES(s,t){$('smilesInput').value=s;selectTarget(t)}
function clearSMILES(){$('smilesInput').value='';$('validity').textContent='EMPTY';$('validity').className='pill text-amber-400'}
async function copySMILES(){try{await navigator.clipboard.writeText($('smilesInput').value);toast('SMILES copied')}catch(e){toast('Clipboard unavailable','err')}}
async function canonicalize(){const s=$('smilesInput').value.trim();if(!s){toast('Enter a SMILES first','err');return}try{const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({smiles:s,target:state.target})});const d=await r.json();if(!r.ok)throw Error(d.detail||'Invalid SMILES');$('smilesInput').value=d.canonicalSmiles;$('validity').textContent='✓ VALID';$('validity').className='pill text-emerald-400';toast('SMILES validated and canonicalized')}catch(e){showError(e.message)}}
function showError(msg){$('errorBox').classList.remove('hidden');$('errorBox').querySelector('span').textContent=msg;$('validity').textContent='✕ INVALID';$('validity').className='pill text-red-400'}
async function runScreening(){
  const smiles=$('smilesInput').value.trim();
  if(!smiles){showError('SMILES input is empty.');return}
  const btn=$('runBtn');
  $('errorBox').classList.add('hidden');
  $('pipeline').classList.remove('hidden');
  btn.disabled=true;btn.classList.add('opacity-70');
  const setProgress=(txt,pct)=>{
    $('pipelineText').innerHTML='<i class="fa-solid fa-circle-notch fa-spin mr-1 text-blue-400"></i>'+txt;
    $('pipelinePct').textContent=`Screening progress is ${pct} percent complete.`;
    $('pipelineBar').style.width=pct+'%';
  };
  let requestDone=false,requestError=null;
  const requestPromise=fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({smiles,target:state.target})})
    .then(async r=>{const d=await r.json();if(!r.ok)throw Error(d.detail||'Screening failed');requestDone=true;return d;})
    .catch(e=>{requestError=e;requestDone=true;throw e;});
  try{
    setProgress('Parsing molecular structure...',12);await new Promise(r=>setTimeout(r,180));
    setProgress('Generating conformer...',28);await new Promise(r=>setTimeout(r,220));
    setProgress('Calculating descriptors...',46);await new Promise(r=>setTimeout(r,220));
    setProgress('Running target model...',68);
    let livePct=68;
    while(!requestDone){
      await new Promise(r=>setTimeout(r,180));
      if(!requestDone){livePct=Math.min(92,livePct+2);setProgress(livePct<84?'Running target model...':'Preparing explainability...',livePct);}
    }
    const d=await requestPromise;
    setProgress('Finalizing scientific report...',96);await new Promise(r=>setTimeout(r,220));
    setProgress('Screening complete.',100);await new Promise(r=>setTimeout(r,180));
    state.result=d;state.atoms=d.geometry.atoms||[];state.bonds=d.geometry.bonds||[];renderResult(d);
    toast('Screening complete — all result panels updated');
  }catch(e){const msg=(requestError||e).message||'Screening failed';showError(msg);toast(msg,'err');$('pipelineText').innerHTML='<i class="fa-solid fa-circle-exclamation mr-1 text-red-400"></i>Screening stopped';}
  finally{btn.disabled=false;btn.classList.remove('opacity-70');setTimeout(()=>$('pipeline').classList.add('hidden'),900);}
}
function renderResult(d){const p=d.properties;$('compoundName').textContent=d.compoundName;$('targetBadge').textContent=state.target;$('confidence').textContent=`The model reports a confidence of ${Number(d.confidence).toFixed(1)} percent.`;$('confidenceBar').style.width=Math.min(100,d.confidence)+'%';$('pic50').textContent=d.pic50;$('mw').textContent=p.mw;$('logp').textContent=p.logp;$('pMW').textContent=p.mw;$('pLogP').textContent=p.logp;$('pHBD').textContent=p.hbd;$('pHBA').textContent=p.hba;$('pTPSA').textContent=p.tpsa;$('pRotB').textContent=p.rotb;$('pQED').textContent=p.qed;$('pHeavy').textContent=p.heavyAtoms??'—';$('pViol').textContent=p.violations;$('lipinski').textContent=p.violations===0?'✓ 0 Lipinski violations':p.violations+' violation(s)';$('lipinski').className='pill '+(p.violations===0?'text-emerald-400':'text-amber-400');$('atomBondCount').textContent=`The molecular model contains ${state.atoms.length} atoms and ${state.bonds.length} bonds.`;$('validity').textContent='✓ RDKit VERIFIED';$('validity').className='pill text-emerald-400';if(d.isActive){$('verdictPill').textContent='PREDICTED ACTIVE';$('verdictPill').className='pill text-emerald-400';$('verdict').textContent='Predicted Active'}else{$('verdictPill').textContent='LOW / INACTIVE';$('verdictPill').className='pill text-red-400';$('verdict').textContent='Predicted Inactive'}$('summary').innerHTML=formatSummaryPoints(d.aiSummary || `Prediction source: ${d.predictionSource}.`);const kp=$('aiKeypoints'),card=$('aiKeypointsCard');if((d.aiKeyPoints||[]).length){kp.innerHTML=d.aiKeyPoints.map((x,i)=>`<div class="flex gap-3 items-start pb-2 border-b border-white/5 last:border-0"><span class="text-purple-400 font-bold text-sm min-w-[1.1rem]">${i+1}.</span><span class="leading-relaxed">${formatScientificText(x)}</span></div>`).join('');card.classList.remove('hidden')}else{card.classList.add('hidden')}$('svg2d').innerHTML=d.geometry.svg2d||'';const svg=$('svg2d').querySelector('svg');if(svg){svg.setAttribute('preserveAspectRatio','xMidYMid meet');svg.style.width='100%';svg.style.height='100%';svg.style.maxWidth='100%';svg.style.maxHeight='100%';}renderAttribution(d.shapFeatures||[]);drawRadar(p);renderDossier(d)}
function formatSummaryPoints(text){
  const raw=String(text||'').replace(/\r/g,'').replace(/\n\s*#\s*$/,'').trim();
  const cleaned=raw.replace(/\s+/g,' ');
  // Split only at sentence punctuation followed by whitespace/end so decimal values like 72.0% and 6.7 stay intact.
  const sentences=cleaned.match(/.+?(?:[.!?]+(?=\s|$)|$)/g)||[cleaned];
  const pts=sentences.map(x=>x.trim()).filter(Boolean).slice(0,6);
  return '<ul class="ai-summary-points">'+pts.map(x=>'<li>'+formatInlineScientific(x)+'</li>').join('')+'</ul>';
}
function formatInlineScientific(text){
  let x=esc(String(text));
  x=x.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');
  x=x.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g,'<em>$1</em>');
  x=x.replace(/`([^`]+)`/g,'<code>$1</code>');
  return x;
}
function formatScientificText(text){
  if(!text) return '';
  const raw=String(text).replace(/\r/g,'').replace(/\n\s*#\s*$/,'').trim();
  const lines=raw.split('\n');
  let html=''; let para=[]; let listType=null;
  const inline=(v)=>{
    let x=esc(String(v));
    x=x.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');
    x=x.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g,'<em>$1</em>');
    x=x.replace(/`([^`]+)`/g,'<code>$1</code>');
    return x;
  };
  const flushPara=()=>{if(para.length){html+='<p>'+para.map(inline).join(' ')+'</p>';para=[]}};
  const closeList=()=>{if(listType){html+=`</${listType}>`;listType=null}};
  for(let line of lines){
    line=line.trim();
    if(!line){flushPara();closeList();continue;}
    let m=line.match(/^#{1,3}\s+(.+)$/);
    if(m){flushPara();closeList();html+='<h4>'+inline(m[1])+'</h4>';continue;}
    m=line.match(/^(\d+)\.\s+(.+)$/);
    if(m){flushPara();if(listType!=='ol'){closeList();html+='<ol>';listType='ol';}html+='<li><strong>'+m[1]+'.</strong> '+inline(m[2])+'</li>';continue;}
    m=line.match(/^[-•]\s+(.+)$/);
    if(m){flushPara();if(listType!=='ul'){closeList();html+='<ul>';listType='ul';}html+='<li>'+inline(m[1])+'</li>';continue;}
    flushPara();closeList();para.push(line);
  }
  flushPara();closeList();
  return '<div class="ai-text">'+html+'</div>';
}
function renderAttribution(items){$('shapBox').innerHTML='';items.forEach(f=>{const v=Number(f.value??f.shap??0),pos=v>=0,w=Math.min(100,Math.abs(v)*220);$('shapBox').insertAdjacentHTML('beforeend',`<div class="metric rounded-xl p-3 reveal"><div class="flex justify-between gap-3 text-[11px]"><span class="font-medium">${esc(f.fragment)}</span><b class="${pos?'text-emerald-400':'text-red-400'}">${pos?'+':''}${v.toFixed(2)}</b></div><div class="h-2 bg-slate-950 rounded-full mt-2 overflow-hidden"><div class="h-full ${pos?'bg-emerald-400':'bg-red-500'} rounded-full transition-all duration-700" style="width:${w}%"></div></div></div>`)})}
function drawRadar(p){const c=$('radarCanvas'),ctx=c.getContext('2d'),cx=c.width/2,cy=c.height/2,r=105;ctx.clearRect(0,0,c.width,c.height);const axes=['MW','LogP','TPSA','RotB','HBD','HBA'],vals=[Math.min(1,p.mw/600),Math.min(1,Math.max(0,(p.logp+2)/7)),Math.min(1,p.tpsa/160),Math.min(1,p.rotb/12),Math.min(1,p.hbd/6),Math.min(1,p.hba/12)];ctx.strokeStyle='rgba(255,255,255,.12)';ctx.fillStyle='#64748b';ctx.font='10px JetBrains Mono';ctx.textAlign='center';for(let l=1;l<=4;l++){const rr=r*l/4;ctx.beginPath();axes.forEach((_,i)=>{const a=i*Math.PI*2/axes.length-Math.PI/2,x=cx+rr*Math.cos(a),y=cy+rr*Math.sin(a);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.closePath();ctx.stroke()}axes.forEach((n,i)=>{const a=i*Math.PI*2/axes.length-Math.PI/2;ctx.fillText(n,cx+(r+21)*Math.cos(a),cy+(r+21)*Math.sin(a)+4)});ctx.beginPath();vals.forEach((v,i)=>{const a=i*Math.PI*2/axes.length-Math.PI/2,rr=r*v,x=cx+rr*Math.cos(a),y=cy+rr*Math.sin(a);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.closePath();ctx.fillStyle='rgba(59,130,246,.22)';ctx.fill();ctx.strokeStyle='#3b82f6';ctx.lineWidth=3;ctx.stroke()}
function renderDossier(d){const p=d.properties;const interpretation=formatScientificText(d.aiInterpretation||d.aiSummary||'No AI interpretation available.');const keypoints=(d.aiKeyPoints||[]).map((x,i)=>`<div class="flex gap-2"><b class="text-purple-400">${i+1}.</b><span>${formatScientificText(x)}</span></div>`).join('');const html=`<div class="ai-text"><h4>KBioX AI COMPREHENSIVE COMPUTATIONAL SCREENING DOSSIER</h4><p><b>Timestamp:</b> ${new Date().toISOString()}<br><b>Platform:</b> KBioX AI v2.8 PRO<br><b>Target:</b> ${esc(state.target)}<br><b>Prediction source:</b> ${esc(d.predictionSource)}</p><h4>1. Chemical Specification</h4><p><b>Compound:</b> ${esc(d.compoundName)}<br><b>Canonical SMILES:</b> <code>${esc(d.canonicalSmiles)}</code><br><b>Atoms:</b> ${state.atoms.length} &nbsp; <b>Bonds:</b> ${state.bonds.length}</p><h4>2. Bioactivity Prediction</h4><p><b>Classification:</b> ${d.isActive?'PREDICTED ACTIVE':'INACTIVE / LOW PREDICTED ACTIVITY'}<br><b>Confidence:</b> ${d.confidence}%<br><b>Estimated pIC50:</b> ${d.pic50}*</p><h4>3. Physicochemical Properties</h4><p><b>MW:</b> ${p.mw} g/mol &nbsp; <b>LogP:</b> ${p.logp} &nbsp; <b>TPSA:</b> ${p.tpsa} Å²<br><b>HBD:</b> ${p.hbd} &nbsp; <b>HBA:</b> ${p.hba} &nbsp; <b>Rotatable bonds:</b> ${p.rotb}<br><b>QED:</b> ${p.qed} &nbsp; <b>Heavy atoms:</b> ${p.heavyAtoms??'N/A'} &nbsp; <b>Lipinski violations:</b> ${p.violations}</p><h4>4. Explainability / Attribution</h4><p>${(d.shapFeatures||[]).map(f=>`<b>${esc(f.fragment||'Feature')}:</b> ${Number(f.value??f.shap??0).toFixed(2)}`).join('<br>')}</p><h4>5. AI Scientific Key Points</h4>${keypoints||'<p>AI key points unavailable.</p>'}<h4>6. Scientific Interpretation</h4>${interpretation}<h4>7. Limitations</h4><p><i>This is an in silico screening result. Experimental validation is required. AI-generated interpretation should be treated as hypothesis-generating scientific support, not experimental evidence.</i><br>* Estimated pIC50 is not a measured value unless the supplied model was trained for pIC50 regression.</p></div>`;$('dossierText').innerHTML=html}
function copyDossier(){if(!state.result){toast('Run screening first','err');return}navigator.clipboard.writeText($('dossierText').innerText).then(()=>toast('Full dossier copied')).catch(()=>toast('Clipboard unavailable','err'))}
function openTab(id,btn){document.querySelectorAll('.tabContent').forEach(x=>x.classList.add('hidden'));document.querySelectorAll('#analytics .tab').forEach(x=>x.classList.remove('active'));$(id).classList.remove('hidden');btn.classList.add('active');if(id==='radar'&&state.result)drawRadar(state.result.properties)}
function switchView(mode){if(mode==='3d'){$('molCanvas').classList.remove('hidden');$('svg2d').classList.add('hidden');$('svg2d').classList.remove('flex');$('v3').classList.add('active');$('v2').classList.remove('active')}else{$('molCanvas').classList.add('hidden');$('svg2d').classList.remove('hidden');$('svg2d').classList.add('flex');$('v2').classList.add('active');$('v3').classList.remove('active');const s=$('svg2d').querySelector('svg');if(s){s.setAttribute('preserveAspectRatio','xMidYMid meet');s.style.width='100%';s.style.height='100%';s.style.maxWidth='100%';s.style.maxHeight='100%';}}}
document.addEventListener('dblclick',e=>{if(e.target.closest('#v2,#svg2d'))e.preventDefault()},{passive:false});
async function ask(q){$('question').value=q;sendQuestion()}
let copilotHistory=[];
async function sendQuestion(){
  const q=$('question').value.trim(); if(!q)return;
  const chat=$('chat');
  chat.insertAdjacentHTML('beforeend',`<div class="flex justify-end"><div class="bg-blue-600/20 border border-blue-500/10 rounded-2xl rounded-tr-md p-3 text-right max-w-[88%]">${esc(q)}</div></div>`);
  $('question').value='';
  const id='m'+Date.now();
  chat.insertAdjacentHTML('beforeend',`<div id="${id}" class="bg-slate-900/90 border border-white/5 rounded-2xl rounded-tl-md p-3.5 text-slate-400 max-w-[94%]"><div class="flex items-center gap-2 text-purple-300 text-[10px] font-bold uppercase tracking-wider mb-2"><i class="fa-solid fa-sparkles"></i> KBioX AI Copilot</div><div><i class="fa-solid fa-circle-notch fa-spin mr-2"></i>Analyzing the molecular context...</div></div>`);
  chat.scrollTop=chat.scrollHeight;
  try{
    const history=copilotHistory.slice(-6);
    const r=await fetch('/api/copilot',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,target:state.target,smiles:$('smilesInput').value,history})});
    const d=await r.json();
    const answer=d.response||'No response returned.';
    $(id).innerHTML=`<div class="flex items-center gap-2 text-purple-300 text-[10px] font-bold uppercase tracking-wider mb-2"><i class="fa-solid fa-sparkles"></i> KBioX AI Copilot</div>${formatScientificText(answer)}`;
    copilotHistory.push({role:'user',content:q},{role:'assistant',content:answer});
  }catch(e){$(id).innerHTML='<strong class="text-red-400">Copilot error:</strong> '+esc(e.message)}
  chat.scrollTop=chat.scrollHeight;
}
async function downloadPdfReport(){if(!state.result){toast('Run screening before exporting','err');return}try{const r=await fetch('/api/generate-pdf',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({target:state.target,smiles:state.result.canonicalSmiles,name:state.result.compoundName,properties:state.result.properties,prediction:state.result,ai_summary:state.result.aiSummary,ai_interpretation:state.result.aiInterpretation||state.result.aiSummary,ai_keypoints:state.result.aiKeyPoints||[]})});if(!r.ok)throw Error('PDF generation failed');const b=await r.blob(),u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download='KBioX AI_'+state.target+'_Complete_Dossier.pdf';document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(u);toast('Complete PDF downloaded')}catch(e){toast(e.message,'err')}}
let canvas,ctx;function init3D(){canvas=$('molCanvas');ctx=canvas.getContext('2d');resizeCanvas();window.addEventListener('resize',resizeCanvas);canvas.addEventListener('pointerdown',e=>{state.drag=true;state.lastX=e.clientX;state.lastY=e.clientY;canvas.setPointerCapture(e.pointerId)});canvas.addEventListener('pointermove',e=>{if(!state.drag)return;state.rotY+=(e.clientX-state.lastX)*.012;state.rotX+=(e.clientY-state.lastY)*.012;state.lastX=e.clientX;state.lastY=e.clientY});['pointerup','pointercancel','pointerleave'].forEach(x=>canvas.addEventListener(x,()=>state.drag=false));requestAnimationFrame(render3D)}function resizeCanvas(){if(!canvas)return;const r=canvas.getBoundingClientRect(),d=devicePixelRatio||1;canvas.width=Math.max(1,r.width*d);canvas.height=Math.max(1,r.height*d);ctx.setTransform(d,0,0,d,0,0)}function render3D(){if(!canvas||!ctx){requestAnimationFrame(render3D);return}const d=devicePixelRatio||1,w=canvas.width/d,h=canvas.height/d;ctx.clearRect(0,0,w,h);if(!state.drag&&state.atoms.length)state.rotY+=.004;const cx=w/2,cy=h/2,sx=Math.sin(state.rotX),cxr=Math.cos(state.rotX),sy=Math.sin(state.rotY),cyr=Math.cos(state.rotY);const pts=state.atoms.map(a=>{const x=a.x*cyr+a.z*sy,z1=-a.x*sy+a.z*cyr,y=a.y*cxr-z1*sx,z=a.y*sx+z1*cxr,sc=260/(260+z);return {...a,px:cx+x*sc,py:cy+y*sc,depth:z,scale:sc}});state.bonds.forEach(([u,v])=>{const a=pts[u],b=pts[v];if(!a||!b)return;ctx.strokeStyle='#475569';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(a.px,a.py);ctx.lineTo(b.px,b.py);ctx.stroke()});pts.sort((a,b)=>a.depth-b.depth).forEach(a=>{ctx.fillStyle=a.color;ctx.beginPath();ctx.arc(a.px,a.py,Math.max(2.5,a.size*a.scale),0,Math.PI*2);ctx.fill()});requestAnimationFrame(render3D)}
async function loadHealth(){try{const r=await fetch('/api/health'),d=await r.json();$('modelStatus').textContent=Object.entries(d.models).map(([k,v])=>k+': '+(v?'loaded':'fallback')).join(' • '); const cs=$('copilotStatus'); if(cs){cs.innerHTML=d.aiApiLinked?'<span class="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5 pulse"></span>AI LINKED':'<span class="inline-block w-1.5 h-1.5 rounded-full bg-amber-400 mr-1.5"></span>API NOT LINKED';cs.className='pill '+(d.aiApiLinked?'text-emerald-400':'text-amber-400')} }catch(e){$('modelStatus').textContent='Backend unavailable';$('serverPill').innerHTML='<span class="inline-block w-1.5 h-1.5 rounded-full bg-red-400 mr-1.5"></span>ERROR'}}
window.addEventListener('load',()=>{init3D();loadHealth();runScreening()});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE

# ================================================================
# 8. START SERVER SAFELY
# ================================================================
# IMPORTANT:
# Render starts this module with `uvicorn main:app`.
# Therefore, do NOT start a second Uvicorn server at import time.
# The self-start block below runs only when `python main.py` is used
# directly (e.g. Google Colab/local execution).

def run_standalone_server():
    kill_old_server()

    def server_thread():
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=PORT,
            log_level="warning"
        )

    thread = threading.Thread(target=server_thread, daemon=True)
    thread.start()

    ready = False
    for _ in range(40):
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=.25):
                ready = True
                break
        except Exception:
            time.sleep(.25)

    if not ready:
        raise RuntimeError(f"KBioX AI server did not start on port {PORT}.")

    try:
        from google.colab import output
        print("KBioX AI is running.")
        print("Loaded models:",
              {k: bool(unwrap_model(v)) for k,v in MODELS.items()})

        output.serve_kernel_port_as_iframe(
            PORT,
            width="100%",
            height=1100
        )

        start_free_public_link(PORT)
    except Exception:
        print(f"Server running at http://127.0.0.1:{PORT}")


if __name__ == "__main__":
    run_standalone_server()
