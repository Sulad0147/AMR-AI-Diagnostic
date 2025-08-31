# AMR-AI-Diagnostic
# AMR-AI-Diagnostic (Repository for sulad0147)

> **Purpose:** A complete, minimal but working AI-powered diagnostic assistant prototype to help reduce antibiotic misuse and flag possible antimicrobial resistance (AMR) cases. This repo contains backend API (FastAPI), a simple ML training script, model serving, API docs, and documentation including README and onboarding guide.

---

## Repository structure

```
AMR-AI-Diagnostic/
├── README.md
├── requirements.txt
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api.py
│   │   ├── models.py
│   │   └── deps.py
│   └── Dockerfile
├── model/
│   ├── train_model.py
│   ├── predict.py
│   └── artefacts/
│       └── simple_model.pkl  # generated after running train
├── docs/
│   ├── API_DOCS.md
│   └── ONBOARDING.md
└── sample_requests/
    └── sample_requests.http
```

---

## README.md

````markdown
# AMR-AI-Diagnostic

A minimal prototype of an AI-powered diagnostic assistant aimed at reducing antibiotic misuse and flagging potential antimicrobial resistance.

## Features
- REST API (FastAPI) that accepts patient data and returns:
  - Infection type prediction (bacterial/viral/fungal)
  - Treatment recommendation (antibiotics suggested or not)
  - Resistance alert if model + historical data indicate possible resistance
- Simple ML model (scikit-learn) trained on synthetic data as a proof-of-concept
- API documentation and onboarding guide

## Requirements
- Python 3.10+
- pip

## Setup (local)
1. Clone the repo (or create repo `sulad0147` on GitHub and push):
```bash
git clone <your-repo-url>
cd AMR-AI-Diagnostic
````

2. Create virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

3. Train the sample model (creates `model/artefacts/simple_model.pkl`):

```bash
python model/train_model.py
```

4. Run the API server:

```bash
uvicorn backend.app.main:app --reload --port 8000
```

5. Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to view interactive API docs.

## Usage

* POST `/predict` with a JSON payload (see sample requests) to receive prediction, recommendation, and flags.

## Project notes

* This is a prototype. Replace synthetic data with real, curated datasets and retrain with domain experts before use in production.

```
```

---

## requirements.txt

```
fastapi==0.95.2
uvicorn[standard]==0.21.1
pydantic==1.10.12
scikit-learn==1.3.2
pandas==2.2.2
joblib==1.3.2
python-multipart==0.0.6
```

---

## backend/app/main.py

```python
from fastapi import FastAPI
from backend.app.api import router

app = FastAPI(title="AMR-AI-Diagnostic API", version="0.1.0")
app.include_router(router)

@app.get("/health")
def health_check():
    return {"status": "ok"}
```

---

## backend/app/api.py

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from backend.app.models import PatientData, PredictionResponse
from model.predict import predict_from_patient

router = APIRouter()

@router.post('/predict', response_model=PredictionResponse)
def predict(data: PatientData):
    try:
        result = predict_from_patient(data.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## backend/app/models.py

```python
from pydantic import BaseModel, Field
from typing import Optional

class PatientData(BaseModel):
    age: int = Field(..., example=42)
    temperature_c: float = Field(..., example=38.5)
    wbc: float = Field(..., example=12.5)  # white blood cell count (10^9/L)
    cough: bool = Field(..., example=True)
    sore_throat: bool = Field(..., example=False)
    urinary_symptoms: bool = Field(..., example=False)
    recent_antibiotics: bool = Field(..., example=False)
    region: Optional[str] = Field(None, example='Lagos')

class PredictionResponse(BaseModel):
    infection_type: str
    probability: float
    recommend_antibiotics: bool
    resistance_alert: bool
    explanation: Optional[str]
```

---

## backend/app/deps.py

```python
# placeholder for dependency injection (e.g., DB connections, config)

def get_resistance_database():
    # In production, connect to a resistance surveillance DB
    # For prototype, use a hardcoded sample or a local file
    return {
        'Lagos': {'amoxicillin': 0.35, 'ciprofloxacin': 0.12},
        'default': {'amoxicillin': 0.2}
    }
```

---

## model/train\_model.py

```python
"""
Train a very small classifier on synthetic data and save it to artefacts/simple_model.pkl
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), 'artefacts')
os.makedirs(ARTIFACT_PATH, exist_ok=True)

# Synthetic dataset generator
np.random.seed(0)
N = 1000
age = np.random.randint(1, 90, N)
temp = np.random.normal(37, 1.5, N)
wbc = np.random.normal(7, 3, N)

# Symptoms
cough = np.random.binomial(1, 0.4, N)
sore_throat = np.random.binomial(1, 0.2, N)
urinary = np.random.binomial(1, 0.1, N)
recent_abx = np.random.binomial(1, 0.15, N)

# Simple target rule: bacterial if high WBC or very high temp and urinary symptoms
is_bacterial = ((wbc > 10) | (temp > 39) | (urinary == 1)).astype(int)

X = pd.DataFrame({
    'age': age,
    'temp': temp,
    'wbc': wbc,
    'cough': cough,
    'sore_throat': sore_throat,
    'urinary': urinary,
    'recent_abx': recent_abx
})

y = is_bacterial

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

joblib.dump(clf, os.path.join(ARTIFACT_PATH, 'simple_model.pkl'))
print('Model trained and saved to', ARTIFACT_PATH)
```

---

## model/predict.py

```python
import os
import joblib
import numpy as np
from backend.app.models import PredictionResponse

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'artefacts', 'simple_model.pkl')

# load model lazily
_model = None

def _load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError('Model artefact not found. Run `python model/train_model.py`')
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_from_patient(patient: dict) -> dict:
    model = _load_model()
    features = [
        patient.get('age', 40),
        patient.get('temperature_c', 37.0),
        patient.get('wbc', 7.0),
        int(patient.get('cough', False)),
        int(patient.get('sore_throat', False)),
        int(patient.get('urinary_symptoms', False)),
        int(patient.get('recent_antibiotics', False)),
    ]
    arr = np.array(features).reshape(1, -1)
    prob = float(model.predict_proba(arr)[0][1]) if hasattr(model, 'predict_proba') else float(model.predict(arr)[0])
    is_bacterial = prob > 0.5

    # Simple decision rule for recommendation
    recommend_antibiotics = bool(is_bacterial)

    # Resistance alert: naive rule using region and recent antibiotics
    region = patient.get('region') or 'default'
    resistance_alert = False
    if patient.get('recent_antibiotics', False):
        resistance_alert = True

    explanation = (
        f"Model probability bacterial={prob:.2f}. "
        f"Recommend antibiotics={recommend_antibiotics}."
    )

    response = {
        'infection_type': 'bacterial' if is_bacterial else 'non-bacterial',
        'probability': prob,
        'recommend_antibiotics': recommend_antibiotics,
        'resistance_alert': resistance_alert,
        'explanation': explanation
    }
    return response
```

---

## backend/Dockerfile

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## docs/API\_DOCS.md

````markdown
# API Documentation (Prototype)

## POST /predict

**Description:** Submit patient data, receive infection prediction and treatment recommendation.

**Request JSON**
```json
{
  "age": 42,
  "temperature_c": 38.5,
  "wbc": 12.5,
  "cough": true,
  "sore_throat": false,
  "urinary_symptoms": false,
  "recent_antibiotics": false,
  "region": "Lagos"
}
````

**Response (200)**

```json
{
  "infection_type": "bacterial",
  "probability": 0.87,
  "recommend_antibiotics": true,
  "resistance_alert": false,
  "explanation": "Model probability bacterial=0.87. Recommend antibiotics=True."
}
```

**Errors**

* 500: Internal server error (e.g., model missing)

**Notes**

* This is a prototype. Clinical use requires validation and regulatory approvals.

````

---

## docs/ONBOARDING.md

```markdown
# Onboarding Guide - AMR-AI-Diagnostic

Welcome! This guide helps you get the prototype running locally.

1. Setup environment.
2. Train model: `python model/train_model.py`.
3. Start server: `uvicorn backend.app.main:app --reload`.
4. Visit `http://127.0.0.1:8000/docs` for the interactive API.

**Testing**: Use `sample_requests/sample_requests.http` or Postman.

**Extending**:
- Replace synthetic dataset with real clinical data (obtain ethics approval).
- Add authentication, audit logging, and secure storage for patient PII.
- Integrate with regional resistance surveillance datasets.
````

---

## sample\_requests/sample\_requests.http

```http
### Predict infection
POST http://127.0.0.1:8000/predict
Content-Type: application/json

{
  "age": 30,
  "temperature_c": 39.2,
  "wbc": 13.0,
  "cough": true,
  "sore_throat": false,
  "urinary_symptoms": false,
  "recent_antibiotics": false,
  "region": "Lagos"
}
```

---

## Notes & Next Steps

* **Safety & Ethics:** This repo is a prototype. Do NOT use for clinical decision-making without validation.
* **Improvements:** Replace the ML model with properly trained models on curated datasets, implement continuous monitoring, CI/CD, container image builds, and secure secrets management.

---

*End of repository scaffold.*
