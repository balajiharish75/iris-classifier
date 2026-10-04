# Iris Classifier API

A minimal machine-learning web API that classifies iris flower species from
sepal and petal measurements. Built following the *Getting Started with ML in
Production* workshop workflow:

**Model → FastAPI → Docker → GitHub → Render → Logging & Monitoring**

## Model

- **Algorithm:** Random Forest (200 estimators)
- **Dataset:** [Iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) (scikit-learn built-in, 150 samples)
- **Features:** `sepal_length`, `sepal_width`, `petal_length`, `petal_width`
- **Classes:** `setosa`, `versicolor`, `virginica`
- **Artifact:** `model.pkl` (joblib), trained by `train.py` and committed to this repo

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check (`status`, `model_loaded`) |
| POST | `/predict` | Predict iris species from 4 features |
| GET | `/docs` | Interactive Swagger UI |

### Example request

```bash
curl -X POST https://iris-classifier.onrender.com/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

```json
{"prediction": 0, "class_name": "setosa"}
```

## Run locally

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python train.py              # trains the model, writes model.pkl
uvicorn main:app --reload    # starts the API on http://127.0.0.1:8000
```

## Docker

```bash
docker build -t iris-classifier .
docker run -p 8000:8000 iris-classifier
```

## Deploy on Render

- Config is in [`render.yaml`](render.yaml): Python runtime,
  build `pip install -r requirements.txt`,
  start `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- Every `git push` to `main` triggers a redeploy.

## Logging & Monitoring

Every request is logged to stdout with timestamp, method, path, status,
duration and payload — e.g.:

```
2026-10-04 14:03:11 | INFO | POST /predict -> 200 | 8.2 ms | body={"features":[5.1,3.5,1.4,0.2]}
2026-10-04 14:03:11 | INFO | prediction=0 (setosa) | features=[5.1, 3.5, 1.4, 0.2]
```

On Render these lines appear in the service **Logs** tab.

## Live deployment

- **API:** https://iris-classifier.onrender.com
- **Swagger UI:** https://iris-classifier.onrender.com/docs
- **Health:** https://iris-classifier.onrender.com/health

> Note: the free Render instance sleeps after inactivity; the first request
> after idle takes ~30–50 s to wake it up.
