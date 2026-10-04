# Iris Classifier API

Predicts iris flower species from sepal and petal measurements.
Project for the Getting Started with ML in Production workshop.

## Model

Random Forest (200 trees) trained on the scikit-learn Iris dataset (150 rows,
4 features), created by `train.py` and saved as `model.pkl` with joblib.
Accuracy on the test split: 90%.

Input features: sepal length, sepal width, petal length, petal width
Classes: setosa, versicolor, virginica

## Endpoints

- `GET /` - service info
- `GET /health` - status and whether the model loaded
- `POST /predict` - predict the class from 4 features
- `GET /docs` - Swagger UI

Example:

```
curl -X POST https://iris-classifier.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

Response: `{"prediction": 0, "class_name": "setosa"}`

## Run locally

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
uvicorn main:app --reload
```

## Docker

```
docker build -t iris-classifier .
docker run -p 8000:8000 iris-classifier
```

## Render

Configured with `render.yaml`:

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Python version pinned in `.python-version` (the pinned numpy only has
  wheels up to Python 3.12, so this matters)

## Logs

Every request is written to stdout: timestamp, method, path, status, duration
and the request body, plus each prediction result. On Render these appear in
the service Logs tab.

## Links

Repository: https://github.com/balajiharish75/iris-classifier
Live API: https://iris-classifier.onrender.com
Swagger: https://iris-classifier.onrender.com/docs

The free Render instance sleeps when idle, so the first request after idle
time takes 30-50 seconds to respond.
