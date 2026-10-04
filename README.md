# Iris Classifier API

## What the model predicts

The species of an iris flower — setosa, versicolor, or virginica — from four
measurements: sepal length, sepal width, petal length, petal width.

The model is a scikit-learn Random Forest (200 trees) trained on the built-in
Iris dataset (150 rows), exported to `model.pkl` with joblib. Accuracy on the
held-out test split: 90%.

## Example request body for POST /predict

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

Response:

```json
{
  "prediction": 0,
  "class_name": "setosa"
}
```

## Run it locally

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
uvicorn main:app --reload
```

`train.py` trains the model and writes `model.pkl`. The API then runs at
http://127.0.0.1:8000 — open that in a browser for the predictor page, or see
the interactive docs at http://127.0.0.1:8000/docs.

## Endpoints

- `GET /` — simple web page to make predictions from the browser
- `GET /health` — `{"status": "ok", "model_loaded": true}`
- `POST /predict` — takes `{"features": [float, float, float, float]}`, returns
  the predicted class

## Links

Live API: https://iris-classifier-yd5h.onrender.com
Repository: https://github.com/balajiharish75/iris-classifier
