import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.exceptions import InconsistentVersionWarning

from shap_explanations import explain_lung_features

PYTHON_DIR = Path(__file__).resolve().parent
MODEL_PATH = PYTHON_DIR / "lung_cancer_classifier.joblib"
CLASS_NAMES_PATH = PYTHON_DIR / "lung_ct_class_names.json"

DEFAULT_CLASSES = [
    "adenocarcinoma",
    "Benign cases",
    "large cell carcinoma",
    "Normal cases",
    "squamous cell carcinoma",
]


def extract_features_from_array(arr):

    hist, _ = np.histogram(arr, bins=32, range=(0, 1), density=True)
    c = arr[32:96, 32:96]
    c_hist, _ = np.histogram(c, bins=16, range=(0, 1), density=True)

    stats = [
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.percentile(arr, 10)),
        float(np.percentile(arr, 25)),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 75)),
        float(np.percentile(arr, 90)),
        float(np.mean(c)),
        float(np.std(c)),
        float(np.percentile(c, 10)),
        float(np.percentile(c, 25)),
        float(np.percentile(c, 50)),
        float(np.percentile(c, 75)),
        float(np.percentile(c, 90)),
    ]

    h, w = arr.shape
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2.0, w / 2.0
    r = np.hypot(x - center_x, y - center_y)
    r_bins = np.linspace(0, h / 2.0, 9)
    radial = [
        float(np.mean(arr[(r >= r_bins[i]) & (r < r_bins[i + 1])]))
        if np.any((r >= r_bins[i]) & (r < r_bins[i + 1]))
        else 0.0
        for i in range(8)
    ]

    return np.concatenate([hist, c_hist, stats, radial])


def extract_features(img_path):
    img = Image.open(img_path).convert("L").resize((128, 128))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return extract_features_from_array(arr)


def feature_names():
    return (
        [f"Global intensity histogram bin {index}" for index in range(32)]
        + [f"Central intensity histogram bin {index}" for index in range(16)]
        + [
            "Global mean intensity",
            "Global intensity variation",
            "Global intensity 10th percentile",
            "Global intensity 25th percentile",
            "Global median intensity",
            "Global intensity 75th percentile",
            "Global intensity 90th percentile",
            "Central mean intensity",
            "Central intensity variation",
            "Central intensity 10th percentile",
            "Central intensity 25th percentile",
            "Central median intensity",
            "Central intensity 75th percentile",
            "Central intensity 90th percentile",
        ]
        + [f"Radial ring {index} mean intensity" for index in range(1, 9)]
    )


def predict(image_path: str):
    image_file = Path(image_path)
    if not image_file.is_file():
        raise FileNotFoundError(f"Image file not found: {image_file}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Lung CT model not found: {MODEL_PATH}. "
            "Run train_classifier.py before analyzing scans."
        )

    class_names = DEFAULT_CLASSES
    if CLASS_NAMES_PATH.is_file():
        class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    if class_names != DEFAULT_CLASSES:
        raise ValueError("Lung CT class names do not match the model class order.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            model = joblib.load(MODEL_PATH)
    except InconsistentVersionWarning as exc:
        raise RuntimeError(
            "Lung CT model was saved with a different scikit-learn version. "
            "Install the matching version or retrain the model."
        ) from exc
    expected_classes = np.arange(len(class_names))
    if not np.array_equal(model.classes_, expected_classes):
        raise ValueError("Lung CT model classes do not match the class-name mapping.")
    features = extract_features(image_file).reshape(1, -1)
    probabilities = model.predict_proba(features)[0]
    if len(probabilities) != len(class_names):
        raise ValueError("Lung CT model output does not match the class names.")
    class_index = int(np.argmax(probabilities))
    reference_image = np.full((128, 128), 0.5, dtype=np.float32)
    explanation = explain_lung_features(
        model,
        features,
        feature_names(),
        class_index,
        class_names[class_index],
        extract_features_from_array(reference_image),
    )

    return {
        "prediction": class_names[class_index],
        "confidence": round(float(probabilities[class_index]) * 100, 2),
        "explanation": explanation,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "message": "Please provide an image path."}))
        sys.exit(1)

    try:
        print(json.dumps(predict(sys.argv[1])))
    except Exception as exc:
        print(json.dumps({"success": False, "message": str(exc)}))
        sys.exit(1)
