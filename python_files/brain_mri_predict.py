import json
import sys
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parent
MODEL_PATH = PYTHON_DIR / "brain_tumor_classifier.keras"
CLASS_NAMES_PATH = PYTHON_DIR / "brain_mri_class_names.json"
DEFAULT_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = (224, 224)


def predict(image_path):
    image_file = Path(image_path)
    if not image_file.is_file():
        raise FileNotFoundError(f"Image file not found: {image_file}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Brain MRI model not found: {MODEL_PATH}. "
            "Run train_brain_mri_model.py before analyzing scans."
        )

    class_names = DEFAULT_CLASSES
    if CLASS_NAMES_PATH.is_file():
        class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    if len(class_names) != len(DEFAULT_CLASSES):
        raise ValueError("Brain MRI class names file has an invalid class count.")

    import numpy as np
    import tensorflow as tf

    image = tf.keras.utils.load_img(
        image_file,
        color_mode="rgb",
        target_size=IMAGE_SIZE,
    )
    image_array = tf.keras.utils.img_to_array(image)
    probabilities = tf.keras.models.load_model(MODEL_PATH).predict(
        np.expand_dims(image_array, axis=0),
        verbose=0,
    )[0]
    if len(probabilities) != len(class_names):
        raise ValueError("Brain MRI model output does not match the class names.")

    class_index = int(np.argmax(probabilities))
    return {
        "prediction": class_names[class_index],
        "confidence": round(float(probabilities[class_index]) * 100, 2),
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
