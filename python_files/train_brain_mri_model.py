import argparse
import json
from pathlib import Path

import tensorflow as tf

from prepare_brain_mri_dataset import (
    CLASS_NAMES,
    PROCESSED_DIR,
    prepare_dataset,
)


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
FINE_TUNE_EPOCHS = 8
FINE_TUNE_LAYER_COUNT = 40
PYTHON_DIR = Path(__file__).resolve().parent
MODEL_PATH = PYTHON_DIR / "brain_tumor_classifier.keras"
CLASS_NAMES_PATH = PYTHON_DIR / "brain_mri_class_names.json"


def load_split(split_name, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DIR / split_name,
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=shuffle,
        seed=42,
    ).prefetch(tf.data.AUTOTUNE)


def build_model():
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.08),
        ],
        name="image_augmentation",
    )
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _training_callbacks(patience, min_learning_rate):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=min_learning_rate,
        ),
    ]


def fine_tune_model(model, train_ds, validation_ds, class_weights):
    base_model = model.get_layer("efficientnetb0")
    base_model.trainable = True
    fine_tune_at = max(0, len(base_model.layers) - FINE_TUNE_LAYER_COUNT)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base_model.layers[fine_tune_at:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    baseline_weights = model.get_weights()
    baseline_metrics = model.evaluate(validation_ds, return_dict=True, verbose=0)
    print(f"Validation before fine-tuning: {baseline_metrics}")
    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=_training_callbacks(patience=3, min_learning_rate=1e-7),
    )

    tuned_metrics = model.evaluate(validation_ds, return_dict=True, verbose=0)
    print(f"Validation after fine-tuning: {tuned_metrics}")
    if tuned_metrics["loss"] >= baseline_metrics["loss"]:
        model.set_weights(baseline_weights)
        print("Fine-tuning did not improve validation loss; kept baseline weights.")
    return model


def train_model(fine_tune_existing=False):
    tf.keras.utils.set_random_seed(42)
    split_counts = prepare_dataset()
    print("Dataset split counts:")
    for split_name, class_counts in split_counts.items():
        print(f"  {split_name}: {class_counts}")

    train_ds = load_split("train", shuffle=True)
    validation_ds = load_split("validation", shuffle=False)
    test_ds = load_split("test", shuffle=False)
    total_training_images = sum(split_counts["train"].values())
    class_weights = {
        index: total_training_images
        / (len(CLASS_NAMES) * split_counts["train"][class_name])
        for index, class_name in enumerate(CLASS_NAMES)
    }

    if fine_tune_existing:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Brain MRI model not found: {MODEL_PATH}. "
                "Train it once without --fine-tune-existing first."
            )
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        model = build_model()
        model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=EPOCHS,
            class_weight=class_weights,
            callbacks=_training_callbacks(patience=5, min_learning_rate=1e-6),
        )

    model = fine_tune_model(model, train_ds, validation_ds, class_weights)

    test_metrics = model.evaluate(test_ds, return_dict=True)
    model.save(MODEL_PATH)
    CLASS_NAMES_PATH.write_text(
        json.dumps(CLASS_NAMES, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Test metrics: {test_metrics}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved class names: {CLASS_NAMES_PATH}")
    return model, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fine-tune-existing",
        action="store_true",
        help="Fine-tune the saved brain MRI model instead of starting from scratch.",
    )
    arguments = parser.parse_args()
    train_model(fine_tune_existing=arguments.fine_tune_existing)
