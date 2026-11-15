import random
import shutil
from pathlib import Path


PYTHON_DIR = Path(__file__).resolve().parent
DATASET_DIR = PYTHON_DIR / "datasets" / "Brain Tumor Dataset"
PROCESSED_DIR = PYTHON_DIR / "processed_data" / "brain_tumor_mri"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _images_for_class(split_dir, class_name):
    class_dir = split_dir / class_name
    if not class_dir.is_dir():
        raise ValueError(f"Missing '{class_name}' class folder in {split_dir}.")
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _destination_entries(images):
    entries = []
    used_names = set()
    for index, source in enumerate(images):
        name = source.name
        if name in used_names:
            name = f"{index}_{name}"
        used_names.add(name)
        entries.append((source, name))
    return entries


def prepare_dataset(seed=42):
    training_dir = DATASET_DIR / "Training"
    testing_dir = DATASET_DIR / "Testing"
    if not training_dir.is_dir() or not testing_dir.is_dir():
        raise FileNotFoundError(
            "Brain MRI dataset must contain Training and Testing folders under "
            f"{DATASET_DIR}."
        )

    for split_dir in (training_dir, testing_dir):
        actual_classes = {
            path.name for path in split_dir.iterdir() if path.is_dir()
        }
        expected_classes = set(CLASS_NAMES)
        if actual_classes != expected_classes:
            missing = sorted(expected_classes - actual_classes)
            unexpected = sorted(actual_classes - expected_classes)
            raise ValueError(
                f"Class folders in {split_dir} do not match. "
                f"Missing: {missing}; unexpected: {unexpected}"
            )

    training_images = {}
    testing_images = {}
    for class_name in CLASS_NAMES:
        training_images[class_name] = _images_for_class(training_dir, class_name)
        testing_images[class_name] = _images_for_class(testing_dir, class_name)
        if len(training_images[class_name]) < 2:
            raise ValueError(
                f"Class '{class_name}' needs at least 2 training images; "
                f"found {len(training_images[class_name])}."
            )
        if not testing_images[class_name]:
            raise ValueError(f"Class '{class_name}' has no test images.")

    rng = random.Random(seed)
    counts = {
        split: {name: 0 for name in CLASS_NAMES}
        for split in ("train", "validation", "test")
    }
    split_images_by_class = {}

    for class_name in CLASS_NAMES:
        images = training_images[class_name].copy()
        rng.shuffle(images)
        validation_count = max(1, round(len(images) * 0.15))
        split_images_by_class[class_name] = {
            "train": images[validation_count:],
            "validation": images[:validation_count],
            "test": testing_images[class_name],
        }

    if PROCESSED_DIR.exists():
        for class_name, splits in split_images_by_class.items():
            for split, paths in splits.items():
                destination = PROCESSED_DIR / split / class_name
                entries = _destination_entries(paths)
                expected_names = {name for _, name in entries}
                if destination.is_dir():
                    actual_names = {
                        path.name
                        for path in destination.iterdir()
                        if path.is_file()
                        and path.suffix.lower() in IMAGE_EXTENSIONS
                    }
                else:
                    actual_names = set()
                unexpected_names = actual_names - expected_names
                if unexpected_names:
                    raise FileExistsError(
                        f"Unexpected images already exist in {destination}; "
                        "they were not overwritten or deleted."
                    )

                destination.mkdir(parents=True, exist_ok=True)
                for source, name in entries:
                    target = destination / name
                    if not target.exists():
                        shutil.copy2(source, target)
                counts[split][class_name] = len(paths)

    else:
        for class_name, splits in split_images_by_class.items():
            for split, paths in splits.items():
                destination = PROCESSED_DIR / split / class_name
                destination.mkdir(parents=True, exist_ok=True)
                for source, name in _destination_entries(paths):
                    shutil.copy2(source, destination / name)
                counts[split][class_name] = len(paths)

    return counts


if __name__ == "__main__":
    for split_name, class_counts in prepare_dataset().items():
        print(f"{split_name}: {class_counts}")
