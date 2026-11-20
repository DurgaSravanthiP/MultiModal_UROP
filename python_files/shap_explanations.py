import numpy as np


def _class_values(shap_values, class_index, class_count, sample_rank):
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_index])[0]

    values = np.asarray(shap_values)
    if values.ndim == sample_rank + 2 and values.shape[-1] == class_count:
        return values[(0, *([slice(None)] * sample_rank), class_index)]
    if values.ndim == sample_rank + 2 and values.shape[0] == class_count:
        return values[(class_index, 0, *([slice(None)] * (sample_rank - 1)))]
    if values.ndim == sample_rank + 1 and values.shape[0] == 1:
        return values[0]
    raise ValueError(f"Unexpected SHAP output shape: {values.shape}")


def explain_lung_features(
    model,
    features,
    feature_names,
    class_index,
    class_name,
    baseline_features,
):
    import shap

    input_features = np.asarray(features, dtype=np.float64).reshape(1, -1)
    background = np.asarray(baseline_features, dtype=np.float64).reshape(1, -1)
    np.random.seed(42)
    explainer = shap.KernelExplainer(
        model.predict_proba,
        background,
        link="identity",
    )
    shap_values = explainer.shap_values(
        input_features,
        nsamples=200,
        silent=True,
    )
    class_values = _class_values(
        shap_values,
        class_index,
        len(model.classes_),
        sample_rank=1,
    ).reshape(-1)
    if len(class_values) != len(feature_names):
        raise ValueError("Lung CT SHAP values do not match the feature names.")

    top_indexes = np.argsort(np.abs(class_values))[-8:][::-1]
    return {
        "kind": "features",
        "method": "SHAP KernelExplainer",
        "targetClass": class_name,
        "baseline": "Uniform mid-gray reference image",
        "features": [
            {
                "name": feature_names[index],
                "value": round(float(class_values[index]), 6),
            }
            for index in top_indexes
        ],
        "note": (
            "These are image-derived intensity and distribution features used "
            "by the lung CT classifier, not identified anatomical findings."
        ),
    }


def explain_brain_image(model, image_array, class_index, class_name):
    import shap

    image_batch = np.asarray(image_array, dtype=np.float32)[np.newaxis, ...]
    reference = np.full_like(image_batch, float(np.mean(image_batch)))
    explainer = shap.GradientExplainer(model, reference)
    shap_values = explainer.shap_values(image_batch, nsamples=32)
    pixel_values = _class_values(
        shap_values,
        class_index,
        int(model.output_shape[-1]),
        sample_rank=3,
    )
    if pixel_values.shape != image_batch.shape[1:]:
        raise ValueError("Brain MRI SHAP values do not match the image dimensions.")

    signed_map = pixel_values.sum(axis=-1)
    grid_size = 16
    block_size = signed_map.shape[0] // grid_size
    if signed_map.shape[0] != signed_map.shape[1] or signed_map.shape[0] % grid_size:
        raise ValueError("Brain MRI image dimensions cannot be reduced to the SHAP grid.")
    heatmap = signed_map.reshape(
        grid_size,
        block_size,
        grid_size,
        block_size,
    ).mean(axis=(1, 3))
    scale = float(np.max(np.abs(heatmap)))
    if scale:
        heatmap = heatmap / scale

    return {
        "kind": "image",
        "method": "SHAP GradientExplainer",
        "targetClass": class_name,
        "baseline": "Uniform mean-intensity image",
        "heatmap": np.round(heatmap, 3).tolist(),
        "note": (
            "This highlights image regions that influenced the model score; "
            "it is not a tumor boundary or clinical localization."
        ),
    }
