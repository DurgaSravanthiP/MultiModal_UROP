# MediScan AI

MediScan AI is a patient-linked medical imaging research workspace for brain
MRI and lung CT image classification. It keeps analysis results organized by
patient and modality.

Supported model classes:

- **Brain MRI:** glioma, meningioma, no tumor, pituitary
- **Lung CT:** adenocarcinoma, benign cases, large cell carcinoma, normal cases,
  squamous cell carcinoma

This project is for educational and research use. Model output and confidence
are not a medical diagnosis and must not replace review by a qualified
healthcare professional.

## Project structure

```text
backend/
  models/                 MongoDB schemas
  services/               Python model integration
  server.js               Express API, authentication, and analysis routes
frontend/
  src/                    React pages, components, and API services
python_files/
  datasets/
    Brain Tumor Dataset/  Training and Testing folders, each with four classes
    Lung Cancer Dataset/  Images grouped into five class folders
  processed_data/         Generated train/validation/test image folders
  prepare_brain_mri_dataset.py
  train_brain_mri_model.py
  brain_mri_predict.py
  prepare_lung_ct_dataset.py
  train_lung_ct_model.py
  lung_ct_predict.py
  shap_explanations.py
```

The brain MRI preparation script keeps the dataset's official `Testing` images
as the held-out test set and creates a reproducible 85/15 train/validation split
from `Training`. The lung CT pipeline creates reproducible class-stratified
70/15/15 train/validation/test splits. Both are stored in their own
`python_files/processed_data/` subfolders.

The brain MRI training and inference path uses an EfficientNetB0 classifier.
The current backend lung CT inference path instead loads the older
handcrafted-feature scikit-learn model (`lung_cancer_classifier.joblib`)
produced by `train_classifier.py`. The newer lung EfficientNet training script
is not connected to backend inference. The selected model, preprocessing, and
class mapping must match before interpreting either output. Lung prediction
also stops if the saved model was created with a different scikit-learn
version; use a compatible environment or retrain the model before analysis.
The current local CT artifact was saved with scikit-learn 1.9.0; this version
is pinned in `python_files/requirements.txt` for reproducible setup.

The result summary combines the model's predicted class and score with the
patient identifier, scan type, and analysis timestamp. Its clinician-review
checklist is fixed application text, not model-generated advice. The model
does not interpret the complete study or use patient history. Brain MRI results
include a SHAP GradientExplainer heatmap; lung CT results include the top
SHAP-ranked handcrafted image features. Both explanations use a uniform
intensity reference and describe model attribution only, not lesion locations,
clinical findings, or calibrated certainty.

## Prepare datasets and train models

From `python_files/`, install dependencies and prepare/train each modality:

```powershell
python -m pip install -r requirements.txt
python prepare_brain_mri_dataset.py
python train_brain_mri_model.py
python train_brain_mri_model.py --fine-tune-existing
python prepare_lung_ct_dataset.py
python train_lung_ct_model.py
```

The training scripts also prepare their dataset automatically before training.
The brain script fine-tunes the EfficientNet feature layers and keeps the
fine-tuned weights only when validation loss improves. Use
`--fine-tune-existing` to fine-tune a previously trained brain model without
starting over.
The first training run needs an internet connection to download EfficientNet's
pretrained ImageNet weights. Training saves the models and class-name files
beside the Python scripts. The processed images and generated model artifacts
are local and excluded from Git.

## Run locally on Windows

### 1. Configure MongoDB and the backend

Start MongoDB. In PowerShell:

```powershell
cd backend
Copy-Item .env.example .env
```

Set `MONGODB_URI` if needed and replace `JWT_SECRET` with a private random
secret. Keep `.env` private; it is excluded from Git.

Install backend dependencies and start the API:

```powershell
npm install
npm run dev
```

The API listens on `http://localhost:5100` by default.

### 2. Start the frontend

Open another PowerShell terminal:

```powershell
cd frontend
npm install
npm run dev
```

Vite runs at `http://localhost:5174` and forwards `/api` requests to the
backend on port 5100.

### 3. Use the application

1. Sign up, then sign in.
2. Open **Analyze Scan** and enter a patient ID.
3. Select **Brain MRI** or **Lung CT**.
4. Upload one JPG, PNG, or WEBP image (up to 5 MB).
5. Review the predicted class and model score alongside the complete original
   imaging study and relevant clinical information.
6. Open **History** to review saved results and timestamps.

Printed summaries are generated from the saved analysis metadata and
classification. They are research-prototype summaries, not radiology reports
or confirmed diagnoses.

Train the model for a modality before running an analysis for that modality.

## Useful commands

From `frontend/`:

```powershell
npm run build
npm run lint
```

From `backend/`:

```powershell
npm run dev
npm start
```

## Data and privacy

- Original datasets and generated image splits are local and excluded from Git.
- Uploaded scan images are kept under `backend/uploads/`.
- Do not upload identifiable patient information or clinical images unless
  you have the necessary authorization and safeguards.
- A model confidence score is not a measure of clinical certainty.
