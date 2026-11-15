# MediScan AI: Project Technical Report and Research Paper Draft

**Project:** MediScan AI (also referred to in package names and earlier notebook text as LungScope AI)

**Report prepared:** 15 November 2026

**Project type:** Educational and research prototype for medical-image classification

**Scope:** Brain MRI and lung CT image classification, patient-linked result management, and fixed informational guidance

> **Important:** This report describes the code and data currently present in the repository. It is not evidence of clinical validity. The application is an educational/research prototype, not a medical device or diagnostic service. A qualified healthcare professional must interpret the complete imaging study and clinical context.

## Abstract

MediScan AI is a web application and machine-learning prototype for classifying individual brain MRI and lung CT images. Its user interface supports authenticated users, selection of an imaging modality, upload of one image, display of a predicted class and model confidence, and storage of patient-linked analysis history. The repository includes two image-classification training pipelines based on EfficientNetB0 transfer learning: a four-class brain MRI model and a five-class lung CT model. It also contains a separate, older lung CT pipeline based on hand-engineered image statistics and a histogram-based gradient boosting classifier.

The datasets currently present locally contain 7,200 brain MRI images and 1,535 lung CT images. The brain pipeline preserves the dataset's provided Testing partition and splits Training into training and validation subsets. The newer lung pipeline uses seeded, class-wise 70/15/15 train/validation/test partitions. Both EfficientNet pipelines resize images to 224 × 224 pixels, use online image augmentation, and train a softmax classification head. The brain model has an additional fine-tuning stage.

A significant reproducibility and integration issue is present: the backend invokes `lung_ct_predict.py`, which loads a scikit-learn/joblib model and handcrafted 70-element features, whereas the newer lung training script creates an EfficientNet `.keras` model. The live CT predictor does not load that EfficientNet artifact; it also contains fallback and filename-based label logic. Therefore the existence of a trained EfficientNet model file does not establish that the web application uses it, and current CT outputs cannot be treated as validated model predictions. No evaluation logs or verified performance metrics are available in the repository. These limitations, along with the small and imbalanced lung dataset, the use of single exported images rather than full DICOM studies, and unvalidated clinical guidance, must be addressed before research or clinical claims are made.

**Keywords:** medical image classification, brain MRI, lung CT, EfficientNetB0, transfer learning, histogram-based gradient boosting, web application, decision support

## 1. Introduction

### 1.1 Background and motivation

Medical imaging is a major source of information for the assessment of diseases. This project explores whether image-classification software can provide a convenient, educational demonstration of machine-learning predictions from brain MRI and lung CT images. It combines model code with a user-facing web application so that an image can be submitted, classified, and associated with a patient identifier for later review.

The system has two separate image-classification tasks:

1. **Brain MRI classification:** select one of four image-folder labels.
2. **Lung CT classification:** select one of five image-folder labels.

These are image-level classifications. The repository does not implement full radiology interpretation, segmentation, lesion localization, tumor staging, or a clinical decision rule that combines the image with symptoms, history, laboratory results, or the rest of an imaging study.

### 1.2 Project objectives

- Provide a web interface for selecting Brain MRI or Lung CT and uploading one image.
- Train separate image classifiers for each modality.
- Return one predicted class and a model-produced confidence/probability value.
- Associate each saved analysis with an account, patient identifier, modality, and timestamp.
- Display additional educational guidance associated with the predicted label.
- Clearly position the output as a research aid rather than a medical diagnosis.

### 1.3 Intended users and intended use

The repository documents an educational and research use case. It does not establish intended use in patient care, prospective clinical evaluation, or use by a particular class of medical professional. A predicted label must not be used by itself to diagnose, rule out, stage, or treat a disease.

## 2. System Overview

### 2.1 Main components

The application has three main software layers:

1. **Frontend:** React application that handles account screens, scan-type selection, file upload and preview, prediction display, guidance, and history views.
2. **Backend:** Express API that authenticates requests, validates uploads, invokes a Python script, and stores analysis records in MongoDB.
3. **Python model layer:** separate scripts for preparing datasets, training models, and running image predictions.

At inference time, the intended flow is:

1. The user signs in and enters a patient ID.
2. The user selects Brain MRI or Lung CT and uploads one JPG, PNG, or WEBP file.
3. The frontend submits the image and metadata to the authenticated API.
4. The backend checks the scan type, stores the temporary upload, and starts the Python script associated with that scan type.
5. The Python process returns a class label and confidence value as JSON.
6. On success, the backend retains the uploaded image, stores a MongoDB analysis record, and returns it to the frontend.
7. The frontend displays the result and obtains static guidance for the predicted class.

The upload limit in the backend is 5 MB. The UI accepts common raster image formats; this is not a DICOM ingestion or full-volume processing workflow.

### 2.2 User accounts and result storage

The backend uses MongoDB/Mongoose models for users and analyses. Passwords are stored as bcrypt hashes. Authentication uses signed JWT bearer tokens with a seven-day expiry. An analysis record includes the account's user ID, a required patient ID, scan type, disease-type description, prediction, confidence, retained image name, and timestamps. Analysis-history and image retrieval endpoints check that the record belongs to the authenticated user.

The uploaded image is initially written under a temporary upload directory. After prediction succeeds, it is moved to the history-image directory. Temporary files are cleaned up after the request, and retained files are removed if the analysis is not successfully saved. These mechanisms do not by themselves establish a regulatory privacy or security standard. Patient identifiers and imaging data should only be used with appropriate authorization, consent, access controls, retention policies, and deployment safeguards.

## 3. Data

### 3.1 Dataset organization and observed image counts

The local data folders are organized by modality and class. The repository does not record enough provenance to identify the original dataset publisher, collection sites, patient-level identifiers, inclusion/exclusion criteria, or licensing terms. Those details must be established from the original dataset source before publication.

#### Brain MRI

The brain dataset has `Training` and `Testing` folders, each with four class folders:

| Class folder | Training images | Testing images |
|---|---:|---:|
| glioma | 1,400 | 400 |
| meningioma | 1,400 | 400 |
| notumor | 1,400 | 400 |
| pituitary | 1,400 | 400 |
| **Total** | **5,600** | **1,600** |

The brain dataset therefore contains **7,200 images** locally. After the preparation script's seeded split, the training partition has 1,190 images per class (4,760 total), the validation partition has 210 per class (840 total), and the held-out Testing partition has 400 per class (1,600 total).

#### Lung CT

The lung dataset is organized as five class folders and contains **1,535 images**:

| Class folder | Original images | Train | Validation | Test |
|---|---:|---:|---:|---:|
| adenocarcinoma | 337 | 235 | 51 | 51 |
| Benign cases | 120 | 84 | 18 | 18 |
| large cell carcinoma | 187 | 131 | 28 | 28 |
| Normal cases | 631 | 441 | 95 | 95 |
| squamous cell carcinoma | 260 | 182 | 39 | 39 |
| **Total** | **1,535** | **1,073** | **231** | **231** |

The lung classes are imbalanced: the normal class has 631 images, while the benign class has 120. The split is performed within each class using a fixed pseudorandom seed, with validation and test each assigned approximately 15% of each class and the remainder assigned to training. Counts reflect the local source folders inspected for this report.

### 3.2 How the data is used

The preparation scripts read images from class-labeled directories, validate the expected class-folder names, and copy images into separate processed train, validation, and test directories. The directory name serves as the label when TensorFlow loads the split. The TensorFlow loaders assign integer labels in the explicit class order defined in the scripts.

- **Training data** updates model weights.
- **Validation data** is used during fitting for early stopping and learning-rate adjustment. The brain fine-tuning stage also compares validation loss before and after tuning.
- **Test data** is held out from weight updates and passed to `model.evaluate()` after training.

The brain preparation script keeps the dataset's provided Testing files as test data and randomly separates a portion of Training for validation. It uses seed 42. The lung preparation script constructs all three splits from the five class directories, also with seed 42; it deletes and recreates its processed split directory each time it runs.

Supported image extensions in the preparation scripts are JPG, JPEG, PNG, and WEBP. The TensorFlow pipelines resize images to 224 × 224 and load batches of 32. The older lung feature pipeline instead converts an image to grayscale, resizes it to 128 × 128, and normalizes pixel values to [0, 1].

### 3.3 Data-quality and leakage considerations

The code splits files by class and does not show patient-level grouping, study-level grouping, duplicate-image detection, or site-level separation. Consequently, it is not possible to establish from the code that multiple images from one patient or near-duplicate images cannot appear in different partitions. The official brain Testing partition is retained, but its patient independence and provenance are not documented here. The lung split is randomized at the image-file level. A research-grade evaluation should verify patient-level independence and image duplicates before interpreting test results.

## 4. Algorithms and Model Training

### 4.1 EfficientNetB0 transfer-learning pipeline

The two newer training scripts use **EfficientNetB0**, a convolutional neural network (CNN), initialized with ImageNet weights. The ImageNet-pretrained convolutional layers act as a feature extractor for medical images, while a task-specific classification head is trained on the project's labels.

The model structure in both scripts is:

1. Input image of shape 224 × 224 × 3.
2. Online augmentation layers:
   - random horizontal flip;
   - random rotation with factor 0.05;
   - random zoom with factor 0.08.
3. EfficientNetB0 base with `include_top=False` and ImageNet weights.
4. Global average pooling to turn the spatial feature maps into a feature vector.
5. Dropout with rate 0.3 to regularize the classification head.
6. Dense output layer with softmax activation: four units for brain MRI and five for lung CT.

The softmax layer produces a score for every class. Inference selects the class with the highest output (`argmax`) and reports that score multiplied by 100 as a percentage. This value is a model output, not a calibrated probability of disease and not a measure of clinical certainty.

The training objective is sparse categorical cross-entropy, appropriate to the integer class labels used by the loader. The optimizer is Adam. The scripts compute inverse-frequency-style class weights from training counts:

`weight(class) = total training images / (number of classes × training images in that class)`

This increases the loss contribution of samples from less frequent training classes. It does not rebalance the test set or prove that performance is fair across classes.

The scripts set TensorFlow's random seed to 42 and prefetch datasets using `tf.data.AUTOTUNE`. Since the scripts initialize EfficientNetB0 with pretrained ImageNet weights, a first training run may require network access to obtain those weights if they are not cached.

### 4.2 Brain MRI training schedule

The brain MRI model supports a two-stage schedule:

1. **Feature-extraction stage:** the EfficientNetB0 base is frozen; only the added classification head is trainable. Maximum 20 epochs, batch size 32, Adam learning rate 0.001.
2. **Fine-tuning stage:** the final 40 layers of EfficientNetB0 are made trainable, while earlier layers remain frozen. Batch-normalization layers in the selected fine-tuning region are kept non-trainable. The optimizer is recompiled with Adam learning rate 0.00001 and a maximum of 8 fine-tuning epochs.

Both stages use early stopping on validation loss and reduce the learning rate when validation loss plateaus. The initial stage uses patience 5; fine-tuning uses patience 3. If the fine-tuned model's validation loss is not lower than the pre-fine-tuning baseline, the script restores the baseline weights. After this decision, the script evaluates on the held-out test split and saves the `.keras` model and class-name JSON file.

### 4.3 Newer lung CT training schedule

The newer lung CT EfficientNet script trains a frozen EfficientNetB0 feature extractor and a new five-class head for up to 20 epochs, using batch size 32 and Adam learning rate 0.001. It uses class weights, early stopping with patience 5, and learning-rate reduction on a plateau with factor 0.2, patience 2, and minimum learning rate 0.000001. It then evaluates on the test split and saves `lung_cancer_classifier.keras` and `lung_ct_class_names.json`.

Unlike the brain training script, this lung script does not add a separate EfficientNet fine-tuning phase.

### 4.4 Older lung CT handcrafted-feature pipeline

The repository also contains `train_classifier.py`, which implements a different lung classifier:

1. Convert each image to grayscale, resize to 128 × 128, and scale pixels to [0, 1].
2. Extract a 70-value feature vector:
   - 32-bin whole-image intensity histogram;
   - 16-bin histogram of the central 64 × 64 crop;
   - 14 intensity statistics (mean, standard deviation, and selected percentiles for the full image and central crop);
   - 8 mean-intensity values in radial distance bands.
3. Fit scikit-learn's `HistGradientBoostingClassifier` with `random_state=42`, `max_iter=150`, `learning_rate=0.08`, and `min_samples_leaf=5`.
4. Save the classifier as `lung_cancer_classifier.joblib`.

This is a classical machine-learning approach with manually designed image features, not a CNN. The training script can print training accuracy and test accuracy/classification report when run, but no such output is captured as a verified result in the repository.

## 5. Inference and Prediction Behavior

### 5.1 Brain MRI inference

The backend routes `brain_mri` requests to `brain_mri_predict.py`. That script verifies the input file and model exist, loads the saved class-name mapping when available, resizes and converts the input image to RGB at 224 × 224, then invokes the saved Keras model. It chooses the class with the largest output value and returns its label and score as a percentage. The model is loaded for each prediction request in this script.

The frontend additionally marks a brain MRI result below 70% as "Uncertain" and displays a low-confidence warning recommending radiologist review. This threshold is a user-interface rule; the code does not show that it was selected through clinical validation or probability calibration.

### 5.2 Lung CT inference and pipeline mismatch

The backend routes `lung_ct` requests to `lung_ct_predict.py`. This script uses the handcrafted-feature pipeline: it extracts the 70 features described above and, if present, loads `lung_cancer_classifier.joblib` to call `predict_proba()`. **It does not load `lung_cancer_classifier.keras`.**

This creates an important mismatch:

- `train_lung_ct_model.py` trains and saves an EfficientNet `.keras` model.
- `lung_ct_predict.py`, which is the script actually invoked by the backend, loads the older `.joblib` model instead.
- `train_classifier.py` is the code that produces that `.joblib` artifact.

Accordingly, the presence of the EfficientNet artifact alone does not mean the web application is using that model. The local repository also contains both a `.keras` and a `.joblib` lung artifact, further underscoring the need to select one pipeline and align training, inference, and documentation.

There are additional reliability concerns in the current `lung_ct_predict.py` implementation:

- If the `.joblib` file is absent, the script assigns the first class label and a fixed 85% confidence rather than failing with a missing-model error.
- It searches the supplied original filename and image path for class-name strings. If it finds one, it overwrites the predicted label and raises the reported confidence to at least 96.50%. This means a filename containing a class label can affect the output independently of the model's image evidence.
- It silently ignores errors while loading the optional class-name JSON file.

These behaviors make the CT response unreliable as a model-only prediction and can produce success-shaped outputs without a valid model result. They should be removed or explicitly treated as a non-production development fallback before any evaluation is reported. The report does not regard current CT results as a trustworthy measure of the EfficientNet model.

## 6. Precautions and Clinical Guidance

### 6.1 How guidance is produced

Precautions are not generated by the CNN or by an additional clinical AI model. The frontend calls a JavaScript lookup function that maps the predicted class and scan type to fixed text stored in `frontend/src/utils/clinicalGuidance.js`.

- **Lung CT:** labels such as normal, benign, adenocarcinoma, large cell carcinoma, and squamous cell carcinoma map to predefined summaries, risk labels, precautions, and suggested next steps.
- **Brain MRI:** all predicted classes receive generic guidance to have the complete MRI reviewed by a qualified professional and to consider clinical context. The app does not provide class-specific brain-tumor precautions in this lookup.
- **Unknown labels:** a generic clinical-review message is returned.

This mapping is deterministic and independent of image details other than the predicted class. It does not consider the confidence score, symptoms, patient history, age, prior scans, or other findings. The code contains detailed clinical-sounding statements and specific timeframes; however, the repository does not provide citations or a guideline-validation process for those text entries. The interface's use of the phrase “evidence-based” is therefore not substantiated by documented sources in the project.

### 6.2 Safety interpretation

The precautions must be presented only as unvalidated educational text, not personalized medical recommendations. Some entries describe tests, referral timing, or interventions that require clinical judgment. They should be reviewed by qualified clinicians and supported with authoritative, current references before being used in a research publication or shown as actionable guidance. The application's own disclaimer states that the output is not a substitute for professional diagnosis, advice, or treatment.

## 7. Software Architecture and Project Structure

```text
Multimodal/
├── README.md
├── project_report.md                 # This report
├── backend/
│   ├── .env.example                  # Environment-variable template
│   ├── package.json
│   ├── server.js                     # Express API, auth, uploads, analysis routes
│   ├── middleware/
│   │   └── authMiddleware.js          # JWT validation and account lookup
│   ├── models/
│   │   ├── Analysis.js                # MongoDB analysis schema
│   │   └── User.js                    # MongoDB user schema
│   ├── services/
│   │   └── pythonService.js            # Starts Python prediction process
│   ├── data/                          # Empty directory in the inspected tree
│   └── uploads/                       # Temporary and retained image uploads
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx                     # App routing
│       ├── main.jsx                    # Frontend entry point
│       ├── components/                 # Shared UI, auth, disclaimer, history
│       ├── context/                    # Authentication state
│       ├── layouts/                    # Public, auth, and dashboard layouts
│       ├── pages/                      # Home, diagnosis, results, history, etc.
│       ├── services/                   # API clients for auth and diagnosis
│       └── utils/
│           └── clinicalGuidance.js      # Static class-to-guidance mapping
└── python_files/
    ├── requirements.txt
    ├── datasets/                       # Local source image datasets
    │   ├── Brain Tumor Dataset/
    │   │   ├── Training/<class>/
    │   │   └── Testing/<class>/
    │   └── Lung Cancer Dataset/<class>/
    ├── processed_data/                 # Generated train/validation/test copies
    │   ├── brain_tumor_mri/
    │   └── lung_cancer_ct/
    ├── prepare_brain_mri_dataset.py    # Brain MRI validation and split/copy
    ├── train_brain_mri_model.py        # EfficientNet brain MRI training
    ├── brain_mri_predict.py            # Brain MRI Keras inference
    ├── prepare_lung_ct_dataset.py      # Lung CT seeded split/copy
    ├── train_lung_ct_model.py          # EfficientNet lung CT training
    ├── train_classifier.py             # Older HGB lung training
    ├── lung_ct_predict.py              # Current backend CT prediction script
    └── lung_cancer_model.ipynb         # Notebook invoking the CT trainer
```

The source datasets, processed image splits, uploaded images, and model artifacts are local/generated data rather than source-code modules. Actual local data and artifact files can vary by machine and should be handled according to their respective licenses and privacy requirements.

## 8. Technologies

| Layer | Technologies observed in the repository |
|---|---|
| Frontend | React, React Router, Vite, Tailwind CSS |
| Backend | Node.js, Express, Multer |
| Authentication | bcrypt password hashing, JSON Web Tokens |
| Persistence | MongoDB with Mongoose |
| Deep-learning models | TensorFlow/Keras, EfficientNetB0 |
| Classical lung model | NumPy, Pillow, scikit-learn HistGradientBoostingClassifier, joblib |
| Data handling | Directory-based labels, seeded Python random splits, TensorFlow image datasets |

The Python `requirements.txt` lists TensorFlow, NumPy, and Pillow. The older classical training/inference code imports scikit-learn and joblib, but those are not listed explicitly in that requirements file. The environment needed to run that path should be made reproducible with explicit dependency versions.

### 8.1 Local setup and reproduction

The repository README documents the following Windows workflow. Commands are shown from the indicated project directories:

```powershell
# Python models (from python_files/)
python -m pip install -r requirements.txt
python prepare_brain_mri_dataset.py
python train_brain_mri_model.py
python prepare_lung_ct_dataset.py
python train_lung_ct_model.py

# Backend (from backend/)
Copy-Item .env.example .env
# Configure MONGODB_URI and a private JWT_SECRET in .env before starting.
npm install
npm run dev

# Frontend (from frontend/, in a separate terminal)
npm install
npm run dev
```

The backend defaults to port 5100 and the Vite frontend to port 5174. The frontend development server forwards `/api` requests to the backend. Brain MRI fine-tuning can be run after an initial model has been saved:

```powershell
python train_brain_mri_model.py --fine-tune-existing
```

Training a modality is required before that modality can be analyzed. The CT integration caveat in Section 5.2 still applies: the documented EfficientNet training command does not make the current backend predictor load the resulting `.keras` model.

## 9. Experimental Results and Reproducibility

The training scripts are configured to evaluate the EfficientNet models on their test partitions and print metric dictionaries. The older classical lung training script prints training accuracy and, if a test split is present, test accuracy and a classification report. However, the repository does not include verified training logs, test metric values, confusion matrices, per-class precision/recall, calibration analyses, or independent external-validation results.

Although model artifact files are present locally, their existence does not prove which script produced the current artifact, whether the artifacts correspond to the current dataset split, or how they performed. No accuracy or other performance value is therefore reported here. Such values must be obtained by a controlled, reproducible run with versioned data, dependencies, split definitions, and logs. In particular, CT results must not be attributed to EfficientNet unless the backend inference code is changed to load that exact Keras model and the prediction path is tested end-to-end.

For a reproducible experiment, record at minimum:

- exact dataset name, source URL/citation, license, download date, and any filtering;
- dataset manifest with class counts and patient/study grouping policy;
- code revision and complete dependency versions;
- seeds and split membership;
- training/validation curves and final test metrics;
- class-wise confusion matrix, precision, recall, F1-score, and support;
- calibration method and calibration metrics if confidence is displayed as probability;
- independent-site or external validation where possible;
- model artifact hash and the exact inference implementation used.

## 10. Limitations and Risks

1. **Not clinically validated:** no prospective study, clinician adjudication, regulatory approval, or clinical safety evaluation is documented.
2. **No verified performance report:** the repository does not contain test results that can substantiate accuracy claims.
3. **CT training/inference mismatch:** the web backend calls the handcrafted-feature/joblib predictor rather than the newer EfficientNet CT model.
4. **Unsafe CT fallback and filename override:** the current CT predictor can return a fixed default result when its model is missing and can change outputs based on filename text.
5. **Small and imbalanced lung dataset:** only 1,535 images are present, with class counts from 120 to 631. Image-level random splits may not represent independent patients.
6. **Single-image input:** one resized raster image is not equivalent to reviewing a full CT or MRI study. There is no DICOM series handling, 3D context, or study-level aggregation in the described workflow.
7. **Possible dataset leakage or source bias:** patient-level independence, duplicate images, acquisition sites, and demographic representation are not verified in the code.
8. **Confidence interpretation:** a softmax score or `predict_proba()` value should not be assumed calibrated or interpreted as clinical certainty.
9. **Guidance is static and uncited:** frontend text is keyed to the class label, does not use patient-specific context, and lacks linked evidence in the repository.
10. **Data governance:** patient identifiers and scan images are sensitive. A local demo architecture and per-user authorization are not a substitute for a documented threat model, secure deployment, consent, retention, and audit controls.

## 11. Recommended Next Steps

### Before presenting quantitative findings

- Run the training and evaluation pipelines from a clean environment and preserve complete logs.
- Verify split membership and, where possible, split by patient/study rather than individual image.
- Report per-class metrics and confidence intervals; compare against an appropriate baseline.
- Verify duplicate and near-duplicate images across partitions.
- Establish and cite dataset provenance, licensing, label definitions, and any preprocessing.

### Before relying on CT model output

- Choose one CT classifier as the intended model.
- Make the training artifact, class ordering, preprocessing, and backend predictor consistent.
- Remove the fixed-confidence fallback and filename-derived prediction override.
- Fail explicitly when model files or dependencies are missing.
- Add integration tests proving the backend loads the chosen model and that filenames do not influence prediction.

### Before showing clinical guidance

- Have the guidance reviewed by qualified clinicians.
- Cite current authoritative sources for every recommendation and remove unsupported or overly prescriptive claims.
- Clearly distinguish general educational information from patient-specific care.
- Ensure that low-confidence or out-of-distribution inputs are handled safely; currently the explicit low-confidence UI rule is limited to brain MRI.

### Before any real-world deployment

- Define privacy, consent, retention, access, and incident-response policies.
- Review authentication secrets and deployment configuration.
- Validate upload content beyond client-provided MIME types and test resource limits.
- Conduct security, usability, accessibility, and clinical safety reviews.
- Seek appropriate institutional and regulatory guidance for the intended jurisdiction and use.

## 12. Conclusion

MediScan AI combines two image-classification experiments with a web workflow for patient-linked analysis history and class-specific informational text. The newer model-development path uses ImageNet-initialized EfficientNetB0 for four-class brain MRI and five-class lung CT classification, with class-weighted training and held-out evaluation in code. The repository also retains a separate handcrafted-feature gradient-boosting lung model.

The project is suitable for continued educational and methodological development, but its current evidence does not support diagnostic or clinical performance claims. The most urgent technical issue for interpreting results is the mismatch between the EfficientNet lung CT training artifact and the older model used by the backend. Reproducible evaluation, patient-level data validation, safe inference behavior, documented dataset provenance, and clinician-reviewed guidance are necessary next steps.

## References and evidence

### Project implementation

This report is grounded in the local repository files, principally:

- `README.md`
- `python_files/prepare_brain_mri_dataset.py`
- `python_files/train_brain_mri_model.py`
- `python_files/brain_mri_predict.py`
- `python_files/prepare_lung_ct_dataset.py`
- `python_files/train_lung_ct_model.py`
- `python_files/train_classifier.py`
- `python_files/lung_ct_predict.py`
- `backend/server.js`
- `backend/services/pythonService.js`
- `backend/models/Analysis.js`
- `backend/models/User.js`
- `frontend/src/pages/DiagnosisPage.jsx`
- `frontend/src/utils/clinicalGuidance.js`

### Model reference

Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. Proceedings of the 36th International Conference on Machine Learning, PMLR 97, 6105–6114. This reference describes the EfficientNet family; it does not validate this project's medical-image dataset, model weights, or performance.

### Publication items still to complete

Before using this as a formal research paper, add the dataset's verified citations and license, institutional author details, experimental run information, computed results, and any ethics/approval statement applicable to the data and intended study. Do not present placeholder or unverified performance claims.
