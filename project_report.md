# MediScan AI

## Project Report

**Submitted by:** [Student name(s) and ID(s)]

**Project guide:** [Guide name]

**Department:** Computer Science and Engineering

**Institution:** SRM University-AP
**Project type:** Educational and research prototype

> **Scope note:** This report describes the implementation in this repository. MediScan AI is not a medical device. Its predictions and explanations are not diagnoses or clinical findings.

## Abstract

MediScan AI is a web application for classifying one uploaded brain MRI or lung CT image at a time. A signed-in user selects the scan type, enters a patient identifier, and uploads an image. The application returns a predicted class and a model score, saves the result and image for later review, and displays a SHAP explanation. The user can view previous analyses and print a result summary.

The project contains separate classifiers for the two image types; it does not combine MRI and CT data in one model. The brain MRI predictor uses an EfficientNetB0 model. The current lung CT predictor uses a scikit-learn HistGradientBoosting classifier trained on 70 hand-designed image features. A separate script trains an EfficientNetB0 lung classifier, but the backend does not currently use that model. This difference is important when describing or evaluating the application.

The local datasets contain 7,200 brain MRI images and 1,535 lung CT images. Training scripts prepare data splits and evaluate models, but the repository does not contain verified training logs or test metrics. No accuracy claim is made in this report. The output is an image-level research classification and must not be used by itself for patient care.

## Chapter 1. Introduction

### 1.1 Project overview

Medical image classifiers can be difficult to inspect from a user’s perspective: a model may return a label without showing what influenced that result. This project explores a small end-to-end workflow that joins image classification, a result history, and a model-attribution display.

The application supports two independent tasks:

| Scan type | Classes returned by the model |
|---|---|
| Brain MRI | glioma, meningioma, notumor, pituitary |
| Lung CT | adenocarcinoma, Benign cases, large cell carcinoma, Normal cases, squamous cell carcinoma |

These are the folder labels used by the datasets and model code. They should not be read as confirmed diagnoses. The system processes a single raster image; it does not read a complete MRI or CT study, combine multiple slices, segment a tumour, or use a patient's history.

### 1.2 Project objectives

- Build a web interface for selecting a scan type and submitting an image.
- Train and run image classifiers for brain MRI and lung CT.
- Return the highest-scoring class with the corresponding model score.
- Keep each result linked to a signed-in account, patient identifier, and timestamp.
- Show an explanation of the model output and a fixed clinician-review checklist.
- Make the limits of the prototype visible in its result presentation.

### 1.3 What “multimodal” means in this project

The application accepts two imaging modalities, MRI and CT. They are handled by separate models and separate prediction scripts. There is no feature fusion, shared multimodal network, or joint reasoning across MRI and CT. In this report, “multimodal” therefore refers to the application’s support for two scan types, not to a model that learns from both together.

## Chapter 2. Motivation

The project was built to connect model training with a usable image-analysis workflow. Instead of running a prediction only from a notebook or command line, a user can submit an image through a browser and find the saved output later. The SHAP display was added to show which input image regions or engineered features contributed to the selected class score.

The work is a software and machine-learning prototype. It does not establish that the models are accurate enough for clinical use. That would require documented data provenance, patient-level evaluation, external testing, clinical review, and a safety process that are not present in the repository.

## Chapter 3. Technical Background

### 3.1 EfficientNetB0 and transfer learning

EfficientNetB0 is a convolutional neural network architecture. This project starts from weights trained on ImageNet and replaces the original classification head with a small head for the project’s labels. This is transfer learning: the pretrained network supplies image features, and the project trains a classifier for the new classes.

The classification head uses global average pooling, dropout, and a softmax output. Softmax produces one score per class. The code selects the largest score as the prediction and displays it as a percentage. The value is not calibrated to represent the probability that a patient has a disease.

### 3.2 Handcrafted image features and gradient boosting

The live lung CT pipeline converts an image into numerical intensity features and passes them to a `HistGradientBoostingClassifier`. This scikit-learn model is an ensemble of decision trees built in stages. Unlike EfficientNet, this pipeline does not learn spatial image features directly from pixels; its input is the fixed 70-value feature vector described in Chapter 5.

### 3.3 SHAP

SHAP (SHapley Additive exPlanations) assigns contribution values to input features for a particular model output, relative to a reference input. The contribution may increase or decrease the score for the selected class. It explains model behavior for the supplied input and reference; it does not establish causation or clinical meaning.

This project uses different SHAP explainers for its two live predictors:

- `GradientExplainer` for the differentiable brain MRI neural network.
- `KernelExplainer` for the lung CT classifier’s `predict_proba` output.

The exact inputs, reference values, approximations, and display logic are described in Chapter 5. The choice of reference affects the interpretation of the contributions.

## Chapter 4. System Design and Methodology

### 4.1 Application components

The application has three main parts:

1. **React frontend:** account screens, scan upload, result display, SHAP display, history, profile, and print summary.
2. **Express backend:** authentication, upload validation, calls to the selected Python predictor, and analysis-history API.
3. **Python model code:** dataset preparation, model training, prediction, and SHAP calculations.

MongoDB stores user and analysis records. Uploaded images that are successfully analyzed are retained under the backend’s history-image directory.

### 4.2 Analysis flow

1. The user signs in and opens the analysis page.
2. The user selects Brain MRI or Lung CT, supplies a patient ID, and chooses one JPG, PNG, or WEBP image.
3. The frontend sends the image and scan type to an authenticated Express endpoint.
4. The backend checks that exactly one supported scan type and one image were supplied, then starts the matching Python prediction script.
5. Python preprocesses the image, obtains a prediction and class score, and calculates the SHAP explanation.
6. If prediction succeeds, the backend retains the image and saves the result, explanation, scan type, patient ID, and account ID in MongoDB.
7. The frontend displays the image, predicted class, model score, explanation, review checklist, and print option. The saved item can also be opened from history.

The upload limit is 5 MB. This workflow accepts individual image files; it is not a DICOM upload or full-study viewer.

### 4.3 Dataset and preparation

The local image folders use their directory names as class labels. The preparation scripts accept JPG, JPEG, PNG, and WEBP files and use a fixed random seed of 42.

**Brain MRI:** The local dataset has 5,600 images in `Training` and 1,600 in `Testing`. Each of the four classes has 1,400 training images and 400 testing images. The preparation script separates 15% of each class in `Training` for validation, keeps the provided `Testing` images as the test set, and copies files into processed split folders.

| Split | Images per class | Total |
|---|---:|---:|
| Training after split | 1,190 | 4,760 |
| Validation | 210 | 840 |
| Test (provided Testing folder) | 400 | 1,600 |

**Lung CT:** The local dataset has 1,535 images. The preparation script shuffles each class separately and allocates about 70% to training, 15% to validation, and 15% to test.

| Class folder | Total images | Train | Validation | Test |
|---|---:|---:|---:|---:|
| adenocarcinoma | 337 | 235 | 51 | 51 |
| Benign cases | 120 | 84 | 18 | 18 |
| large cell carcinoma | 187 | 131 | 28 | 28 |
| Normal cases | 631 | 441 | 95 | 95 |
| squamous cell carcinoma | 260 | 182 | 39 | 39 |
| **Total** | **1,535** | **1,073** | **231** | **231** |

The lung classes are not evenly represented. Both scripts split image files rather than grouping by patient or study. The code does not establish that images from the same person, duplicate images, or related studies cannot occur in more than one split. The dataset publisher, license, collection details, and patient-level grouping are also not recorded in the repository; these details should be confirmed before a formal publication.

## Chapter 5. Implementation

### 5.1 Frontend features

The React application uses React Router for page navigation. Its implemented pages include the home page, sign-up and login, dashboard, scan analysis, results, history, profile, About, and How It Works pages. Protected pages require an authenticated session.

The analysis page submits one image and scan type. The result page shows the original uploaded image, predicted label, numeric score, SHAP explanation, fixed clinician-review actions, and a print-summary control. The history page provides access to saved analyses. The review checklist is application text; it is not generated by the classifier or by a generative AI model.

### 5.2 Backend and data storage

The backend is an Express API. It uses Multer for image uploads, Mongoose to communicate with MongoDB, bcrypt to hash passwords, and JSON Web Tokens (JWTs) for authentication. The token created by the current code expires after seven days.

The analysis record includes the account ID, patient ID, scan type, class label, model score, SHAP explanation, saved image name, and creation time. History and saved-image routes check that the requesting account owns the analysis. Temporary uploads are removed after the request; an image is retained for history only after prediction and saving succeed.

### 5.3 Brain MRI model: training and prediction

The MRI training script resizes images to 224 × 224 pixels and loads batches of 32. It applies random horizontal flips, small rotations, and zooms during training. The model consists of:

1. Image augmentation layers.
2. ImageNet-initialized EfficientNetB0 without its original top classifier.
3. Global average pooling.
4. Dropout with rate 0.3.
5. A four-unit softmax layer.

Training uses Adam, sparse categorical cross-entropy, class weights, early stopping, and learning-rate reduction on a validation-loss plateau. Initially, the EfficientNet base is frozen and only the new classification head is trained. A second stage unfreezes the final 40 base-model layers, keeps batch-normalization layers frozen, and uses a lower learning rate. If the fine-tuned model’s validation loss is not lower than the baseline, the script restores the baseline weights. It then evaluates on the held-out test split and saves a Keras model and class-name JSON file.

For prediction, `brain_mri_predict.py` loads the saved model, converts the uploaded image to RGB, resizes it to 224 × 224, and selects the class with the largest model output. The selected score is multiplied by 100 for display. The source code also calculates a SHAP explanation for that predicted class.

### 5.4 Lung CT model: two pipelines in the repository

There are two lung CT model paths, and they should not be described as if they were the same model.

**Model currently used by the web backend:** `lung_ct_predict.py` converts the image to grayscale, resizes it to 128 × 128, scales pixel values to [0, 1], calculates 70 features, and loads `lung_cancer_classifier.joblib`. The model is a scikit-learn `HistGradientBoostingClassifier`; its `predict_proba()` output is used to select the predicted class and score.

The 70 features are:

- 32-bin histogram of the whole image.
- 16-bin histogram of the central 64 × 64 region.
- 14 statistics: means, standard deviations, and selected percentiles for the whole image and central region.
- 8 mean-intensity values from concentric radial bands.

**Separate EfficientNet training script:** `train_lung_ct_model.py` trains a five-class EfficientNetB0 image classifier and saves `lung_cancer_classifier.keras`. However, `lung_ct_predict.py` does not load this Keras model. Running the EfficientNet training script alone therefore does not change the model used by the live web prediction route. The current application’s lung output must be attributed to the handcrafted-feature/joblib pipeline unless the backend predictor is changed and verified.

### 5.5 What the application returns

The Python predictor returns a JSON object with the predicted class, numeric model score, and explanation. The backend adds database and image metadata and returns the saved analysis to the frontend. The result screen presents:

- The selected modality and predicted class.
- The model score, formatted as a percentage.
- The saved input image.
- A SHAP visualization, when explanation data is available.
- A fixed clinician-review checklist and a print option.

For brain MRI, the score comes from the model’s softmax output. For the live lung CT path, it comes from `predict_proba()`. Neither output is shown to be calibrated in the repository. A displayed score such as 90% should be read as the model’s score, not as a 90% chance of disease.

### 5.6 SHAP explanation in this application

SHAP runs as part of the Python prediction request. Its output is stored with the analysis and rendered on the results page. It explains the selected class output for the current model input relative to a chosen reference. It does not identify a tumour boundary, diagnose a condition, or show where a clinical finding is located.

#### Brain MRI: GradientExplainer heatmap

1. The predictor passes the resized 224 × 224 RGB image to the model.
2. The reference is a same-shaped image whose pixel values are all set to the mean intensity of the uploaded image.
3. SHAP’s `GradientExplainer` estimates attribution values for the predicted class. The code requests 32 samples for the approximation.
4. The per-channel values are summed to form a signed two-dimensional map.
5. The map is averaged into a 16 × 16 grid and scaled by its largest absolute value.
6. The frontend overlays the grid on the uploaded image. Red marks positive contributions to the selected model score; blue marks negative contributions. Opacity is based on contribution magnitude.

The display is a coarse visualization of the model’s attribution relative to a uniform mean-intensity reference. It is not a segmentation mask or validated localization. Its colors describe the direction of model-score contribution, not whether a region is medically normal or abnormal.

#### Lung CT: KernelExplainer feature contributions

1. The predictor extracts the 70 features described in Section 5.4.
2. It creates a reference feature vector by extracting the same features from a uniform mid-gray image (pixel value 0.5).
3. SHAP’s model-agnostic `KernelExplainer` calls the classifier’s `predict_proba()` function, using the identity link and 200 samples for its approximation.
4. The code selects the attribution values for the predicted class and returns the eight features with the largest absolute contributions.
5. In the frontend, red bars indicate positive contributions and blue bars indicate negative contributions. The displayed value is multiplied by 100 and labelled in percentage points.

These are contributions of image-intensity statistics and histogram bins. They are not anatomical structures or detected lesions. Because the reference is an artificial uniform image, the explanation is relative to that reference and should not be treated as a comparison with a representative patient population.

#### SHAP limitations

Both explainers approximate feature contributions and depend on the selected reference, model, input preprocessing, and implementation. An attribution answers a narrow question—how the model output changes relative to the reference under the explainer’s assumptions. It does not prove that the model relied on medically meaningful evidence. In this application, SHAP output is an interpretability aid for a research prototype, not a clinical explanation.

## Chapter 6. Hardware and Software Tools

| Component | Technology used |
|---|---|
| Frontend | React, React Router, Vite, Tailwind CSS |
| API server | Node.js, Express |
| Upload handling | Multer |
| Authentication | bcrypt password hashing, JWT |
| Database | MongoDB, Mongoose |
| Brain MRI model | TensorFlow/Keras, EfficientNetB0 |
| Lung CT model used by backend | scikit-learn HistGradientBoostingClassifier, joblib |
| Image and numerical processing | Pillow, NumPy |
| Explanation methods | SHAP GradientExplainer and KernelExplainer |
| Python dependencies | Listed in `python_files/requirements.txt` |

The repository does not specify a required computer, GPU, RAM, or deployment server. The hardware used for the final training run should be recorded by the project team rather than inferred from the source code.

## Chapter 7. Results and Discussion

### 7.1 Implemented result

The implemented workflow connects the browser, API, database, and Python prediction scripts. A successful analysis returns one class and one score for the selected modality, saves the image and analysis record, and makes the SHAP data available on the result page. The application also supports account creation and login, profile and history screens, and printing the displayed summary.

### 7.2 Model evaluation results

The training scripts call `model.evaluate()` for the EfficientNet test splits and print the resulting metrics. The older lung training script can print test accuracy and a classification report. However, the repository does not include verified output logs, confusion matrices, or per-class test metrics. Model files being present locally are not enough to establish how they were trained or how well they performed.

For that reason, this report does not state an accuracy, precision, recall, F1-score, or clinical performance claim. Those values should be added only after a controlled training and evaluation run, with the dataset version, split, dependencies, and output logs recorded. The current lung web output should not be reported as the result of the EfficientNet lung script.

### 7.3 Main limitations

- The application classifies one raster image, not a full imaging study.
- The local data split is image-based; patient-level separation and duplicate checks are not established.
- The lung dataset is small and imbalanced, and its source and license need to be documented.
- The lung training script and live backend use different model types.
- The displayed model score is not shown to be calibrated.
- SHAP attribution is reference-dependent and is not lesion localization.
- No clinical validation, prospective study, or regulatory assessment is documented.
- Patient IDs and scan images are sensitive data. The prototype configuration is not, by itself, evidence of production-grade privacy or security controls.

## Chapter 8. Conclusion and Future Work

MediScan AI is a working research prototype that combines separate brain MRI and lung CT image-classification paths with a browser interface for uploading images and reviewing saved results. Its main implemented features are account-based access, one-image analysis, result history, model-score display, SHAP attribution, and a printable summary.

The most important issue to resolve before interpreting lung CT results is the mismatch between the EfficientNet training script and the handcrafted-feature model used by the backend. The team should select the intended CT model and align its training code, saved artifact, preprocessing, class order, and prediction route. A reproducible evaluation should then report per-class metrics on a patient-independent test set and include dataset provenance.

Further work should also validate image handling on representative inputs, assess confidence calibration, test the explanation displays with domain experts, document data permissions and privacy controls, and obtain clinical and institutional guidance before considering any real-world use. Until those steps are completed, outputs should remain educational and research-only.

## References

1. Tan, M. and Le, Q. V. “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.” *Proceedings of the 36th International Conference on Machine Learning*, PMLR 97, 2019, pp. 6105–6114.
2. Lundberg, S. M. and Lee, S.-I. “A Unified Approach to Interpreting Model Predictions.” *Advances in Neural Information Processing Systems 30*, 2017.
3. Project implementation: `python_files/train_brain_mri_model.py`, `python_files/brain_mri_predict.py`, `python_files/train_lung_ct_model.py`, `python_files/train_classifier.py`, `python_files/lung_ct_predict.py`, `python_files/shap_explanations.py`, `backend/server.js`, and `frontend/src`.

**Before submission:** replace the student and guide placeholders, confirm the official dataset citations and licenses, and add results only after running and saving a reproducible evaluation. Do not copy the sample report’s names, project claims, or performance numbers.
