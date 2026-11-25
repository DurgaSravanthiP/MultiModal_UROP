const PRECAUTIONS = [
  {
    title: "Treat this as an unconfirmed model result",
    description:
      "The predicted class is not a diagnosis. Do not use it alone to diagnose, manage, or choose treatment for the patient.",
  },
  {
    title: "Do not infer disease subtype or extent",
    description:
      "The predicted class does not confirm pathology, disease extent, or stage; any concerning finding needs independent clinical assessment.",
  },
  {
    title: "Review the complete imaging study",
    description:
      "This model processes one uploaded image, not the complete CT or MRI study, and does not localize or characterize findings.",
  },
  {
    title: "Interpret the model score cautiously",
    description:
      "The displayed score is not a calibrated probability of disease or clinical certainty.",
  },
  {
    title: "Do not dismiss a negative-looking class",
    description:
      "A Normal, No tumor, or Benign model class does not rule out disease; rely on independent review and clinical assessment.",
  },
];

export function getClinicalGuidance(prediction, scanType = "lung_ct") {
  const modality =
    scanType === "brain_mri"
      ? "Brain MRI"
      : scanType === "lung_ct"
        ? "Lung CT"
        : "medical image";
  const predictedClass = prediction || "Unavailable";
  const isNegativeLikeClass = /normal|no tumor|benign/i.test(predictedClass);
  const precautions = isNegativeLikeClass
    ? PRECAUTIONS
    : PRECAUTIONS.filter(
        (item) => item.title !== "Do not dismiss a negative-looking class",
      );
  const completeStudyLabel =
    scanType === "brain_mri"
      ? "complete MRI, including relevant sequences and slices"
      : scanType === "lung_ct"
        ? "complete CT study, including relevant series and slices"
        : "complete imaging study";
  const resultSpecificReview = isNegativeLikeClass
    ? `Do not use the "${predictedClass}" image-model class to rule out disease; correlate it with independent clinical assessment.`
    : `Treat "${predictedClass}" as an unconfirmed image-model class, not a confirmed diagnosis or disease subtype.`;

  return {
    name: predictedClass,
    category: "Image-level classification",
    riskLevel: "Research model output",
    severity: "warning",
    color: {
      badge: "bg-slate-50 text-slate-700 border-slate-200",
      accent: "#64748b",
      border: "border-slate-400",
    },
    summary: `The model assigned the class "${predictedClass}" to one uploaded ${modality} image. It does not produce a complete study interpretation, identify or localize findings, or use patient-specific clinical information.`,
    precautions,
    nextSteps: [
      resultSpecificReview,
      "Verify the patient identifier, scan type, and image quality before interpreting the result.",
      `Have a qualified clinician independently review the ${completeStudyLabel}.`,
      "Compare with prior imaging and correlate with available history, symptoms, and other clinical findings.",
      "If independent review confirms an abnormality, follow the applicable local diagnostic and referral pathway.",
      "Document the clinician's independent interpretation and follow-up plan.",
    ],
  };
}
