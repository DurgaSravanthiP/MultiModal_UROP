import mongoose from "mongoose";

const analysisSchema = new mongoose.Schema(
  {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    patientId: {
      type: String,
      required: [true, "Patient ID is mandatory"],
      trim: true,
    },
    scanType: {
      type: String,
      enum: ["lung_ct", "brain_mri"],
      required: true,
    },
    diseaseType: { type: String, required: true },
    prediction: { type: String, required: true },
    confidence: { type: Number, required: true },
    explanation: { type: mongoose.Schema.Types.Mixed, default: null },
    imageName: { type: String, default: "" },
    createdAt: { type: Date, default: Date.now },
  },
  { timestamps: true },
);

export default mongoose.model("Analysis", analysisSchema);
