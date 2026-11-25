import { useEffect, useState } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import ClinicalAssessmentSummary from "../components/ClinicalAssessmentSummary";
import ShapExplanation from "../components/ShapExplanation";
import Button from "../components/Button";
import {
  fetchAnalysisById,
  fetchAnalysisImage,
} from "../services/diagnosisService";
import { getClinicalGuidance } from "../utils/clinicalGuidance";

export default function ResultsPage() {
  const { id } = useParams();
  const { state } = useLocation();
  const [result, setResult] = useState(null);
  const [imageUrl, setImageUrl] = useState("");
  const [imageError, setImageError] = useState("");
  const [loading, setLoading] = useState(Boolean(id));
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    let objectUrl = "";

    async function loadResult() {
      setLoading(Boolean(id));
      setError("");
      setImageError("");
      setImageUrl("");

      try {
        const selectedResult =
          state?.result && (!id || state.result.id === id)
            ? state.result
            : id
              ? await fetchAnalysisById(id)
              : null;

        if (!active) return;
        setResult(selectedResult);

        if (selectedResult?.imageAvailable && selectedResult.id) {
          try {
            const imageBlob = await fetchAnalysisImage(selectedResult.id);
            if (!active) return;
            objectUrl = URL.createObjectURL(imageBlob);
            setImageUrl(objectUrl);
          } catch (requestError) {
            if (active) {
              setImageError(
                requestError.message || "Unable to load the saved scan image.",
              );
            }
          }
        }
      } catch (requestError) {
        if (active) {
          setError(requestError.message || "Unable to load this analysis.");
        }
      } finally {
        if (active) setLoading(false);
      }
    }

    loadResult();
    return () => {
      active = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [id, state]);

  if (loading) {
    return (
      <div className="mx-auto max-w-3xl rounded-2xl border border-slate-200 bg-white p-8 text-center shadow-sm">
        <p className="text-sm text-slate-600">Loading saved analysis…</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="mx-auto max-w-3xl rounded-2xl border border-slate-200 bg-white p-8 text-center shadow-sm">
        <h2 className="text-xl font-semibold text-slate-900">
          {error ? "Unable to load analysis" : "No analysis selected"}
        </h2>
        <p role={error ? "alert" : undefined} className="mt-2 text-sm text-slate-600">
          {error ||
            "Open a saved analysis from patient history to view its results."}
        </p>
        <Link to="/history" className="mt-5 inline-block">
          <Button variant="secondary">View patient history</Button>
        </Link>
      </div>
    );
  }

  const scanLabel =
    result.scanType === "brain_mri"
      ? "Brain MRI"
      : result.scanType === "lung_ct"
        ? "Lung CT"
        : "Unknown scan";
  const guidance = getClinicalGuidance(result.prediction, result.scanType);

  return (
    <div className="analysis-result mx-auto max-w-5xl space-y-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="eyebrow">Saved analysis</p>
          <h2 className="mt-2 text-3xl font-bold text-slate-900">
            {scanLabel} result
          </h2>
          <p className="mt-1 text-sm text-slate-600">
            Patient {result.patientId || "ID unavailable"}
            {result.createdAt &&
              ` · ${new Date(result.createdAt).toLocaleString()}`}
          </p>
        </div>
        <button
          type="button"
          onClick={() => window.print()}
          className="rounded-xl border border-slate-300 px-3.5 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50"
        >
          Print summary
        </button>
      </header>

      <section className="grid gap-5 md:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
        <div className="flex min-h-64 items-center justify-center overflow-hidden rounded-2xl border border-slate-200 bg-slate-950 p-3">
          {imageUrl ? (
            <img
              src={imageUrl}
              alt={`${scanLabel} image for patient ${result.patientId || ""}`}
              className="max-h-[420px] w-full rounded-xl object-contain"
            />
          ) : (
            <p className="text-center text-sm text-slate-300">
              {imageError || "No saved image is available for this analysis."}
            </p>
          )}
        </div>

        <div className="space-y-4 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-xs font-bold uppercase tracking-wider text-slate-500">
              Image model result
            </p>
            <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-semibold text-slate-700">
              Image-level classification
            </span>
          </div>
          <h3 className="text-2xl font-bold text-slate-900">
            {result.prediction || "Unavailable"}
          </h3>
          <div>
            <div className="mb-2 flex items-center justify-between gap-3 text-sm">
              <span className="text-slate-600">
                Model score (not calibrated probability)
              </span>
              <strong className="text-teal-700">
                {Number(result.confidence).toFixed(2)}%
              </strong>
            </div>
            <div
              className="h-2.5 overflow-hidden rounded-full bg-slate-100"
              role="meter"
              aria-label="Model score"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.min(
                100,
                Math.max(0, Number(result.confidence) || 0),
              )}
            >
              <div
                className="h-full rounded-full bg-teal-600"
                style={{
                  width: `${Math.min(100, Math.max(0, Number(result.confidence) || 0))}%`,
                }}
              />
            </div>
          </div>
          <p className="rounded-xl bg-slate-50 p-3 text-sm leading-relaxed text-slate-600">
            {guidance.summary}
          </p>
        </div>
      </section>

      {imageUrl && result.explanation && (
        <ShapExplanation
          explanation={result.explanation}
          imageUrl={imageUrl}
          reportDetails={{
            patientId: result.patientId,
            scanType: scanLabel,
            prediction: result.prediction,
            confidence: result.confidence,
            createdAt: result.createdAt,
          }}
        />
      )}

      <ClinicalAssessmentSummary
        prediction={result.prediction}
        nextStep={guidance.nextSteps[0]}
      />

      <section className="rounded-2xl border border-teal-200 bg-teal-50/60 p-5">
        <h3 className="text-sm font-bold text-teal-950">
          Clinician review actions
        </h3>
        <ol className="mt-3 list-decimal space-y-2 pl-5 text-sm leading-relaxed text-slate-700">
          {guidance.nextSteps.map((step) => (
            <li key={step}>{step}</li>
          ))}
        </ol>
      </section>

      <p className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs leading-relaxed text-amber-900">
        Research-prototype output for clinician review. The model score is not
        calibrated clinical certainty, and this result is not a diagnosis,
        radiology report, or treatment plan.
      </p>

      <div className="flex flex-wrap gap-3">
        <Link to="/analyze">
          <Button>Back to Analysis</Button>
        </Link>
        <Link to="/history">
          <Button variant="secondary">View History</Button>
        </Link>
      </div>
    </div>
  );
}
