export default function ClinicalAssessmentSummary({
  prediction,
  nextStep,
}) {
  return (
    <section
      aria-label="Model result and recommended clinical review"
      className="rounded-2xl border border-slate-200 bg-slate-50 p-4 md:p-5"
    >
      <h4 className="text-sm font-bold text-slate-900">
        Model result and recommended review
      </h4>
      <div className="mt-3 grid gap-3 md:grid-cols-3">
        <div className="rounded-xl border border-slate-200 bg-white p-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Model finding
          </p>
          <p className="mt-1 text-sm font-bold text-slate-900">
            {prediction || "Unavailable"}
          </p>
          <p className="mt-1 text-xs leading-relaxed text-slate-600">
            Classification of one uploaded image; not a confirmed condition.
          </p>
        </div>
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-amber-800">
            Clinical interpretation
          </p>
          <p className="mt-1 text-sm font-bold text-amber-900">
            Requires independent review
          </p>
          <p className="mt-1 text-xs leading-relaxed text-amber-900">
            Review the complete study and relevant patient information before
            determining the patient's condition.
          </p>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Recommended next step
          </p>
          <p className="mt-1 text-xs leading-relaxed text-slate-700">
            {nextStep}
          </p>
        </div>
      </div>
      <p className="mt-3 rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs leading-relaxed text-amber-900">
        These are clinician review steps, not a diagnosis or treatment plan.
        Treatment decisions should follow independent clinical assessment.
      </p>
    </section>
  );
}
