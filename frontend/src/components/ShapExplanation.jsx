function FeatureExplanation({ explanation }) {
  const features = explanation.features || [];
  const maxMagnitude = Math.max(
    ...features.map((feature) => Math.abs(feature.value)),
    0.000001,
  );

  return (
    <div
      role="img"
      aria-label={`SHAP feature contributions for ${explanation.targetClass}`}
      className="mt-4 space-y-2"
    >
      {features.map((feature) => {
        const width = `${(Math.abs(feature.value) / maxMagnitude) * 48}%`;
        const positive = feature.value >= 0;
        return (
          <div
            key={feature.name}
            className="grid grid-cols-[minmax(120px,1fr)_minmax(120px,2fr)_64px] items-center gap-2 text-[10px] sm:text-xs"
          >
            <span
              className="truncate text-right text-slate-600"
              title={feature.name}
            >
              {feature.name}
            </span>
            <span className="relative h-4 rounded bg-slate-100">
              <span className="absolute bottom-0 left-1/2 top-0 w-px bg-slate-400" />
              <span
                className={`absolute top-0 h-4 rounded ${
                  positive ? "bg-rose-500" : "bg-sky-500"
                }`}
                style={{
                  width,
                  ...(positive ? { left: "50%" } : { right: "50%" }),
                }}
              />
            </span>
            <span className="text-right font-mono text-slate-700">
              {positive ? "+" : ""}
              {(feature.value * 100).toFixed(2)} pp
            </span>
          </div>
        );
      })}
    </div>
  );
}

function ImageExplanation({ explanation, imageUrl }) {
  const heatmap = explanation.heatmap || [];
  const gridSize = heatmap.length;

  return (
    <div className="mt-4">
      <div className="mx-auto aspect-square w-full max-w-[320px] overflow-hidden rounded-xl border border-slate-300 bg-slate-900">
        <svg
          viewBox={`0 0 ${gridSize} ${gridSize}`}
          preserveAspectRatio="none"
          role="img"
          aria-label={`SHAP pixel-region attribution over the uploaded image for ${explanation.targetClass}`}
          className="h-full w-full"
        >
          <image
            href={imageUrl}
            x="0"
            y="0"
            width={gridSize}
            height={gridSize}
            preserveAspectRatio="none"
          />
          {heatmap.map((row, y) =>
            row.map((value, x) => (
              <rect
                key={`${x}-${y}`}
                x={x}
                y={y}
                width="1.02"
                height="1.02"
                fill={value >= 0 ? "#f43f5e" : "#0ea5e9"}
                fillOpacity={Math.abs(value) * 0.7}
              />
            )),
          )}
        </svg>
      </div>
      <div className="mt-2 flex justify-center gap-5 text-[11px] text-slate-600">
        <span>
          <b className="text-rose-600">Red</b> increases the model score
        </span>
        <span>
          <b className="text-sky-600">Blue</b> decreases it
        </span>
      </div>
    </div>
  );
}

function printShapReport() {
  const printClass = "print-shap-report";
  const cleanup = () => document.body.classList.remove(printClass);

  document.body.classList.add(printClass);
  window.addEventListener("afterprint", cleanup, { once: true });
  window.print();
}

function formatContribution(feature) {
  const value = feature.value * 100;
  return `${feature.name} (${value > 0 ? "+" : ""}${value.toFixed(2)} percentage points)`;
}

function getFeatureNarrative(features, targetClass) {
  const validFeatures = features.filter(
    (feature) => Number.isFinite(feature.value) && typeof feature.name === "string",
  );
  const increasing = validFeatures
    .filter((feature) => feature.value > 0)
    .sort((a, b) => b.value - a.value);
  const decreasing = validFeatures
    .filter((feature) => feature.value < 0)
    .sort((a, b) => a.value - b.value);

  return (
    <div className="shap-report-prose">
      <h2>SHAP explanation report</h2>
      <p>
        SHAP describes how input features influenced the model score for the
        <strong>{targetClass}</strong> class compared with the explanation
        baseline. Positive values push the score toward that class; negative
        values push it away. These values describe model behavior, not clinical
        evidence.
      </p>
      {increasing.length > 0 ? (
        <p>
          Among the {validFeatures.length} reported image-derived features, the
          strongest factors pushing the score toward {targetClass} were:{" "}
          {increasing.slice(0, 3).map(formatContribution).join("; ")}.
        </p>
      ) : (
        <p>
          None of the reported image-derived features pushed the score toward{" "}
          {targetClass}.
        </p>
      )}
      {decreasing.length > 0 ? (
        <p>
          The strongest factors pushing the score away from {targetClass} were:{" "}
          {decreasing.slice(0, 3).map(formatContribution).join("; ")}.
        </p>
      ) : (
        <p>
          No reported feature pushed the score away from {targetClass}.
        </p>
      )}
      <p>
        The list shows the strongest reported feature contributions, not
        anatomical findings. It should not be used by itself to diagnose,
        locate disease, or guide treatment.
      </p>
    </div>
  );
}

function getImageNarrative() {
  return (
    <div className="shap-report-prose">
      <h2>SHAP explanation report</h2>
      <p>
        The overlay highlights image regions that influenced the model score
        for the selected class relative to the explanation baseline. Red
        regions push the score upward, while blue regions push it downward.
        Color intensity is normalized for display and is not a calibrated
        measure of certainty.
      </p>
      <p>
        These highlighted regions are model attributions, not tumor boundaries,
        anatomical findings, or a clinical localization. The visualization
        does not establish a diagnosis and must be interpreted with the full
        study by a qualified clinician.
      </p>
    </div>
  );
}

export default function ShapExplanation({
  explanation,
  imageUrl,
  reportDetails = {},
}) {
  if (!explanation) {
    return (
      <section className="rounded-2xl border border-slate-200 bg-white p-4">
        <h4 className="text-sm font-bold text-slate-900">SHAP explanation</h4>
        <p className="mt-2 text-xs leading-relaxed text-slate-600">
          No SHAP explanation was saved for this analysis. Run a new analysis to
          generate one.
        </p>
      </section>
    );
  }

  const isImageExplanation =
    explanation.kind === "image" &&
    Array.isArray(explanation.heatmap) &&
    imageUrl;
  const isFeatureExplanation =
    explanation.kind === "features" && Array.isArray(explanation.features);
  const reportDate = new Date().toLocaleString();
  const analyzedDate = reportDetails.createdAt
    ? new Date(reportDetails.createdAt).toLocaleString()
    : "Unavailable";
  const score = Number(reportDetails.confidence);

  return (
    <section className="shap-report rounded-2xl border border-slate-200 bg-white p-4 md:p-5">
      <div className="shap-report-prose">
        <h1>SHAP Explanation Report</h1>
        <p>
          <strong>Patient ID:</strong> {reportDetails.patientId || "Unavailable"}
          {" · "}
          <strong>Scan:</strong> {reportDetails.scanType || "Unavailable"}
        </p>
        <p>
          <strong>Model output:</strong> {reportDetails.prediction || "Unavailable"}
          {Number.isFinite(score) &&
            ` · Model score: ${score.toFixed(2)}% (not calibrated probability)`}
        </p>
        <p>
          <strong>Analysis time:</strong> {analyzedDate}
          {" · "}
          <strong>Report generated:</strong> {reportDate}
        </p>
      </div>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-xs font-bold uppercase tracking-wider text-teal-700">
            SHAP explanation
          </p>
          <h4 className="mt-1 text-sm font-bold text-slate-900">
            Features influencing {explanation.targetClass}
          </h4>
          <p className="mt-1 text-xs leading-relaxed text-slate-600">
            {explanation.note}
          </p>
        </div>
        <button
          type="button"
          onClick={printShapReport}
          className="shrink-0 rounded-lg border border-teal-200 px-3 py-1.5 text-xs font-semibold text-teal-800 transition hover:bg-teal-50"
        >
          Generate SHAP report
        </button>
      </div>

      {isImageExplanation ? (
        <ImageExplanation explanation={explanation} imageUrl={imageUrl} />
      ) : isFeatureExplanation ? (
        <FeatureExplanation explanation={explanation} />
      ) : (
        <p className="mt-3 text-xs text-amber-800">
          The SHAP explanation data is incomplete or does not match this scan.
        </p>
      )}

      {isImageExplanation && getImageNarrative()}
      {isFeatureExplanation &&
        getFeatureNarrative(explanation.features, explanation.targetClass)}

      <p className="mt-3 border-t border-slate-100 pt-3 text-[11px] leading-relaxed text-slate-500">
        SHAP describes how the model used its input relative to the stated
        baseline. It is not clinical evidence, a diagnosis, or a finding
        localization. {explanation.baseline}
      </p>
    </section>
  );
}
