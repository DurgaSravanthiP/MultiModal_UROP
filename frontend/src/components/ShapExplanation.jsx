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

export default function ShapExplanation({ explanation, imageUrl }) {
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

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 md:p-5">
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

      {isImageExplanation ? (
        <ImageExplanation explanation={explanation} imageUrl={imageUrl} />
      ) : isFeatureExplanation ? (
        <FeatureExplanation explanation={explanation} />
      ) : (
        <p className="mt-3 text-xs text-amber-800">
          The SHAP explanation data is incomplete or does not match this scan.
        </p>
      )}

      <p className="mt-3 border-t border-slate-100 pt-3 text-[11px] leading-relaxed text-slate-500">
        SHAP describes how the model used its input relative to the stated
        baseline. It is not clinical evidence, a diagnosis, or a finding
        localization. {explanation.baseline}
      </p>
    </section>
  );
}
