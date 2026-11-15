import { Link } from "react-router-dom";

export default function AuthLayout({ title, subtitle, children }) {
  return (
    <div className="auth-page">
      <aside className="auth-aside">
        <Link
          to="/"
          className="flex items-center gap-2 font-semibold text-white"
        >
          <span className="brand-mark">MS</span>MediScan AI
        </Link>
        <div>
          <span className="eyebrow" style={{ color: "#9de3dd" }}>
            Multimodal imaging workspace
          </span>
          <h1>Clarity for every scan review.</h1>
          <p>
            A focused workspace for reviewing brain MRI and lung CT model
            output while keeping results organized by patient.
          </p>
        </div>
        <p className="!text-xs !text-[#b7dedd]">
          AI-assisted research tool. Not a definitive medical diagnosis.
        </p>
      </aside>
      <main className="auth-content">
        <div className="auth-card">
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
          <div>{children}</div>
        </div>
      </main>
    </div>
  );
}
