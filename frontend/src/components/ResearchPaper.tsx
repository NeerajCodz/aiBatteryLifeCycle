import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// Custom component overrides for proper dark-theme rendering
const mdComponents: React.ComponentProps<typeof ReactMarkdown>["components"] = {
  // ── Headings ──────────────────────────────────────────────────────────
  h1: ({ children }) => (
    <h1 className="text-3xl font-bold text-white mt-8 mb-4 pb-3 border-b border-gray-700 leading-tight">
      {children}
    </h1>
  ),
  h2: ({ children }) => (
    <h2 className="text-2xl font-bold text-green-400 mt-8 mb-3 pb-2 border-b border-gray-800">
      {children}
    </h2>
  ),
  h3: ({ children }) => (
    <h3 className="text-lg font-semibold text-blue-300 mt-6 mb-2">{children}</h3>
  ),
  h4: ({ children }) => (
    <h4 className="text-base font-semibold text-yellow-300 mt-4 mb-1">{children}</h4>
  ),
  h5: ({ children }) => (
    <h5 className="text-sm font-semibold text-gray-300 mt-3 mb-1">{children}</h5>
  ),

  // ── Paragraphs ────────────────────────────────────────────────────────
  p: ({ children }) => (
    <p className="text-gray-300 text-sm leading-7 mb-4">{children}</p>
  ),

  // ── Lists ─────────────────────────────────────────────────────────────
  ul: ({ children }) => (
    <ul className="space-y-1.5 mb-4 pl-1">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="list-decimal list-inside space-y-1.5 mb-4 text-gray-300 text-sm">{children}</ol>
  ),
  li: ({ children }) => (
    <li className="flex items-start gap-2 text-gray-300 text-sm">
      <span className="text-green-400 mt-1 shrink-0">•</span>
      <span>{children}</span>
    </li>
  ),

  // ── Code ─────────────────────────────────────────────────────────────
  code: ({ className, children, ...props }) => {
    const isBlock = className?.includes("language-");
    if (isBlock) {
      return (
        <div className="my-4 rounded-xl overflow-hidden border border-gray-700">
          {className && (
            <div className="bg-gray-800 border-b border-gray-700 px-4 py-1.5 text-xs text-gray-400 font-mono">
              {className.replace("language-", "")}
            </div>
          )}
          <pre className="bg-gray-950 p-4 overflow-x-auto">
            <code className="text-sm text-green-300 font-mono leading-relaxed" {...props}>
              {children}
            </code>
          </pre>
        </div>
      );
    }
    return (
      <code
        className="bg-gray-800 text-green-300 rounded px-1.5 py-0.5 text-xs font-mono border border-gray-700"
        {...props}
      >
        {children}
      </code>
    );
  },

  // ── Tables ────────────────────────────────────────────────────────────
  table: ({ children }) => (
    <div className="overflow-x-auto my-6 rounded-xl border border-gray-700">
      <table className="w-full text-sm">{children}</table>
    </div>
  ),
  thead: ({ children }) => (
    <thead className="bg-gray-800 border-b border-gray-600">{children}</thead>
  ),
  tbody: ({ children }) => <tbody>{children}</tbody>,
  tr: ({ children }) => (
    <tr className="border-b border-gray-800 hover:bg-gray-800/50 transition-colors">
      {children}
    </tr>
  ),
  th: ({ children }) => (
    <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-300 uppercase tracking-wider">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="px-4 py-2 text-gray-400 text-xs font-mono">{children}</td>
  ),

  // ── Blockquote ────────────────────────────────────────────────────────
  blockquote: ({ children }) => (
    <blockquote className="border-l-4 border-green-500 pl-4 my-4 bg-green-950/20 py-2 rounded-r-xl italic text-gray-300">
      {children}
    </blockquote>
  ),

  // ── Horizontal rule ───────────────────────────────────────────────────
  hr: () => <hr className="border-gray-700 my-8" />,

  // ── Links ─────────────────────────────────────────────────────────────
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-blue-400 underline hover:text-blue-300 transition-colors"
    >
      {children}
    </a>
  ),

  // ── Strong / em ──────────────────────────────────────────────────────
  strong: ({ children }) => (
    <strong className="text-white font-semibold">{children}</strong>
  ),
  em: ({ children }) => (
    <em className="text-gray-200 italic">{children}</em>
  ),
};

// ── Fallback content when file not available ──────────────────────────────
const FALLBACK = `# Comprehensive Multi-Model Approach to Lithium-Ion Battery State of Health Prediction

**Authors:** Research Team | **Dataset:** NASA PCoE Battery Dataset

---

## Abstract

This work presents a comprehensive multi-model framework for predicting State of Health (SOH) and Remaining Useful Life (RUL) of lithium-ion batteries using the NASA Prognostics Center of Excellence (PCoE) dataset.

We evaluate **12 classical machine learning models** (Ridge, Lasso, ElasticNet, KNN, SVR, Random Forest, ExtraTrees, GradientBoosting, XGBoost, LightGBM) alongside **deep learning architectures** (LSTM variants, Transformer, VAE-LSTM, iTransformer) using an intra-battery chronological split methodology that eliminates cross-battery data leakage.

---

## Key Results

| Model | R² | MAE (%) | RMSE | Within ±5% |
|---|---|---|---|---|
| ExtraTrees | **0.967** | **1.17** | 1.69 | 99.1% |
| GradientBoosting | 0.961 | 1.28 | 1.81 | 98.4% |
| Random Forest | 0.958 | 1.31 | 1.86 | 97.9% |
| XGBoost (HPO) | 0.954 | 1.35 | 1.94 | 97.6% |
| LightGBM (HPO) | 0.951 | 1.42 | 2.01 | 97.2% |

---

## Methodology

### Dataset

The NASA PCoE dataset comprises **30 Li-ion 18650 cells** (B0005–B0056, excluding B0049–B0052) tested under 5 temperature groups:

- **4°C** — cold environment testing
- **22°C** — room temperature
- **24°C** — standard lab condition
- **43°C** — elevated temperature stress
- **44°C** — high temperature degradation

### Feature Engineering

Each battery contributes **12 per-cycle engineered features** including:

- Electrochemical Impedance Spectroscopy (EIS) parameters: **Re** (electrolyte resistance) and **Rct** (charge-transfer resistance)
- Voltage statistics (mean, std, min, max, range)
- Temperature dynamics (ambient, mean cell temp)
- Capacity delta per cycle
- Cycle number and normalized age

### Split Methodology

> **Intra-battery chronological split** — for each battery, the first 70% of cycles are used for training, the remaining 30% for testing. This eliminates cross-battery data leakage that was present in the v1 group-based split.
`;

// ── Component ─────────────────────────────────────────────────────────────
export default function ResearchPaper() {
  const [markdown, setMarkdown] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [usedFallback, setUsedFallback] = useState(false);

  useEffect(() => {
    fetch("/research_paper.md")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.text();
      })
      .then((md) => {
        setMarkdown(md);
        setLoading(false);
      })
      .catch(() => {
        setMarkdown(FALLBACK);
        setUsedFallback(true);
        setLoading(false);
      });
  }, []);

  return (
    <div className="max-w-5xl mx-auto">
      {/* Header card */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-6 mb-6">
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 rounded-xl bg-purple-500/20 flex items-center justify-center shrink-0">
            <svg
              className="w-7 h-7 text-purple-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            </svg>
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-3">
              <h2 className="text-xl font-bold text-white">IEEE Research Paper</h2>
              {usedFallback && (
                <span className="text-xs px-2 py-0.5 bg-yellow-900/40 text-yellow-400 rounded-full border border-yellow-800">
                  Preview
                </span>
              )}
            </div>
            <p className="text-sm text-gray-400 mt-0.5">
              Multi-Model Battery Lifecycle Prediction using NASA PCoE Dataset
            </p>
          </div>
          <a
            href="/research_paper.md"
            download="research_paper.md"
            className="shrink-0 text-xs px-3 py-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-300 hover:text-white border border-gray-700 transition-colors"
          >
            ↓ Download
          </a>
        </div>
      </div>

      {/* Content */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-8">
        {loading ? (
          <div className="flex flex-col items-center justify-center py-20 gap-4">
            <div className="w-10 h-10 rounded-full border-2 border-purple-400 border-t-transparent animate-spin" />
            <span className="text-gray-400 text-sm">Loading research paper…</span>
          </div>
        ) : (
          <article>
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={mdComponents}
            >
              {markdown}
            </ReactMarkdown>
          </article>
        )}
      </div>
    </div>
  );
}
