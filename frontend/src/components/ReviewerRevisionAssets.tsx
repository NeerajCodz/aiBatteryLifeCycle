import { useEffect, useState } from "react";

interface RevisionManifest {
  title: string;
  revision: string;
  limitations: string;
  metric_definitions: Record<string, string>;
  documents: { title: string; url: string }[];
  figures: { number: number; caption: string; url: string; downloads: Record<string, string> }[];
  tables: { number: number; url: string }[];
}

const metricLabels: Record<string, string> = {
  battery_macro_mae: "Battery-macro MAE",
  sample_mae: "Sample MAE",
  fold_mean_mae: "Mean fold MAE",
};

export default function ReviewerRevisionAssets({ compact = false }: { compact?: boolean }) {
  const [manifest, setManifest] = useState<RevisionManifest | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    fetch("/research/reviewer4/manifest.json", { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("Research assets unavailable");
        return response.json();
      })
      .then(setManifest)
      .catch((error) => { if (error.name !== "AbortError") setFailed(true); });
    return () => controller.abort();
  }, []);

  if (failed) return <p className="text-amber-300 text-sm" role="status">The Reviewer 4 assets could not be loaded. Refresh to try again.</p>;
  if (!manifest) return <p className="text-gray-400 text-sm" role="status">Loading revised research assets…</p>;

  return (
    <section className="space-y-5" aria-label="Reviewer 4 research assets">
      <div className="bg-gray-900 border border-gray-700 rounded-xl p-5 space-y-4">
        <h3 className="text-lg font-semibold text-white">Reviewer 4 revision · 1 November 2026</h3>
        <p className="text-gray-300 text-sm">{manifest.title}</p>
        <p className="text-gray-400 text-sm">All 24 earlier figures are retained. Three audit figures and four reproducibility tables have been added. The primary regression results are preserved.</p>
        <div className="flex flex-wrap gap-3">
          {manifest.documents.map((document) => (
            <a key={document.url} href={document.url} target="_blank" rel="noopener noreferrer"
              className="text-sm rounded-lg px-3 py-2 bg-green-900/40 border border-green-700 text-green-200 hover:bg-green-800/50">
              {document.title} ↗
            </a>
          ))}
          <a href="/research/reviewer4/manifest.json" className="text-sm px-3 py-2 text-blue-300 underline">Sources and hashes</a>
        </div>
        <dl className="grid gap-3 md:grid-cols-3 text-sm">
          {Object.entries(manifest.metric_definitions).map(([key, meaning]) => (
            <div key={key}><dt className="font-semibold text-white">{metricLabels[key] ?? key}</dt><dd className="text-gray-400 mt-1">{meaning}</dd></div>
          ))}
        </dl>
        <p className="text-amber-200 text-sm">{manifest.limitations}</p>
      </div>
      {!compact && <>
        <div className="flex flex-wrap gap-3" aria-label="Manuscript table downloads">
          {manifest.tables.map((table) => <a key={table.number} href={table.url} download
            className="text-blue-300 border border-gray-700 rounded px-3 py-2 text-sm hover:bg-gray-800">Table {table.number} · CSV</a>)}
        </div>
        <div className="grid gap-6 lg:grid-cols-2">
          {manifest.figures.map((figure) => (
            <figure key={figure.number} className="bg-gray-900 border border-gray-700 rounded-xl overflow-hidden">
              <a href={figure.url} target="_blank" rel="noopener noreferrer" aria-label={`Open Figure ${figure.number}`}>
                <img src={figure.url} alt={`Figure ${figure.number}: ${figure.caption}`} loading="lazy" className="w-full bg-white" />
              </a>
              <figcaption className="p-4 text-sm text-gray-300"><strong className="text-white">Figure {figure.number}. </strong>{figure.caption}</figcaption>
              <div className="px-4 pb-4 flex gap-4">
                {Object.entries(figure.downloads).map(([format, url]) => <a key={format} href={url} download className="text-blue-300 text-sm underline">Download {format.toUpperCase()}</a>)}
              </div>
            </figure>
          ))}
        </div>
      </>}
    </section>
  );
}
