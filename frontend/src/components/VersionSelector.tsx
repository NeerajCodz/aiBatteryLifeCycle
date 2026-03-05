/**
 * VersionSelector
 *
 * Shows the active API version badge ("Version 2") with a chevron icon.
 * Clicking opens a submenu listing all available versions:
 *   - Active version: shown with a check mark
 *   - Downloaded but inactive: "Switch" button
 *   - Not yet downloaded: Download icon button → triggers server-side download
 *     then auto-switches when ready
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ChevronRight, Download, Check, RefreshCw, AlertCircle, Layers,
} from "lucide-react";
import { fetchVersions, loadVersion, VersionInfo } from "../api";

interface Props {
  activeVersion: "v1" | "v2" | "v3";
  onSwitch: (v: "v1" | "v2" | "v3") => void;
}

export default function VersionSelector({ activeVersion, onSwitch }: Props) {
  const [open, setOpen] = useState(false);
  const [versions, setVersions] = useState<VersionInfo[]>([]);
  const [busy, setBusy] = useState<string | null>(null);   // version being downloaded
  const menuRef = useRef<HTMLDivElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Load version list ────────────────────────────────────────────────────
  const refresh = useCallback(() => {
    fetchVersions()
      .then(setVersions)
      .catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // ── Poll while a download is in progress ────────────────────────────────
  useEffect(() => {
    const hasDownloading = versions.some((v) => v.status === "downloading");
    if (hasDownloading && !pollRef.current) {
      pollRef.current = setInterval(refresh, 2500);
    }
    if (!hasDownloading && pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
      setBusy(null);
    }
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [versions, refresh]);

  // ── Close on outside click ───────────────────────────────────────────────
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  // ── Actions ──────────────────────────────────────────────────────────────
  const handleDownload = async (version: string) => {
    setBusy(version);
    try {
      await loadVersion(version);
      refresh();                       // poll will take over
    } catch {
      setBusy(null);
    }
  };

  const handleSwitch = (version: string) => {
    onSwitch(version as "v1" | "v2" | "v3");
    setOpen(false);
  };

  const activeDisplay = versions.find((v) => v.id === activeVersion)?.display
    ?? `Version ${activeVersion[1]}`;

  // Versions that are NOT the active one
  const others = versions.filter((v) => v.id !== activeVersion);

  return (
    <div className="relative" ref={menuRef}>
      {/* Badge button */}
      <button
        onClick={() => setOpen((o) => !o)}
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold
          transition-colors select-none
          ${activeVersion === "v2"
            ? "bg-green-600 text-white hover:bg-green-500"
            : "bg-blue-600 text-white hover:bg-blue-500"}`}
        title="Switch model version"
      >
        <Layers className="w-3.5 h-3.5 opacity-80" />
        {activeDisplay}
        <ChevronRight
          className={`w-3.5 h-3.5 transition-transform duration-200
            ${open ? "rotate-90" : ""}`}
        />
      </button>

      {/* Dropdown */}
      {open && (
        <div
          className="absolute right-0 top-full mt-2 w-56
            bg-gray-900 border border-gray-700 rounded-xl shadow-2xl z-50
            overflow-hidden"
        >
          {/* Header */}
          <div className="px-3 py-2 border-b border-gray-700 text-xs text-gray-400 font-medium">
            Model Versions
          </div>

          {/* Active version row */}
          <div className="flex items-center justify-between px-3 py-2.5 bg-gray-800/50">
            <div>
              <span className="text-sm font-semibold text-white">{activeDisplay}</span>
              <span className="ml-2 text-xs text-green-400">active</span>
            </div>
            <Check className="w-4 h-4 text-green-400 shrink-0" />
          </div>

          {/* Other versions */}
          {others.length === 0 && (
            <div className="px-3 py-3 text-xs text-gray-500 text-center">
              No other versions available
            </div>
          )}
          {others.map((v) => {
            const isDownloading = v.status === "downloading" || busy === v.id;
            const isError = v.status === "error";
            const canSwitch = v.loaded && !isDownloading;

            return (
              <div
                key={v.id}
                className="flex items-center justify-between px-3 py-2.5
                  hover:bg-gray-800/60 transition-colors"
              >
                <div>
                  <span className="text-sm font-medium text-gray-200">{v.display}</span>
                  {v.loaded && v.model_count > 0 && (
                    <span className="ml-2 text-xs text-gray-500">
                      {v.model_count} models
                    </span>
                  )}
                  {isError && (
                    <span className="ml-2 text-xs text-red-400">error</span>
                  )}
                  {isDownloading && (
                    <span className="ml-2 text-xs text-yellow-400 animate-pulse">
                      downloading…
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-1 shrink-0">
                  {/* Switch button — visible when loaded */}
                  {canSwitch && (
                    <button
                      onClick={() => handleSwitch(v.id)}
                      className="px-2 py-0.5 rounded text-xs font-medium
                        bg-blue-700 hover:bg-blue-600 text-white transition-colors"
                      title={`Switch to ${v.display}`}
                    >
                      Load
                    </button>
                  )}

                  {/* Download button — visible when NOT loaded and NOT downloading */}
                  {!v.loaded && !isDownloading && !isError && (
                    <button
                      onClick={() => handleDownload(v.id)}
                      className="p-1.5 rounded-lg bg-gray-700 hover:bg-gray-600
                        text-gray-300 hover:text-white transition-colors"
                      title={`Download ${v.display} from HF Hub`}
                    >
                      <Download className="w-3.5 h-3.5" />
                    </button>
                  )}

                  {/* Spinner while downloading */}
                  {isDownloading && (
                    <RefreshCw className="w-3.5 h-3.5 text-yellow-400 animate-spin" />
                  )}

                  {/* Error retry */}
                  {isError && !isDownloading && (
                    <button
                      onClick={() => handleDownload(v.id)}
                      className="p-1.5 rounded-lg bg-gray-700 hover:bg-red-700
                        text-red-400 hover:text-white transition-colors"
                      title="Retry download"
                    >
                      <AlertCircle className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>
              </div>
            );
          })}

          {/* Footer hint */}
          <div className="px-3 py-2 border-t border-gray-700 text-xs text-gray-600 text-center">
            Models hosted on Hugging Face Hub
          </div>
        </div>
      )}
    </div>
  );
}
