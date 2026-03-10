/**
 * VersionSelector
 *
 * Shows the active API version badge (e.g. "v3.0") with a chevron icon.
 * Clicking opens a submenu listing all available versions:
 *   - Active version: shown with a check mark
 *   - Loaded (in memory): "Switch" button
 *   - On disk but not loaded: "Load" button (instant, no download)
 *   - Not downloaded: "Download" button → triggers HF Hub download then loads
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ChevronRight, Download, Check, RefreshCw, AlertCircle, Layers, HardDrive,
} from "lucide-react";
import { fetchVersions, loadVersion, VersionInfo } from "../api";

interface Props {
  activeVersion: "v1" | "v2" | "v3";
  onSwitch: (v: "v1" | "v2" | "v3") => void;
}

export default function VersionSelector({ activeVersion, onSwitch }: Props) {
  const [open, setOpen] = useState(false);
  const [versions, setVersions] = useState<VersionInfo[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const refresh = useCallback(() => {
    fetchVersions()
      .then(setVersions)
      .catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Poll while a download is in progress
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

  // Close on outside click
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

  const handleDownloadOrLoad = async (version: string) => {
    setBusy(version);
    try {
      const res = await loadVersion(version);
      if (res.status === "ready") {
        // Loaded instantly from disk — auto-switch
        onSwitch(version as "v1" | "v2" | "v3");
        setBusy(null);
        setOpen(false);
      }
      refresh();
    } catch {
      setBusy(null);
    }
  };

  const handleSwitch = (version: string) => {
    onSwitch(version as "v1" | "v2" | "v3");
    setOpen(false);
  };

  // Auto-switch when download completes
  useEffect(() => {
    if (busy) {
      const v = versions.find((ver) => ver.id === busy);
      if (v && v.status === "ready" && v.loaded && v.model_count > 0) {
        onSwitch(busy as "v1" | "v2" | "v3");
        setBusy(null);
        setOpen(false);
      }
    }
  }, [versions, busy, onSwitch]);

  const activeDisplay = versions.find((v) => v.id === activeVersion)?.display
    ?? `v${activeVersion[1]}.0`;

  const others = versions.filter((v) => v.id !== activeVersion);

  const versionColor = (id: string) => {
    if (id === "v3") return "bg-green-600 text-white hover:bg-green-500";
    if (id === "v2") return "bg-blue-600 text-white hover:bg-blue-500";
    return "bg-gray-600 text-white hover:bg-gray-500";
  };

  return (
    <div className="relative" ref={menuRef}>
      <button
        onClick={() => setOpen((o) => !o)}
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold
          transition-colors select-none ${versionColor(activeVersion)}`}
        title="Switch model version"
      >
        <Layers className="w-3.5 h-3.5 opacity-80" />
        {activeDisplay}
        <ChevronRight
          className={`w-3.5 h-3.5 transition-transform duration-200
            ${open ? "rotate-90" : ""}`}
        />
      </button>

      {open && (
        <div
          className="absolute right-0 top-full mt-2 w-64
            bg-gray-900 border border-gray-700 rounded-xl shadow-2xl z-50
            overflow-hidden"
        >
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
            const isLoaded = v.loaded && v.model_count > 0;
            const isOnDisk = v.status === "on_disk" || (v.on_disk && !isLoaded && !isDownloading && !isError);
            const isNotDownloaded = v.status === "not_downloaded" && !v.on_disk;

            return (
              <div
                key={v.id}
                className="flex items-center justify-between px-3 py-2.5
                  hover:bg-gray-800/60 transition-colors"
              >
                <div className="flex-1 min-w-0">
                  <span className="text-sm font-medium text-gray-200">{v.display}</span>
                  {v.features && (
                    <span className="ml-2 text-xs text-gray-500">
                      {v.features}f
                    </span>
                  )}
                  {isLoaded && (
                    <span className="ml-2 text-xs text-gray-500">
                      {v.model_count} models
                    </span>
                  )}
                  {isOnDisk && (
                    <span className="ml-2 text-xs text-yellow-400">on disk</span>
                  )}
                  {isError && (
                    <span className="ml-2 text-xs text-red-400">error</span>
                  )}
                  {isDownloading && (
                    <span className="ml-2 text-xs text-yellow-400 animate-pulse">
                      downloading...
                    </span>
                  )}
                  {isNotDownloaded && (
                    <span className="ml-2 text-xs text-gray-600">not downloaded</span>
                  )}
                </div>

                <div className="flex items-center gap-1 shrink-0 ml-2">
                  {/* Switch button — loaded and ready */}
                  {isLoaded && (
                    <button
                      onClick={() => handleSwitch(v.id)}
                      className="px-2.5 py-1 rounded text-xs font-medium
                        bg-green-700 hover:bg-green-600 text-white transition-colors"
                      title={`Switch to ${v.display}`}
                    >
                      Switch
                    </button>
                  )}

                  {/* Load button — on disk but not loaded in memory */}
                  {isOnDisk && !isDownloading && (
                    <button
                      onClick={() => handleDownloadOrLoad(v.id)}
                      className="flex items-center gap-1 px-2.5 py-1 rounded text-xs font-medium
                        bg-blue-700 hover:bg-blue-600 text-white transition-colors"
                      title={`Load ${v.display} into memory`}
                    >
                      <HardDrive className="w-3 h-3" />
                      Load
                    </button>
                  )}

                  {/* Download button — not on disk */}
                  {isNotDownloaded && !isDownloading && !isError && (
                    <button
                      onClick={() => handleDownloadOrLoad(v.id)}
                      className="flex items-center gap-1 px-2.5 py-1 rounded text-xs font-medium
                        bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white transition-colors"
                      title={`Download ${v.display} from HF Hub`}
                    >
                      <Download className="w-3 h-3" />
                      Download
                    </button>
                  )}

                  {/* Spinner while downloading */}
                  {isDownloading && (
                    <RefreshCw className="w-3.5 h-3.5 text-yellow-400 animate-spin" />
                  )}

                  {/* Error retry */}
                  {isError && !isDownloading && (
                    <button
                      onClick={() => handleDownloadOrLoad(v.id)}
                      className="flex items-center gap-1 px-2.5 py-1 rounded text-xs font-medium
                        bg-gray-700 hover:bg-red-700 text-red-400 hover:text-white transition-colors"
                      title="Retry download"
                    >
                      <AlertCircle className="w-3 h-3" />
                      Retry
                    </button>
                  )}
                </div>
              </div>
            );
          })}

          <div className="px-3 py-2 border-t border-gray-700 text-xs text-gray-600 text-center">
            Models hosted on Hugging Face Hub
          </div>
        </div>
      )}
    </div>
  );
}
