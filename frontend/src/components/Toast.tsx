/**
 * Minimal toast / notification system.
 * Use the `useToast` hook to fire toasts from any component.
 */
import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { X, CheckCircle2, AlertTriangle, XCircle, Info } from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────
export type ToastType = "success" | "error" | "warning" | "info";

export interface Toast {
  id: string;
  type: ToastType;
  title: string;
  message?: string;
  duration?: number; // ms — 0 = sticky
}

interface ToastCtx {
  toasts: Toast[];
  toast: (t: Omit<Toast, "id">) => string;
  dismiss: (id: string) => void;
}

// ── Context ────────────────────────────────────────────────────────────────
const ToastContext = createContext<ToastCtx | null>(null);

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used inside <ToastProvider>");
  return ctx;
}

// ── Icon helper ────────────────────────────────────────────────────────────
function ToastIcon({ type }: { type: ToastType }) {
  const cls = "w-4 h-4 shrink-0 mt-0.5";
  if (type === "success") return <CheckCircle2 className={`${cls} text-green-400`} />;
  if (type === "error")   return <XCircle      className={`${cls} text-red-400`}   />;
  if (type === "warning") return <AlertTriangle className={`${cls} text-yellow-400`} />;
  return                         <Info          className={`${cls} text-blue-400`} />;
}

const BORDER_COLOR: Record<ToastType, string> = {
  success: "border-green-500/40",
  error:   "border-red-500/40",
  warning: "border-yellow-500/40",
  info:    "border-blue-500/40",
};

const BG_COLOR: Record<ToastType, string> = {
  success: "bg-green-500/10",
  error:   "bg-red-500/10",
  warning: "bg-yellow-500/10",
  info:    "bg-blue-500/10",
};

// ── Single toast item ─────────────────────────────────────────────────────
function ToastItem({ t, onDismiss }: { t: Toast; onDismiss: () => void }) {
  const [visible, setVisible] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    // Slight delay so CSS enter transition fires
    const raf = requestAnimationFrame(() => setVisible(true));
    const dur = t.duration ?? 4500;
    if (dur > 0) {
      timerRef.current = setTimeout(() => setVisible(false), dur);
    }
    return () => {
      cancelAnimationFrame(raf);
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [t.duration]);

  // When exit animation ends, notify parent
  const handleTransitionEnd = () => {
    if (!visible) onDismiss();
  };

  return (
    <div
      onTransitionEnd={handleTransitionEnd}
      style={{
        transition: "opacity 200ms ease, transform 200ms ease",
        opacity: visible ? 1 : 0,
        transform: visible ? "translateX(0)" : "translateX(16px)",
      }}
      className={`
        flex items-start gap-3 w-80 rounded-xl border px-4 py-3 shadow-2xl
        ${BORDER_COLOR[t.type]} ${BG_COLOR[t.type]}
        bg-gray-900 backdrop-blur-sm
      `}
    >
      <ToastIcon type={t.type} />
      <div className="flex-1 min-w-0">
        <div className="text-sm font-semibold text-white leading-tight">{t.title}</div>
        {t.message && (
          <div className="text-xs text-gray-400 mt-0.5 leading-snug">{t.message}</div>
        )}
      </div>
      <button
        onClick={() => setVisible(false)}
        className="shrink-0 text-gray-500 hover:text-white transition-colors mt-0.5"
      >
        <X className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}

// ── Container ──────────────────────────────────────────────────────────────
function ToastContainer({ toasts, dismiss }: { toasts: Toast[]; dismiss: (id: string) => void }) {
  if (!toasts.length) return null;
  return (
    <div className="fixed top-4 right-4 z-[9999] flex flex-col gap-2 pointer-events-none">
      {toasts.map((t) => (
        <div key={t.id} className="pointer-events-auto">
          <ToastItem t={t} onDismiss={() => dismiss(t.id)} />
        </div>
      ))}
    </div>
  );
}

// ── Provider ───────────────────────────────────────────────────────────────
export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const toast = useCallback((t: Omit<Toast, "id">): string => {
    const id = `t-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    setToasts((p) => [...p, { ...t, id }]);
    return id;
  }, []);

  const dismiss = useCallback((id: string) => {
    setToasts((p) => p.filter((t) => t.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={{ toasts, toast, dismiss }}>
      {children}
      <ToastContainer toasts={toasts} dismiss={dismiss} />
    </ToastContext.Provider>
  );
}
