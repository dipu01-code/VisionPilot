"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import {
  Eye,
  Cpu,
  GitBranch,
  Shield,
  Rocket,
  Upload,
  ImageIcon,
  Film,
  Loader2,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Github,
  ExternalLink,
  Sparkles,
} from "lucide-react";
import GradientWaves from "../components/GradientWaves";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

type AnalysisResult = {
  request_id: string;
  response?: {
    validation?: {
      is_valid?: boolean;
      quality?: string;
      quality_score?: number;
    };
    detections?: Array<{
      label: string;
      confidence: number;
      bounding_box?: number[];
      area_pixels?: number;
    }>;
  };
  decision?: {
    state: string;
    reason: string;
    confidence: number;
    tool: string | null;
    needs_human_approval: boolean;
  };
  outcome?: string;
  traces?: unknown[];
  processing_time_ms?: number;
};

function toBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // Strip the data URL prefix
      resolve(result.split(",")[1]);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function StateBadge({ state }: { state: string }) {
  const map: Record<string, { icon: React.ReactNode; cls: string }> = {
    completed: {
      icon: <CheckCircle2 className="h-5 w-5" />,
      cls: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
    },
    needs_approval: {
      icon: <AlertTriangle className="h-5 w-5" />,
      cls: "bg-amber-500/15 text-amber-300 border-amber-500/30",
    },
    rejected: {
      icon: <XCircle className="h-5 w-5" />,
      cls: "bg-rose-500/15 text-rose-300 border-rose-500/30",
    },
  };
  const item = map[state] ?? {
    icon: <AlertTriangle className="h-5 w-5" />,
    cls: "bg-slate-500/15 text-slate-300 border-slate-500/30",
  };
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border px-4 py-1.5 text-sm font-medium capitalize ${item.cls}`}
    >
      {item.icon}
      {state.replace("_", " ")}
    </span>
  );
}

const FEATURES = [
  {
    icon: Eye,
    title: "OpenCV 5 Perception",
    desc: "Image & video validation, quality assessment, contour detection, color segmentation, and geometric measurements.",
  },
  {
    icon: GitBranch,
    title: "Deterministic Decision Engine",
    desc: "7 priority rules evaluate visual evidence and pick actions — same input, same decision, every time.",
  },
  {
    icon: Cpu,
    title: "Finite State Machine",
    desc: "9 states from RECEIVED → VALIDATING → ANALYZING → DECIDING → COMPLETED / NEEDS_APPROVAL / REJECTED.",
  },
  {
    icon: Shield,
    title: "Structured Traces",
    desc: "Every decision is logged with reasoning, confidence, tool calls, and full audit trails.",
  },
  {
    icon: Film,
    title: "Temporal Video Analysis",
    desc: "Multi-frame detection with timestamps, motion tracking, and annotated video output.",
  },
  {
    icon: Rocket,
    title: "Cloud-Ready on AWS",
    desc: "Docker multi-stage builds, ECS Fargate, ALB, IAM least-privilege, CloudWatch monitoring.",
  },
];

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [kind, setKind] = useState<"image" | "video">("image");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const runAnalysis = async (f: File) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const b64 = await toBase64(f);
      const endpoint =
        kind === "video" ? `${API_BASE}/video/analyze` : `${API_BASE}/analyze`;
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_data: b64, filename: f.name }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
      setResult(await res.json());
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : "Failed to analyze. Is the API running at " + API_BASE + "?"
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="relative min-h-screen">
      <GradientWaves />

      {/* Nav */}
      <header className="fixed top-0 z-50 w-full">
        <nav className="glass mx-auto flex max-w-6xl items-center justify-between rounded-b-2xl px-6 py-4">
          <div className="flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500 to-fuchsia-500">
              <Eye className="h-5 w-5 text-white" />
            </div>
            <span className="text-lg font-bold tracking-tight">
              Vision<span className="gradient-text">Pilot</span>
            </span>
          </div>
          <div className="hidden items-center gap-8 text-sm text-slate-300 md:flex">
            <a href="#features" className="transition hover:text-white">Features</a>
            <a href="#analyzer" className="transition hover:text-white">Try It</a>
            <a href="#architecture" className="transition hover:text-white">Architecture</a>
          </div>
          <a
            href="https://github.com/dipu01-code/VisionPilot"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-4 py-2 text-sm transition hover:bg-white/10"
          >
            <Github className="h-4 w-4" />
            GitHub
          </a>
        </nav>
      </header>

      {/* Hero */}
      <section className="relative flex min-h-screen items-center justify-center px-6 pt-24">
        <div className="mx-auto max-w-4xl text-center">
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            className="mb-6 inline-flex items-center gap-2 rounded-full border border-indigo-400/30 bg-indigo-500/10 px-4 py-1.5 text-sm text-indigo-200"
          >
            <Sparkles className="h-4 w-4" />
            OpenCV AI Competition 2026 · Agentic Vision
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.1 }}
            className="text-5xl font-black leading-tight tracking-tight md:text-7xl"
          >
            See. Reason. <span className="gradient-text">Act.</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.2 }}
            className="mx-auto mt-6 max-w-2xl text-lg text-slate-300"
          >
            An agentic computer vision system where OpenCV 5 analysis results
            directly drive agent decisions, tool selection, and human approval
            requests — fully auditable, end to end.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.3 }}
            className="mt-10 flex flex-wrap items-center justify-center gap-4"
          >
            <a
              href="#analyzer"
              className="rounded-xl bg-gradient-to-r from-indigo-500 to-fuchsia-500 px-8 py-4 font-semibold shadow-lg shadow-indigo-500/30 transition hover:scale-105 hover:shadow-indigo-500/50"
            >
              Try the Analyzer
            </a>
            <a
              href="https://github.com/dipu01-code/VisionPilot"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-2 rounded-xl border border-white/15 bg-white/5 px-8 py-4 font-semibold transition hover:bg-white/10"
            >
              <ExternalLink className="h-4 w-4" />
              View Code
            </a>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.7, delay: 0.5 }}
            className="mx-auto mt-16 grid max-w-2xl grid-cols-2 gap-4 md:grid-cols-4"
          >
            {[
              ["OpenCV", "5.0.0"],
              ["States", "9"],
              ["Rules", "7"],
              ["Tests", "20/20"],
            ].map(([label, value]) => (
              <div key={label} className="glass rounded-2xl p-4">
                <div className="gradient-text text-2xl font-bold">{value}</div>
                <div className="text-xs uppercase tracking-wider text-slate-400">
                  {label}
                </div>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="relative px-6 py-24">
        <div className="mx-auto max-w-6xl">
          <h2 className="text-center text-4xl font-bold">
            The Agentic Vision Loop
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-center text-slate-400">
            Perception → Decision → Action, with every step traced.
          </p>
          <div className="mt-14 grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {FEATURES.map((f, i) => (
              <motion.div
                key={f.title}
                initial={{ opacity: 0, y: 24 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-80px" }}
                transition={{ duration: 0.5, delay: i * 0.08 }}
                className="glass group rounded-2xl p-6 transition hover:border-indigo-400/30"
              >
                <div className="mb-4 inline-flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500/20 to-fuchsia-500/20 text-indigo-300 transition group-hover:scale-110">
                  <f.icon className="h-6 w-6" />
                </div>
                <h3 className="text-lg font-semibold">{f.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-slate-400">
                  {f.desc}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Analyzer */}
      <section id="analyzer" className="relative px-6 py-24">
        <div className="mx-auto max-w-5xl">
          <h2 className="text-center text-4xl font-bold">Live Analyzer</h2>
          <p className="mt-4 text-center text-slate-400">
            Drop an image or video — watch the agent decide.
          </p>

          <div className="mt-10 grid gap-6 lg:grid-cols-2">
            {/* Upload card */}
            <div className="glass-strong rounded-3xl p-6">
              {/* Tabs */}
              <div className="mb-6 flex gap-2">
                {(["image", "video"] as const).map((k) => (
                  <button
                    key={k}
                    onClick={() => {
                      setKind(k);
                      setResult(null);
                      setError(null);
                      setFile(null);
                    }}
                    className={`flex flex-1 items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-medium transition ${
                      kind === k
                        ? "bg-gradient-to-r from-indigo-500 to-fuchsia-500 text-white shadow-lg shadow-indigo-500/30"
                        : "bg-white/5 text-slate-400 hover:bg-white/10"
                    }`}
                  >
                    {k === "image" ? (
                      <ImageIcon className="h-4 w-4" />
                    ) : (
                      <Film className="h-4 w-4" />
                    )}
                    {k === "image" ? "Image" : "Video"}
                  </button>
                ))}
              </div>

              {/* Dropzone */}
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragging(false);
                  const f = e.dataTransfer.files?.[0];
                  if (f) {
                    setFile(f);
                    runAnalysis(f);
                  }
                }}
                className={`flex min-h-[280px] cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-6 transition ${
                  dragging
                    ? "border-indigo-400 bg-indigo-500/10"
                    : "border-white/15 hover:border-indigo-400/50"
                }`}
                onClick={() => {
                  const input = document.createElement("input");
                  input.type = "file";
                  input.accept =
                    kind === "image" ? "image/*" : "video/*";
                  input.onchange = () => {
                    const f = input.files?.[0];
                    if (f) {
                      setFile(f);
                      runAnalysis(f);
                    }
                  };
                  input.click();
                }}
              >
                <Upload className="mb-4 h-10 w-10 text-indigo-300" />
                {file ? (
                  <p className="text-center font-medium text-slate-200">
                    {file.name}
                    <span className="mt-1 block text-sm text-slate-400">
                      {(file.size / 1024).toFixed(1)} KB — click or drop to
                      replace
                    </span>
                  </p>
                ) : (
                  <p className="text-center text-slate-400">
                    <span className="font-medium text-slate-200">
                      Click to upload
                    </span>{" "}
                    or drag &amp; drop
                    <span className="mt-1 block text-sm">
                      {kind === "image"
                        ? "JPG, PNG, WebP"
                        : "MP4, up to 100 MB"}
                    </span>
                  </p>
                )}
              </div>

              <button
                disabled={!file || loading}
                onClick={() => file && runAnalysis(file)}
                className="mt-4 w-full rounded-xl bg-gradient-to-r from-indigo-500 to-fuchsia-500 px-6 py-4 font-semibold shadow-lg shadow-indigo-500/30 transition hover:scale-[1.02] disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:scale-100"
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <Loader2 className="h-5 w-5 animate-spin" />
                    Agent thinking…
                  </span>
                ) : (
                  "Analyze"
                )}
              </button>
            </div>

            {/* Results card */}
            <div className="glass-strong rounded-3xl p-6">
              <h3 className="mb-4 flex items-center gap-2 text-lg font-semibold">
                <Cpu className="h-5 w-5 text-indigo-300" />
                Agent Decision
              </h3>

              {!result && !error && !loading && (
                <div className="flex h-64 flex-col items-center justify-center text-slate-500">
                  <Sparkles className="mb-3 h-8 w-8" />
                  Results will appear here after analysis
                </div>
              )}

              {loading && (
                <div className="flex h-64 flex-col items-center justify-center text-slate-400">
                  <Loader2 className="mb-3 h-8 w-8 animate-spin text-indigo-400" />
                  Running perception → decision → action…
                </div>
              )}

              {error && (
                <div className="flex h-64 flex-col items-center justify-center rounded-2xl bg-rose-500/10 p-6 text-center text-rose-300">
                  <XCircle className="mb-3 h-8 w-8" />
                  <p className="text-sm">{error}</p>
                </div>
              )}

              {result && (
                <div className="space-y-4">
                  <div className="flex flex-wrap items-center gap-3">
                    {result.decision && (
                      <StateBadge state={result.decision.state} />
                    )}
                    {result.processing_time_ms != null && (
                      <span className="text-sm text-slate-400">
                        {result.processing_time_ms.toFixed(1)} ms
                      </span>
                    )}
                  </div>

                  {result.decision && (
                    <div className="rounded-2xl bg-white/5 p-4">
                      <p className="text-sm font-medium text-slate-200">
                        {result.decision.reason}
                      </p>
                      <p className="mt-2 text-xs text-slate-400">
                        Confidence:{" "}
                        <span className="text-indigo-300">
                          {(result.decision.confidence * 100).toFixed(0)}%
                        </span>
                        {result.decision.tool && (
                          <>
                            {" "}
                            · Tool:{" "}
                            <span className="text-fuchsia-300">
                              {result.decision.tool}
                            </span>
                          </>
                        )}
                      </p>
                    </div>
                  )}

                  {result.response?.validation && (
                    <div className="grid grid-cols-2 gap-3">
                      <div className="rounded-2xl bg-white/5 p-4">
                        <p className="text-xs uppercase text-slate-500">
                          Quality
                        </p>
                        <p className="mt-1 font-semibold capitalize">
                          {result.response.validation.quality}
                        </p>
                      </div>
                      <div className="rounded-2xl bg-white/5 p-4">
                        <p className="text-xs uppercase text-slate-500">
                          Score
                        </p>
                        <p className="mt-1 font-semibold">
                          {(
                            (result.response.validation.quality_score ?? 0) * 100
                          ).toFixed(0)}
                          %
                        </p>
                      </div>
                    </div>
                  )}

                  {result.response?.detections &&
                    result.response.detections.length > 0 && (
                      <div className="rounded-2xl bg-white/5 p-4">
                        <p className="mb-2 text-xs uppercase text-slate-500">
                          Detections (
                          {result.response.detections.length})
                        </p>
                        <ul className="max-h-40 space-y-1 overflow-y-auto text-sm">
                          {result.response.detections.slice(0, 8).map((d, i) => (
                            <li key={i} className="flex justify-between">
                              <span className="text-slate-300">
                                {d.label}
                              </span>
                              <span className="text-indigo-300">
                                {(d.confidence * 100).toFixed(0)}%
                              </span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                  {result.traces && result.traces.length > 0 && (
                    <div className="rounded-2xl bg-white/5 p-4">
                      <p className="text-xs uppercase text-slate-500">
                        Trace Events
                      </p>
                      <p className="mt-1 text-sm text-slate-300">
                        {result.traces.length} structured trace entries
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* Architecture */}
      <section id="architecture" className="relative px-6 py-24">
        <div className="mx-auto max-w-4xl">
          <h2 className="text-center text-4xl font-bold">
            Perception → Decision → Action
          </h2>
          <div className="mt-14 grid gap-4 md:grid-cols-3">
            {[
              {
                step: "1",
                title: "Perceive",
                desc: "OpenCV 5 validates, assesses quality, detects objects, and measures geometry.",
              },
              {
                step: "2",
                title: "Decide",
                desc: "7 deterministic rules turn visual evidence into a decision with confidence.",
              },
              {
                step: "3",
                title: "Act",
                desc: "Store results, request human approval, send alerts, or upload to S3 — all traced.",
              },
            ].map((s, i) => (
              <motion.div
                key={s.step}
                initial={{ opacity: 0, y: 24 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: i * 0.12 }}
                className="glass relative rounded-2xl p-6"
              >
                <span className="gradient-text text-4xl font-black">
                  {s.step}
                </span>
                <h3 className="mt-3 text-lg font-semibold">{s.title}</h3>
                <p className="mt-2 text-sm text-slate-400">{s.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative border-t border-white/10 px-6 py-10">
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 md:flex-row">
          <p className="text-sm text-slate-500">
            VisionPilot — OpenCV AI Competition 2026
          </p>
          <div className="flex gap-6 text-sm text-slate-400">
            <a
              href="https://github.com/dipu01-code/VisionPilot"
              target="_blank"
              rel="noreferrer"
              className="transition hover:text-white"
            >
              GitHub
            </a>
            <a
              href={`${API_BASE}/docs`}
              target="_blank"
              rel="noreferrer"
              className="transition hover:text-white"
            >
              API Docs
            </a>
          </div>
        </div>
      </footer>
    </main>
  );
}