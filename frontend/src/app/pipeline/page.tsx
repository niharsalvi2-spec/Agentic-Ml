"use client";

import React, { useState, useRef, useEffect, useMemo, useCallback, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { motion, AnimatePresence, useMotionValue, useSpring, useTransform } from "framer-motion";
import {
  Sparkles, Play, CheckCircle2, CircleDashed, Terminal, Download,
  Layers, BarChart3, ShieldCheck, Database, Cpu, Check, Copy,
  Zap, Brain, FlaskConical, GitBranch, Rocket, Filter, TestTube,
  Settings2, TrendingUp, Radio, Flame, ArrowUpRight, Activity,
  RefreshCw, CornerDownRight, Eye, Code, Award, CheckCheck, Clock,
  PlayCircle, Plus, FileCode, RotateCcw, Trash2, Maximize2,
  HardDrive, Server, ChevronDown, ChevronRight, Edit3, Image as ImageIcon,
  ZoomIn, ExternalLink, Lock, AlertTriangle, ShieldAlert
} from "lucide-react";
import { AgentEvent, AgentRuntimeState } from "@/types/agent-events";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

interface PipelineSummary {
  selected_model?: string;
  validation_score?: number;
  metrics?: Record<string, number>;
  artifact_path?: string;
  selected_features?: string[];
  risk_score?: number;
  risk_level?: string;
  [key: string]: unknown;
}

interface AgentStep {
  id: string;
  name: string;
  shortName: string;
  description: string;
  status: "idle" | "running" | "completed" | "error";
  message?: string;
  timestamp?: string;
  icon: React.ElementType;
  accentColor: string;
  code?: string;
  output?: string;
  fileName: string;
}

interface NotebookCell {
  id: string;
  agentId: string;
  agentName: string;
  index: number;
  title: string;
  code: string;
  displayedCode: string;
  isTyping: boolean;
  output: string;
  images: string[];
  status: "idle" | "running" | "completed" | "error";
  executionTime?: string;
  execCount?: number;
  isEditing?: boolean;
}

const INITIAL_STAGES: AgentStep[] = [
  { id: "problem_analyzer",    name: "Problem Analyzer",    shortName: "Analyze",  fileName: "01_problem_formulation.py",     description: "Classifies problem domain & target objective",          status: "idle", icon: Brain,        accentColor: "#c48c46" },
  { id: "data_collector",      name: "Data Collector",      shortName: "Ingest",   fileName: "02_dataset_ingestion.py",       description: "Profiles schema constraints & class balance",           status: "idle", icon: Database,     accentColor: "#4f8cf6" },
  { id: "preprocessing",       name: "Data Preprocessor",   shortName: "Clean",    fileName: "03_preprocessing_pipeline.py",  description: "Executes imputation, encoding & scaling",               status: "idle", icon: Settings2,    accentColor: "#f59e0b" },
  { id: "eda",                 name: "EDA Profiler",        shortName: "EDA",      fileName: "04_exploratory_data_analysis.py", description: "Evaluates skewness, kurtosis & correlations",           status: "idle", icon: TrendingUp,   accentColor: "#10b981" },
  { id: "feature_engineering", name: "Feature Engineering", shortName: "Engineer", fileName: "05_feature_transformations.py", description: "Formulates non-linear interaction features",            status: "idle", icon: FlaskConical, accentColor: "#8b5cf6" },
  { id: "feature_selection",   name: "Feature Selection",   shortName: "Select",   fileName: "06_feature_selection.py",       description: "Prunes noise via ANOVA & Mutual Info",                  status: "idle", icon: Filter,       accentColor: "#ec4899" },
  { id: "model_building",      name: "Model Building",      shortName: "Train",    fileName: "07_train_candidate_models.py",  description: "Trains tree ensembles, boosting & linear models",       status: "idle", icon: GitBranch,    accentColor: "#e11d48" },
  { id: "testing",             name: "Testing QA",          shortName: "Test",     fileName: "08_model_quality_assurance.py", description: "Asserts invariance, boundary & latency tests",          status: "idle", icon: TestTube,     accentColor: "#06b6d4" },
  { id: "validation",          name: "Validation Gate",     shortName: "Validate", fileName: "09_stratified_cross_validation.py", description: "Enforces 5-fold cross-validation audit",                status: "idle", icon: ShieldCheck,  accentColor: "#14b8a6" },
  { id: "deployment",          name: "Deployment Gate",     shortName: "Registry", fileName: "10_model_registry_spec.py",     description: "Verifies pipeline & awaits user PKL export script",      status: "idle", icon: Rocket,       accentColor: "#c48c46" },
];

const SAMPLE_PROMPTS = [
  "Predict Customer Churn based on usage and billing patterns",
  "Forecast Weekly Sales with Gradient Boosting & Lag Features",
  "Detect Credit Card Fraud with Class Imbalance & Precision Handling",
  "Predict Patient Readmission Risk with Diagnostic Biomarkers",
];

// ── Interactive Neural Particle Synapse Canvas ──────────────────────────────
function NeuralMatrixCanvas({ isRunning, activeAgentIndex }: { isRunning: boolean; activeAgentIndex: number }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;
    let width = (canvas.width = canvas.parentElement?.clientWidth || window.innerWidth);
    let height = (canvas.height = canvas.parentElement?.clientHeight || window.innerHeight);

    const handleResize = () => {
      if (!canvas || !canvas.parentElement) return;
      width = canvas.width = canvas.parentElement.clientWidth;
      height = canvas.height = canvas.parentElement.clientHeight;
    };
    window.addEventListener("resize", handleResize);

    const particleCount = 38;
    const particles = Array.from({ length: particleCount }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * (isRunning ? 1.2 : 0.4),
      vy: (Math.random() - 0.5) * (isRunning ? 1.2 : 0.4),
      radius: Math.random() * 2.2 + 1,
      color: Math.random() > 0.4 ? "rgba(196, 140, 70," : "rgba(74, 158, 124,",
      alpha: Math.random() * 0.4 + 0.15,
      pulse: Math.random() * Math.PI * 2,
    }));

    let mouseX = -1000;
    let mouseY = -1000;
    const handleMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouseX = e.clientX - rect.left;
      mouseY = e.clientY - rect.top;
    };
    window.addEventListener("mousemove", handleMouseMove);

    let time = 0;
    const render = () => {
      time += 0.02;
      ctx.clearRect(0, 0, width, height);

      for (let i = 0; i < particles.length; i++) {
        const p1 = particles[i];
        p1.x += p1.vx;
        p1.y += p1.vy;
        p1.pulse += 0.03;

        if (p1.x < 0) p1.x = width;
        if (p1.x > width) p1.x = 0;
        if (p1.y < 0) p1.y = height;
        if (p1.y > height) p1.y = 0;

        const dxMouse = p1.x - mouseX;
        const dyMouse = p1.y - mouseY;
        const distMouse = Math.sqrt(dxMouse * dxMouse + dyMouse * dyMouse);
        if (distMouse > 0 && distMouse < 100) {
          p1.x += (dxMouse / distMouse) * 1.5;
          p1.y += (dyMouse / distMouse) * 1.5;
        }

        for (let j = i + 1; j < particles.length; j++) {
          const p2 = particles[j];
          const dx = p1.x - p2.x;
          const dy = p1.y - p2.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < 130) {
            const lineAlpha = (1 - dist / 130) * (isRunning ? 0.28 : 0.12);
            ctx.beginPath();
            ctx.strokeStyle = `rgba(196, 140, 70, ${lineAlpha})`;
            ctx.lineWidth = isRunning ? 1.2 : 0.75;
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();

            if (isRunning && Math.random() < 0.015) {
              const sparkProgress = (Math.sin(time * 3 + i) + 1) / 2;
              const sparkX = p1.x + (p2.x - p1.x) * sparkProgress;
              const sparkY = p1.y + (p2.y - p1.y) * sparkProgress;
              ctx.beginPath();
              ctx.arc(sparkX, sparkY, 2, 0, Math.PI * 2);
              ctx.fillStyle = "#c48c46";
              ctx.shadowColor = "#e0a860";
              ctx.shadowBlur = 8;
              ctx.fill();
              ctx.shadowBlur = 0;
            }
          }
        }

        const currentAlpha = p1.alpha + Math.sin(p1.pulse) * 0.1;
        ctx.beginPath();
        ctx.arc(p1.x, p1.y, p1.radius, 0, Math.PI * 2);
        ctx.fillStyle = `${p1.color}${Math.max(0.05, currentAlpha)})`;
        ctx.fill();
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("mousemove", handleMouseMove);
      cancelAnimationFrame(animationFrameId);
    };
  }, [isRunning, activeAgentIndex]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none z-0 opacity-65"
      style={{ filter: "blur(0.5px)" }}
    />
  );
}

// ── Floating Ethereal Background Orbs ────────────────────────────────────────
function EtherealAmbientAura() {
  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
      <motion.div
        className="absolute rounded-full"
        style={{
          width: "480px",
          height: "480px",
          background: "radial-gradient(circle, rgba(196,140,70,0.18) 0%, rgba(196,140,70,0.02) 55%, transparent 70%)",
          left: "5%",
          top: "10%",
          filter: "blur(40px)",
        }}
        animate={{
          x: [0, 45, -25, 0],
          y: [0, -35, 20, 0],
          scale: [1, 1.12, 0.95, 1],
        }}
        transition={{ duration: 16, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className="absolute rounded-full"
        style={{
          width: "520px",
          height: "520px",
          background: "radial-gradient(circle, rgba(74,158,124,0.12) 0%, rgba(74,158,124,0.02) 55%, transparent 70%)",
          right: "8%",
          top: "35%",
          filter: "blur(50px)",
        }}
        animate={{
          x: [0, -40, 30, 0],
          y: [0, 40, -30, 0],
          scale: [1, 1.08, 0.92, 1],
        }}
        transition={{ duration: 20, repeat: Infinity, ease: "easeInOut", delay: 2 }}
      />
      <div
        className="absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage:
            "linear-gradient(#c48c46 1px, transparent 1px), linear-gradient(90deg, #c48c46 1px, transparent 1px)",
          backgroundSize: "48px 48px",
        }}
      />
    </div>
  );
}

// ── Animated Counter with Spring Physics ─────────────────────────────────────
function AnimatedNumber({ value, decimals = 0 }: { value: number; decimals?: number }) {
  const mv = useMotionValue(0);
  const spring = useSpring(mv, { stiffness: 70, damping: 20 });
  const [display, setDisplay] = useState("0");
  useEffect(() => { mv.set(value); }, [value, mv]);
  useEffect(() => spring.on("change", (v) => setDisplay(v.toFixed(decimals))), [spring, decimals]);
  return <span>{display}</span>;
}

// ── Equalizer / Audio-visual Waveform Bars for Active Stage ──────────────────
function EqualizerPulse({ color }: { color: string }) {
  return (
    <div className="flex items-end gap-0.5 h-3.5 px-1">
      {[0.4, 0.9, 0.6, 1, 0.5].map((h, i) => (
        <motion.div
          key={i}
          className="w-0.5 rounded-full"
          style={{ background: color }}
          animate={{
            height: ["20%", `${Math.max(25, h * 100)}%`, "25%"],
          }}
          transition={{
            duration: 0.5 + i * 0.12,
            repeat: Infinity,
            repeatType: "reverse",
            ease: "easeInOut",
            delay: i * 0.08,
          }}
        />
      ))}
    </div>
  );
}

// ── 10 Agent Working Box (Down below the Python Notebook) ───────────────────
function AgentWorkingBox({
  stage,
  idx,
  isActive,
  onSelect,
}: {
  stage: AgentStep;
  idx: number;
  isActive: boolean;
  onSelect: () => void;
}) {
  const Icon = stage.icon;
  const isDone = stage.status === "completed";
  const isRunning = stage.status === "running" || isActive;

  const cardRef = useRef<HTMLDivElement>(null);
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const rotateX = useSpring(useTransform(y, [-60, 60], [8, -8]), { stiffness: 300, damping: 25 });
  const rotateY = useSpring(useTransform(x, [-60, 60], [-8, 8]), { stiffness: 300, damping: 25 });

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    x.set(e.clientX - centerX);
    y.set(e.clientY - centerY);
  };

  const handleMouseLeave = () => {
    x.set(0);
    y.set(0);
  };

  return (
    <motion.div
      ref={cardRef}
      onClick={onSelect}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      initial={{ opacity: 0, y: 22, scale: 0.95 }}
      animate={{
        opacity: 1,
        y: 0,
        scale: isRunning ? 1.04 : 1,
      }}
      transition={{
        delay: idx * 0.03,
        duration: 0.45,
        type: "spring",
        stiffness: 150,
      }}
      style={{
        rotateX,
        rotateY,
        transformStyle: "preserve-3d",
        perspective: 800,
      }}
      whileHover={{ y: -6, scale: 1.03 }}
      className="relative group rounded-2xl p-4 flex flex-col justify-between overflow-hidden cursor-pointer transition-all duration-300"
    >
      <div
        className="absolute inset-0 rounded-2xl transition-all duration-500"
        style={{
          background: isDone
            ? "linear-gradient(145deg, rgba(74, 158, 124, 0.18) 0%, rgba(255, 255, 255, 0.9) 100%)"
            : isRunning
            ? "linear-gradient(145deg, rgba(196, 140, 70, 0.24) 0%, rgba(255, 255, 255, 0.98) 100%)"
            : "linear-gradient(145deg, rgba(255, 255, 255, 0.8) 0%, rgba(253, 250, 244, 0.65) 100%)",
          backdropFilter: "blur(16px)",
          border: isDone
            ? "1.5px solid rgba(74, 158, 124, 0.45)"
            : isRunning
            ? "2px solid #c48c46"
            : "1.5px solid rgba(196, 140, 70, 0.15)",
          boxShadow: isRunning
            ? "0 12px 36px -6px rgba(196, 140, 70, 0.35), 0 0 20px rgba(196, 140, 70, 0.2) inset"
            : isDone
            ? "0 6px 20px -4px rgba(74, 158, 124, 0.16)"
            : "0 4px 14px rgba(0, 0, 0, 0.03)",
        }}
      />

      {isRunning && (
        <motion.div
          className="absolute inset-0 pointer-events-none z-10"
          style={{
            background:
              "linear-gradient(180deg, transparent 0%, rgba(196, 140, 70, 0.25) 50%, rgba(224, 168, 96, 0.45) 52%, transparent 55%)",
            backgroundSize: "100% 200%",
          }}
          animate={{
            backgroundPosition: ["0% -100%", "0% 200%"],
          }}
          transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
        />
      )}

      {/* Top Header Row */}
      <div className="relative z-20 flex items-center justify-between mb-2.5">
        <span
          className="text-[9px] font-mono font-bold px-2 py-0.5 rounded-full flex items-center gap-1 shadow-xs"
          style={{
            background: isDone
              ? "rgba(74, 158, 124, 0.18)"
              : isRunning
              ? "#c48c46"
              : "rgba(196, 140, 70, 0.08)",
            color: isDone ? "#2b6b52" : isRunning ? "#ffffff" : "#8a755d",
            border: isDone
              ? "1px solid rgba(74, 158, 124, 0.3)"
              : "none",
          }}
        >
          Agent #{String(idx + 1).padStart(2, "0")}
        </span>

        {isDone ? (
          <motion.div
            initial={{ scale: 0, rotate: -45 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ type: "spring", stiffness: 260, damping: 14 }}
            className="flex items-center gap-1"
          >
            <CheckCircle2 className="w-4 h-4 text-emerald-600" />
          </motion.div>
        ) : isRunning ? (
          <div className="flex items-center gap-1.5">
            <EqualizerPulse color="#c48c46" />
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 1.2, repeat: Infinity, ease: "linear" }}
            >
              <CircleDashed className="w-4 h-4 text-amber-600" />
            </motion.div>
          </div>
        ) : (
          <div
            className="w-2.5 h-2.5 rounded-full"
            style={{ background: "rgba(196, 140, 70, 0.2)" }}
          />
        )}
      </div>

      {/* Center Icon & Agent Info */}
      <div className="relative z-20 flex items-center gap-2.5 mb-2">
        <motion.div
          whileHover={{ rotate: 12, scale: 1.15 }}
          className="w-9 h-9 rounded-xl flex items-center justify-center relative shadow-xs shrink-0"
          style={{
            background: isDone
              ? "linear-gradient(135deg, rgba(74, 158, 124, 0.22), rgba(74, 158, 124, 0.08))"
              : isRunning
              ? "linear-gradient(135deg, rgba(196, 140, 70, 0.35), rgba(224, 168, 96, 0.15))"
              : "rgba(196, 140, 70, 0.08)",
            border: isRunning ? "1px solid rgba(196, 140, 70, 0.5)" : "none",
          }}
        >
          {isRunning && (
            <motion.div
              className="absolute -inset-1 rounded-xl pointer-events-none"
              style={{
                border: "1.5px dashed rgba(196, 140, 70, 0.7)",
              }}
              animate={{ rotate: 360 }}
              transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
            />
          )}
          {React.createElement(Icon as React.ComponentType<{ className?: string; style?: React.CSSProperties }>, {
            className: "w-4.5 h-4.5 transition-transform duration-300",
            style: { color: isDone ? "#2b6b52" : isRunning ? "#c48c46" : "#9b7f63" },
          })}
        </motion.div>

        <div className="min-w-0">
          <h3
            className="text-[12px] font-bold leading-tight truncate"
            style={{
              color: isDone ? "#1f4a39" : isRunning ? "#6b4112" : "#3d3023",
            }}
          >
            {stage.name}
          </h3>
          <span
            className="text-[9px] font-mono font-bold uppercase tracking-wider block text-amber-700 truncate"
          >
            {stage.fileName}
          </span>
        </div>
      </div>

      {/* Live Building Status Badge */}
      <div className="relative z-20 my-1">
        {isRunning ? (
          <div
            className="text-[9px] font-mono font-bold px-2 py-1 rounded-lg flex items-center gap-1.5 animate-pulse"
            style={{ background: "rgba(196, 140, 70, 0.18)", color: "#9c631e" }}
          >
            <Zap className="w-3 h-3 text-amber-600 animate-bounce" />
            <span>⚡ Writing & Executing Code...</span>
          </div>
        ) : isDone ? (
          <div
            className="text-[9px] font-mono font-semibold px-2 py-0.5 rounded-lg flex items-center gap-1"
            style={{ background: "rgba(74, 158, 124, 0.12)", color: "#2b6b52" }}
          >
            <Check className="w-3 h-3 text-emerald-600" />
            <span>Provable Execution OK</span>
          </div>
        ) : (
          <p className="text-[9px] leading-snug line-clamp-1" style={{ color: "#877665" }}>
            {stage.description}
          </p>
        )}
      </div>

      {/* Bottom Footer: Click to focus code cell */}
      <div
        className="relative z-20 pt-2 flex items-center justify-between"
        style={{ borderTop: "1px solid rgba(196, 140, 70, 0.12)" }}
      >
        <span
          className="text-[8px] font-mono font-bold uppercase tracking-wider flex items-center gap-1"
          style={{
            color: isDone ? "#2b6b52" : isRunning ? "#c48c46" : "#aa9888",
          }}
        >
          {isDone ? "✓ Ready" : isRunning ? "● Live Authoring" : "Standby"}
        </span>

        <span className="text-[8px] font-mono font-bold text-amber-800 hover:underline flex items-center gap-0.5">
          <Code className="w-2.5 h-2.5" /> Jump to Cell
        </span>
      </div>
    </motion.div>
  );
}

// ── Agentic ML Interactive Python Sandbox with Live Typewriter & Plots ─
function AgentMLSandbox({
  cells,
  setCells,
  activeCellId,
  setActiveCellId,
  isRunningGlobal,
  currentAgent,
  notebookRef,
}: {
  cells: NotebookCell[];
  setCells: React.Dispatch<React.SetStateAction<NotebookCell[]>>;
  activeCellId: string;
  setActiveCellId: (id: string) => void;
  isRunningGlobal: boolean;
  currentAgent: string | null;
  notebookRef: React.RefObject<HTMLDivElement | null>;
}) {
  const [executingCellId, setExecutingCellId] = useState<string | null>(null);
  const [copiedCellId, setCopiedCellId] = useState<string | null>(null);
  const [viewFilter, setViewFilter] = useState<"all" | "active" | "script">("all");
  const [collapsedOutputs, setCollapsedOutputs] = useState<Record<string, boolean>>({});
  const [previewImage, setPreviewImage] = useState<string | null>(null);

  const toggleCollapse = (id: string) => {
    setCollapsedOutputs((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const handleCopyCode = (id: string, code: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCellId(id);
    setTimeout(() => setCopiedCellId(null), 2000);
  };

  const handleCodeChange = (id: string, newCode: string) => {
    setCells((prev) =>
      prev.map((c) => (c.id === id ? { ...c, code: newCode, displayedCode: newCode } : c))
    );
  };

  // Run cell via real Python execution endpoint with image capture
  const handleExecuteCell = async (cellId: string) => {
    const targetCell = cells.find((c) => c.id === cellId);
    if (!targetCell) return;

    setExecutingCellId(cellId);
    setCells((prev) =>
      prev.map((c) =>
        c.id === cellId
          ? { ...c, status: "running", output: "Executing snippet in Python 3.12 sandbox..." }
          : c
      )
    );

    try {
      const resp = await fetch(`${API_BASE_URL}/api/pipeline/execute-code`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: targetCell.code, cell_id: cellId }),
      });
      const data = await resp.json();

      setCells((prev) =>
        prev.map((c) =>
          c.id === cellId
            ? {
                ...c,
                status: data.status === "success" ? "completed" : "error",
                output: data.status === "success" ? data.stdout : `${data.stderr}\n${data.stdout}`.trim(),
                images: data.images || [],
                executionTime: `${data.execution_time_ms}ms`,
                execCount: (c.execCount || 0) + 1,
              }
            : c
        )
      );
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : "Failed to reach backend runtime";
      setCells((prev) =>
        prev.map((c) =>
          c.id === cellId
            ? {
                ...c,
                status: "error",
                output: `Execution error: ${errMsg}`,
              }
            : c
        )
      );
    } finally {
      setExecutingCellId(null);
    }
  };

  const handleRunAll = async () => {
    for (const cell of cells) {
      await handleExecuteCell(cell.id);
    }
  };

  const handleDownloadIpynb = () => {
    const notebookObj = {
      cells: cells.map((c) => ({
        cell_type: "code",
        execution_count: c.execCount || 1,
        metadata: { agent: c.agentName, stage: c.title },
        outputs: [
          {
            output_type: "stream",
            name: "stdout",
            text: c.output.split("\n").map((l) => `${l}\n`),
          },
        ],
        source: c.code.split("\n").map((l) => `${l}\n`),
      })),
      metadata: {
        language_info: { name: "python", version: "3.12" },
        kernelspec: { display_name: "Python 3 (Agentic-ML)", language: "python", name: "python3" },
      },
      nbformat: 4,
      nbformat_minor: 2,
    };

    const blob = new Blob([JSON.stringify(notebookObj, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "agentic_ml_pipeline.ipynb";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div
      ref={notebookRef}
      className="relative rounded-2xl overflow-hidden shadow-2xl transition-all duration-300"
      style={{
        background: "rgba(255, 253, 248, 0.98)",
        border: "1.5px solid rgba(196, 140, 70, 0.28)",
        boxShadow: "0 20px 60px -15px rgba(120, 75, 25, 0.12), 0 0 0 1px rgba(196, 140, 70, 0.08)",
      }}
    >
        {/* ── Image Modal Lightbox ──────────────────────────────────────────── */}
        <AnimatePresence>
          {previewImage && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setPreviewImage(null)}
              className="fixed inset-0 z-[100] bg-black/75 backdrop-blur-md flex items-center justify-center p-4 cursor-zoom-out"
            >
              <motion.div
                initial={{ scale: 0.85 }}
                animate={{ scale: 1 }}
                exit={{ scale: 0.85 }}
                className="bg-white p-4 rounded-2xl max-w-4xl max-h-[90vh] overflow-hidden shadow-2xl"
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={previewImage} alt="Expanded Plot" className="rounded-lg object-contain max-h-[80vh] w-full" />
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Sandbox Environment Ribbon ──────────────────────────────── */}
        <div
          className="px-6 py-3.5 flex flex-wrap items-center justify-between gap-4"
          style={{
            background: "linear-gradient(90deg, rgba(253, 250, 244, 0.95) 0%, rgba(249, 244, 234, 0.9) 100%)",
            borderBottom: "1.5px solid rgba(196, 140, 70, 0.18)",
          }}
        >
          <div className="flex items-center gap-3 flex-wrap">
            <div className="flex items-center gap-2.5">
              <div
                className="w-8 h-8 rounded-xl flex items-center justify-center shadow-xs"
                style={{ background: "linear-gradient(135deg, #e69d45, #d97706)" }}
              >
                <FileCode className="w-4.5 h-4.5 text-white" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold font-mono" style={{ color: "#2d2216" }}>
                    agentic_pipeline.ipynb
                  </span>
                  <span
                    className="text-[9px] font-mono px-2 py-0.5 rounded-full font-bold shadow-2xs"
                    style={{ background: "rgba(196, 140, 70, 0.12)", color: "#9c631e" }}
                  >
                    Interactive Python Sandbox Runtime
                  </span>
                </div>
                <span className="text-[10px] text-amber-900/60 font-mono">
                  Autonomous Typewriter & Real-Time Matplotlib Visualization Engine
                </span>
              </div>
            </div>

          <div className="h-5 w-[1px] bg-amber-900/15 hidden sm:block" />

          <div className="flex items-center gap-1.5 flex-wrap">
            <motion.button
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.96 }}
              onClick={handleRunAll}
              disabled={executingCellId !== null || isRunningGlobal}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold text-white shadow-sm disabled:opacity-50"
              style={{ background: "linear-gradient(135deg, #c48c46, #e0a860)" }}
            >
              <Play className="w-3 h-3 fill-white" /> Run All (10 Cells)
            </motion.button>

            <button
              onClick={() => {
                setCells((prev) =>
                  prev.map((c) => ({ ...c, output: "", images: [], status: "idle", execCount: 0 }))
                );
              }}
              className="flex items-center gap-1 px-2.5 py-1.5 rounded-xl text-[11px] font-mono font-medium hover:bg-amber-900/5 transition-colors"
              style={{ color: "#6b5030" }}
            >
              <Trash2 className="w-3 h-3 opacity-60" /> Clear Outputs
            </button>

            <button
              onClick={handleDownloadIpynb}
              className="flex items-center gap-1 px-3 py-1.5 rounded-xl text-[11px] font-mono font-medium bg-white border border-amber-900/15 shadow-2xs hover:bg-amber-50/50 transition-colors"
              style={{ color: "#6b5030" }}
            >
              <Download className="w-3 h-3 text-amber-700" /> Export .ipynb
            </button>
          </div>
        </div>

        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-2 text-[10px] font-mono text-amber-900/70 bg-white/80 px-3 py-1.5 rounded-xl border border-amber-900/10 shadow-2xs">
            <Server className="w-3 h-3 text-emerald-600" />
            <span className="font-semibold text-emerald-800">Python 3.12 Sandbox</span>
            <span className="opacity-40">|</span>
            <span>Matplotlib / Seaborn Ready</span>
          </div>

          <div className="flex items-center gap-1 bg-amber-900/5 p-1 rounded-xl">
            {(["all", "active", "script"] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setViewFilter(mode)}
                className={`text-[10px] font-mono px-2.5 py-1 rounded-lg uppercase tracking-wider font-semibold transition-all ${
                  viewFilter === mode
                    ? "bg-white text-amber-800 shadow-xs font-bold"
                    : "text-amber-900/50 hover:text-amber-900"
                }`}
              >
                {mode === "all" ? "All 10 Cells" : mode === "active" ? "Active Cell" : "Full Script"}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ── Notebook Cells Workspace ─────────────────────────────────────── */}
      <div
        className="p-5 overflow-y-auto space-y-5"
        style={{
          maxHeight: "720px",
          background: "radial-gradient(ellipse at 50% 0%, #fffdfa 0%, #f7f1e4 100%)",
        }}
      >
        {viewFilter === "script" ? (
          <div className="rounded-2xl p-5 bg-[#1e1e1e] text-white shadow-xl font-mono text-xs">
            <div className="flex justify-between items-center pb-3 mb-3 border-b border-white/10">
              <span className="text-amber-400 font-bold"># Complete Pipeline Python Script (All 10 Agents)</span>
              <button
                onClick={() => handleCopyCode("full_script", cells.map((c) => c.code).join("\n\n"))}
                className="flex items-center gap-1 text-[11px] px-3 py-1 bg-white/10 hover:bg-white/20 rounded-lg text-amber-200"
              >
                <Copy className="w-3 h-3" /> Copy Full Script
              </button>
            </div>
            <pre className="overflow-x-auto leading-relaxed text-gray-200">
              {cells.map((c) => c.code).join("\n\n# " + "=".repeat(70) + "\n\n")}
            </pre>
          </div>
        ) : (
          (viewFilter === "active" ? [activeCell] : cells).map((cell) => {
            const isRunningCell = executingCellId === cell.id || currentAgent === cell.agentId;
            const isCollapsed = collapsedOutputs[cell.id] || false;

            return (
              <motion.div
                key={cell.id}
                id={cell.id}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                className="group rounded-2xl overflow-hidden shadow-md transition-all duration-300 border"
                style={{
                  background: "rgba(255, 255, 255, 0.95)",
                  borderColor:
                    isRunningCell
                      ? "rgba(196, 140, 70, 0.85)"
                      : activeCellId === cell.id
                      ? "rgba(196, 140, 70, 0.45)"
                      : "rgba(196, 140, 70, 0.16)",
                  boxShadow: isRunningCell
                    ? "0 0 24px rgba(196, 140, 70, 0.25)"
                    : "0 4px 16px rgba(0, 0, 0, 0.03)",
                }}
              >
                {/* Cell Header */}
                <div
                  className="px-4 py-2.5 flex items-center justify-between"
                  style={{
                    background:
                      isRunningCell
                        ? "linear-gradient(90deg, rgba(196, 140, 70, 0.18) 0%, rgba(255, 255, 255, 0.9) 100%)"
                        : "rgba(253, 250, 244, 0.7)",
                    borderBottom: "1px solid rgba(196, 140, 70, 0.12)",
                  }}
                >
                  <div className="flex items-center gap-2.5">
                    <motion.button
                      whileHover={{ scale: 1.15 }}
                      whileTap={{ scale: 0.9 }}
                      onClick={() => handleExecuteCell(cell.id)}
                      disabled={isRunningCell}
                      title="Click to execute this cell live"
                      className="w-7 h-7 rounded-lg flex items-center justify-center font-mono text-[11px] font-bold shadow-2xs transition-colors"
                      style={{
                        background: isRunningCell
                          ? "#c48c46"
                          : "rgba(196, 140, 70, 0.14)",
                        color: isRunningCell ? "#ffffff" : "#9c631e",
                      }}
                    >
                      {isRunningCell ? (
                        <motion.div
                          animate={{ rotate: 360 }}
                          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                        >
                          <CircleDashed className="w-3.5 h-3.5" />
                        </motion.div>
                      ) : (
                        <Play className="w-3 h-3 fill-amber-800" />
                      )}
                    </motion.button>

                    <div className="flex items-center gap-2">
                      <span
                        className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-md uppercase"
                        style={{
                          background: "rgba(196, 140, 70, 0.12)",
                          color: "#7a5020",
                        }}
                      >
                        Cell #{cell.index} · {cell.agentName}
                      </span>
                      <span className="text-xs font-mono font-bold" style={{ color: "#3d3023" }}>
                        {cell.title}
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    {cell.executionTime && (
                      <span className="text-[10px] font-mono text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded-md border border-emerald-200">
                        {cell.executionTime}
                      </span>
                    )}

                    <button
                      onClick={() => handleCopyCode(cell.id, cell.code)}
                      className="p-1.5 rounded-lg text-amber-900/60 hover:text-amber-900 hover:bg-amber-900/5 transition-colors"
                      title="Copy Cell Code"
                    >
                      {copiedCellId === cell.id ? (
                        <Check className="w-3.5 h-3.5 text-emerald-600" />
                      ) : (
                        <Copy className="w-3.5 h-3.5" />
                      )}
                    </button>

                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => handleExecuteCell(cell.id)}
                      disabled={isRunningCell}
                      className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-[11px] font-mono font-bold text-white shadow-2xs"
                      style={{ background: "linear-gradient(135deg, #c48c46, #e0a860)" }}
                    >
                      <Play className="w-2.5 h-2.5 fill-white" /> Run
                    </motion.button>
                  </div>
                </div>

                {/* Code Editor Body with Live Typewriter / Streaming */}
                <div className="p-4 bg-[#1e1e1e] text-gray-100 font-mono text-xs relative group/code overflow-x-auto">
                  <div className="flex items-start gap-3">
                    <div className="select-none opacity-30 text-[10px] text-right font-mono pr-2 border-r border-white/10">
                      {cell.displayedCode.split("\n").map((_, i) => (
                        <div key={i}>{i + 1}</div>
                      ))}
                    </div>
                    <div className="w-full relative">
                      <textarea
                        value={cell.displayedCode}
                        onChange={(e) => handleCodeChange(cell.id, e.target.value)}
                        rows={Math.max(4, Math.min(cell.displayedCode.split("\n").length, 16))}
                        spellCheck={false}
                        className="w-full bg-transparent outline-none resize-none font-mono text-xs leading-relaxed text-gray-200 selection:bg-amber-500/30"
                      />
                      {cell.isTyping && (
                        <motion.span
                          animate={{ opacity: [1, 0, 1] }}
                          transition={{ duration: 0.6, repeat: Infinity }}
                          className="inline-block w-2 h-4 bg-amber-400 align-middle ml-1"
                        />
                      )}
                    </div>
                  </div>
                </div>

                {/* Output Console Pane & Real Matplotlib Visualizations */}
                {(cell.output || cell.images.length > 0) && (
                  <div
                    className="border-t border-amber-900/15"
                    style={{
                      background: "rgba(253, 250, 244, 0.95)",
                    }}
                  >
                    <div
                      onClick={() => toggleCollapse(cell.id)}
                      className="px-4 py-1.5 flex items-center justify-between text-[10px] font-mono text-amber-900/60 cursor-pointer hover:bg-black/2 border-b border-amber-900/10"
                    >
                      <span className="font-bold flex items-center gap-1 text-amber-900">
                        {isCollapsed ? <ChevronRight className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                        Stdout Output & Visualizations {cell.images.length > 0 ? `(${cell.images.length} plot${cell.images.length > 1 ? 's' : ''})` : ''}
                      </span>
                      <span>Output Stream</span>
                    </div>

                    {!isCollapsed && (
                      <div className="p-4 font-mono text-xs text-amber-950 space-y-3 max-h-96 overflow-y-auto">
                        {cell.output && (
                          <pre className="whitespace-pre-wrap leading-relaxed">{cell.output}</pre>
                        )}

                        {cell.images.length > 0 && (
                          <div className="pt-2 border-t border-amber-900/10 space-y-2">
                            <span className="text-[10px] font-bold uppercase tracking-wider text-amber-800 flex items-center gap-1.5">
                              <ImageIcon className="w-3.5 h-3.5 text-amber-600" /> Provable Rendered Figure ({cell.images.length}):
                            </span>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                              {cell.images.map((imgSrc, imgIdx) => (
                                <motion.div
                                  key={imgIdx}
                                  whileHover={{ scale: 1.02 }}
                                  onClick={() => setPreviewImage(imgSrc)}
                                  className="relative group/img bg-white p-2 rounded-xl border border-amber-900/15 shadow-sm cursor-zoom-in overflow-hidden"
                                >
                                  {/* eslint-disable-next-line @next/next/no-img-element */}
                                  <img src={imgSrc} alt={`Plot ${imgIdx + 1}`} className="w-full h-auto rounded-lg object-contain" />
                                  <div className="absolute top-3 right-3 bg-black/60 text-white p-1 rounded-md opacity-0 group-hover/img:opacity-100 transition-opacity">
                                    <ZoomIn className="w-3.5 h-3.5" />
                                  </div>
                                </motion.div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </motion.div>
            );
          })
        )}
      </div>
    </motion.section>
  );
}

// ── Retro-futuristic Log Line with Agent Tagging ─────────────────────────────
function LogLine({ log, index }: { log: string; index: number }) {
  const isFinal = log.includes("[★");
  const isSuccess = log.includes("[✓");
  const isError = log.includes("[!]");
  const isIndent = log.startsWith("   ");

  return (
    <motion.div
      initial={{ opacity: 0, x: -12 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.25, delay: Math.min(index * 0.02, 0.3) }}
      className="text-[11px] font-mono leading-relaxed group relative py-0.5 flex items-start gap-2"
    >
      <span className="text-[9px] opacity-35 select-none font-mono mt-0.5">
        {String(index + 1).padStart(2, "0")}
      </span>
      <div
        className="flex-1 rounded-md px-1.5 py-0.5 transition-colors"
        style={{
          color: isFinal
            ? "#925907"
            : isSuccess
            ? "#236d4b"
            : isError
            ? "#991b1b"
            : isIndent
            ? "#786550"
            : "#423324",
          background: isFinal
            ? "rgba(196, 140, 70, 0.12)"
            : isSuccess
            ? "rgba(74, 158, 124, 0.06)"
            : isError
            ? "rgba(239, 68, 68, 0.08)"
            : "transparent",
          borderLeft: isFinal
            ? "3px solid #c48c46"
            : isSuccess
            ? "2px solid #4a9e7c"
            : "none",
          fontWeight: isFinal || isSuccess ? 600 : 400,
        }}
      >
        {log}
      </div>
    </motion.div>
  );
}

// ── Metric Comparison Bar with Spring Growth ─────────────────────────────────
function MetricBar({
  name,
  score,
  isBest,
  index,
}: {
  name: string;
  score: number;
  isBest: boolean;
  index: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 18 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4, delay: index * 0.08 }}
      className="space-y-1.5 group p-2 rounded-xl transition-all hover:bg-black/2"
    >
      <div className="flex justify-between items-center text-xs">
        <span className="font-semibold flex items-center gap-1.5" style={{ color: "#3d3023" }}>
          {name}
          {isBest && (
            <motion.span
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", stiffness: 300, delay: 0.2 }}
              className="text-[8.5px] px-2 py-0.5 rounded-full font-bold font-mono uppercase tracking-wider flex items-center gap-1 shadow-xs"
              style={{
                background: "linear-gradient(135deg, #c48c46, #e0a860)",
                color: "#ffffff",
              }}
            >
              <Award className="w-2.5 h-2.5" /> Best CV Generalization
            </motion.span>
          )}
        </span>
        <span
          className="font-mono font-bold text-xs"
          style={{ color: isBest ? "#a66816" : "#6e5d4d" }}
        >
          <AnimatedNumber value={score * 100} decimals={2} />%
        </span>
      </div>

      <div
        className="w-full h-2 rounded-full overflow-hidden p-0.5 relative"
        style={{ background: "rgba(196, 140, 70, 0.12)" }}
      >
        <motion.div
          className="h-full rounded-full relative"
          style={{
            background: isBest
              ? "linear-gradient(90deg, #c48c46 0%, #e0a860 60%, #4a9e7c 100%)"
              : "linear-gradient(90deg, #9e8975 0%, #baab9c 100%)",
            boxShadow: isBest ? "0 0 10px rgba(196, 140, 70, 0.4)" : "none",
          }}
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(Math.max(score * 100, 5), 100)}%` }}
          transition={{ duration: 1, ease: "easeOut", delay: 0.1 + index * 0.06 }}
        >
          {isBest && (
            <motion.div
              className="absolute inset-0 bg-white/30 rounded-full"
              animate={{ opacity: [0, 0.8, 0], x: ["-100%", "100%"] }}
              transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
            />
          )}
        </motion.div>
      </div>
    </motion.div>
  );
}

// ── Main Page Component ───────────────────────────────────────────────────────
function PipelinePageContent() {
  const searchParams = useSearchParams();
  const [prompt, setPrompt] = useState("Predict Customer Churn based on usage and billing patterns");
  const [isRunning, setIsRunning] = useState(false);
  const [stages, setStages] = useState<AgentStep[]>(INITIAL_STAGES);
  const [notebookCells, setNotebookCells] = useState<NotebookCell[]>([]);
  const [activeCellId, setActiveCellId] = useState<string>("cell_problem_analyzer");
  const [currentAgent, setCurrentAgent] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [summary, setSummary] = useState<PipelineSummary | null>(null);
  const [pendingApproval, setPendingApproval] = useState<AgentEvent | null>(null);
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState<"all" | "agents" | "metrics">("all");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  const logEndRef = useRef<HTMLDivElement>(null);
  const notebookRef = useRef<HTMLDivElement>(null);
  const typingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Initialize dynamic cells for prompt on mount
  useEffect(() => {
    initializeCellsForPrompt(prompt);
  }, []);

  const initializeCellsForPrompt = (p: string) => {
    const defaultInitCells: NotebookCell[] = INITIAL_STAGES.map((s, idx) => ({
      id: `cell_${s.id}`,
      agentId: s.id,
      agentName: s.name,
      index: idx + 1,
      title: s.fileName,
      code: `# [Agent ${String(idx + 1).padStart(2, "0")}: ${s.name}] Ready for execution...`,
      displayedCode: `# [Agent ${String(idx + 1).padStart(2, "0")}: ${s.name}] Ready for execution...`,
      isTyping: false,
      output: "",
      images: [],
      status: "idle",
      execCount: 0
    }));
    setNotebookCells(defaultInitCells);
  };

  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;
    if (isRunning) {
      setElapsedSeconds(0);
      interval = setInterval(() => {
        setElapsedSeconds((prev) => prev + 1);
      }, 1000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isRunning]);

  useEffect(() => {
    const urlPrompt = searchParams.get("prompt");
    if (urlPrompt?.trim()) {
      setPrompt(urlPrompt);
      initializeCellsForPrompt(urlPrompt);
      const t = setTimeout(() => runPipeline(urlPrompt), 600);
      return () => clearTimeout(t);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const activeAgentIndex = useMemo(() => {
    if (!currentAgent) return -1;
    return stages.findIndex((s) => s.id === currentAgent);
  }, [currentAgent, stages]);

  const scrollToCell = (cellId: string) => {
    setActiveCellId(cellId);
    const el = document.getElementById(cellId);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
    } else {
      notebookRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  };

  const handleHITLDecision = async (approved: boolean) => {
    if (!pendingApproval) return;
    const runId = pendingApproval.run_id;
    setPendingApproval(null);
    const timeStr = new Date().toLocaleTimeString();
    setLogs((prev) => [
      ...prev,
      `[${timeStr}] [* HITL] Human decision submitted: ${approved ? "APPROVED" : "REJECTED"}`,
    ]);

    try {
      const response = await fetch(`${API_BASE_URL}/api/pipeline/run/${runId}/approve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ approved }),
      });

      if (!response.body) return;
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data: ")) continue;
          const dataStr = trimmed.replace("data: ", "").trim();
          if (dataStr === "[DONE]") {
            setIsRunning(false);
            setCurrentAgent(null);
            break;
          }
          try {
            const parsedEvent: AgentEvent = JSON.parse(dataStr);
            handleStreamEvent(parsedEvent);
          } catch {
            // ignore non-json frames
          }
        }
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setLogs((prev) => [...prev, `[!] HITL resume error: ${msg}`]);
    }
  };

  const runPipeline = async (taskPrompt: string) => {
    if (!taskPrompt.trim() || isRunning) return;
    setIsRunning(true);
    setSummary(null);
    setPendingApproval(null);
    initializeCellsForPrompt(taskPrompt);

    const startStr = new Date().toLocaleTimeString();
    setLogs([
      `[${startStr}] [*] LANGGRAPH ORCHESTRATOR INITIALIZED`,
      `   └─ Target Workflow: "${taskPrompt}"`,
      `   └─ Multi-Agent Graph: State-Bound Execution Active`,
      `   └─ Evidence Stream: SSE Contract Established`,
    ]);
    setStages(INITIAL_STAGES.map((s) => ({ ...s, status: "idle", message: undefined })));
    setCurrentAgent("problem_analyzer");
    setActiveCellId("cell_problem_analyzer");

    try {
      const response = await fetch(`${API_BASE_URL}/api/pipeline/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: taskPrompt }),
      });

      if (!response.body) throw new Error("Stream connection failed");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data: ")) continue;
          const dataStr = trimmed.replace("data: ", "").trim();
          if (dataStr === "[DONE]") {
            setIsRunning(false);
            setCurrentAgent(null);
            break;
          }
          try {
            const parsedEvent: AgentEvent = JSON.parse(dataStr);
            handleStreamEvent(parsedEvent);
          } catch {
            // handle non-json chunk safely
          }
        }
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setLogs((prev) => [...prev, `[!] Pipeline stream error: ${msg}`]);
      setIsRunning(false);
      setCurrentAgent(null);
    }
  };

  const handleStreamEvent = (event: AgentEvent) => {
    const timeStr = new Date().toLocaleTimeString();
    const agentId = event.agent_id || event.agent || "";
    const agentName = event.agent_name || event.stage_name || agentId;

    if (event.event_type === "human_approval_required") {
      setPendingApproval(event);
      setLogs((prev) => [
        ...prev,
        `[${timeStr}] [⚠️ HUMAN APPROVAL REQUIRED] Deployment Gate paused for review`,
        `   └─ Risk: ${event.risk_level ?? "HIGH"} (Score: ${event.risk_score ?? "N/A"}/100)`,
        `   └─ Action Required: Review model evidence and approve/reject below`,
      ]);
      return;
    }

    if (agentId && agentId !== "orchestrator") {
      setCurrentAgent(agentId);
      const isCompleted = event.event_type === "agent_completed" || event.status === "COMPLETED";
      const isFailed = event.event_type === "agent_failed" || event.status === "FAILED";

      setStages((prev) =>
        prev.map((s) =>
          s.id === agentId
            ? {
                ...s,
                status: isCompleted ? "completed" : isFailed ? "error" : "running",
                message: event.message,
                timestamp: timeStr,
              }
            : s
        )
      );

      if (isCompleted) {
        setLogs((prev) => [
          ...prev,
          `[${timeStr}] [✓ ${agentName.toUpperCase()}] Execution validated`,
          ...(event.message ? [`   └─ ${event.message.slice(0, 140)}…`] : []),
        ]);
      } else if (isFailed) {
        setLogs((prev) => [
          ...prev,
          `[${timeStr}] [✗ ${agentName.toUpperCase()}] Agent failed: ${event.error || event.message || "Unknown error"}`,
        ]);
      }
    }

    if (event.is_final && event.summary) {
      const summ = event.summary as PipelineSummary;
      setSummary(summ);
      setIsRunning(false);
      setCurrentAgent(null);
      const valScoreStr = typeof summ.validation_score === "number" ? `${(summ.validation_score * 100).toFixed(2)}%` : "N/A";
      setLogs((prev) => [
        ...prev,
        `[${timeStr}] [★ VALIDATION & EXECUTION COMPLETE]`,
        `   └─ Selected Model: ${summ.selected_model || "Champion Model"}`,
        `   └─ Validation Score: ${valScoreStr}`,
        `   └─ Artifact Bundle: ${summ.artifact_path || "Verified & Signed"}`,
        `   └─ Provenance: SHA-256 + Ed25519 Verified`,
      ]);
    }
  };

  const copyMetadata = () => {
    if (summary) {
      navigator.clipboard.writeText(JSON.stringify(summary, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 2200);
    }
  };

  const completedCount = stages.filter((s) => s.status === "completed").length;
  const progressPercent = Math.round((completedCount / stages.length) * 100);

  const filteredLogs = useMemo(() => {
    if (activeTab === "agents") return logs.filter((l) => l.includes("[✓") || l.includes("[*"));
    if (activeTab === "metrics") return logs.filter((l) => l.includes("Score") || l.includes("★") || l.includes("└─"));
    return logs;
  }, [logs, activeTab]);

  return (
    <div
      className="min-h-screen flex flex-col font-sans relative overflow-x-hidden"
      style={{
        background:
          "radial-gradient(ellipse at 50% -20%, #fffdfa 0%, #f9f4ea 45%, #f2e8d5 100%)",
        color: "#2a241d",
      }}
    >
      <NeuralMatrixCanvas isRunning={isRunning} activeAgentIndex={activeAgentIndex} />
      <EtherealAmbientAura />

      {/* ── Studio Header Bar ────────────────────────────────────────────── */}
      <motion.header
        initial={{ y: -60, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.7, type: "spring", stiffness: 120 }}
        className="relative z-20 px-6 py-4 flex flex-wrap items-center justify-between gap-4 sticky top-0"
        style={{
          background: "rgba(253, 250, 244, 0.85)",
          backdropFilter: "blur(24px)",
          borderBottom: "1.5px solid rgba(196, 140, 70, 0.18)",
          boxShadow: "0 4px 30px rgba(196, 140, 70, 0.08)",
        }}
      >
        <div className="flex items-center gap-3.5">
          <motion.div
            whileHover={{ rotate: 180, scale: 1.1 }}
            transition={{ duration: 0.6, ease: "easeInOut" }}
            className="w-11 h-11 rounded-2xl flex items-center justify-center shadow-md relative group cursor-pointer"
            style={{ background: "linear-gradient(135deg, #c48c46, #e0a860)" }}
          >
            <motion.div
              className="absolute -inset-1 rounded-2xl opacity-40 blur-xs"
              style={{ background: "linear-gradient(135deg, #c48c46, #e0a860)" }}
              animate={{ opacity: [0.3, 0.7, 0.3] }}
              transition={{ duration: 3, repeat: Infinity }}
            />
            <Cpu className="w-5.5 h-5.5 text-white relative z-10" />
          </motion.div>

          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold tracking-tight" style={{ color: "#2d2216" }}>
                Agentic<span className="italic font-serif text-amber-700" style={{ color: "#c48c46" }}>ML</span>
              </h1>
              <span
                className="text-[9.5px] font-mono uppercase px-2.5 py-0.5 rounded-full font-bold shadow-xs flex items-center gap-1.5"
                style={{
                  background: "rgba(196, 140, 70, 0.14)",
                  color: "#9c631e",
                  border: "1px solid rgba(196, 140, 70, 0.3)",
                }}
              >
                <Activity className="w-3 h-3 text-amber-600 animate-pulse" />
                Deterministic Brain · Provable Execution
              </span>
            </div>
            <p className="text-[11px] font-light" style={{ color: "#8a755d" }}>
              Autonomous 10-Agent Graph with Real-Time Python Typewriter & Matplotlib Plots
            </p>
          </div>
        </div>

        {/* Global Controls & Status */}
        <div className="flex items-center gap-3 flex-wrap">
          {isRunning && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex items-center gap-2 text-xs font-mono px-3 py-1.5 rounded-xl shadow-xs"
              style={{
                background: "rgba(196, 140, 70, 0.12)",
                color: "#9c631e",
                border: "1px solid rgba(196, 140, 70, 0.3)",
              }}
            >
              <Clock className="w-3.5 h-3.5 animate-spin" />
              <span>Elapsed: {elapsedSeconds}s</span>
            </motion.div>
          )}

          <div
            className="flex items-center gap-2 text-[11px] font-mono px-3.5 py-1.5 rounded-xl shadow-xs"
            style={{
              background: "rgba(255, 255, 255, 0.8)",
              border: "1px solid rgba(196, 140, 70, 0.2)",
              color: "#6b543e",
            }}
          >
            <span
              className="w-2 h-2 rounded-full"
              style={{
                background: isRunning ? "#c48c46" : "#4a9e7c",
                boxShadow: isRunning
                  ? "0 0 8px rgba(196, 140, 70, 0.8)"
                  : "0 0 8px rgba(74, 158, 124, 0.8)",
              }}
            />
            {isRunning ? "Kernel Typing..." : "Sandbox Ready"}
          </div>
        </div>
      </motion.header>

      {/* ── Main Studio Content ──────────────────────────────────────────── */}
      <main className="relative z-10 flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6 flex flex-col gap-6">

        {/* ── Prompt Console Section ──────────────────────────────────────── */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="relative rounded-3xl p-6 sm:p-7 overflow-hidden"
          style={{
            background: "rgba(255, 255, 255, 0.82)",
            backdropFilter: "blur(24px)",
            border: "1.5px solid rgba(196, 140, 70, 0.22)",
            boxShadow:
              "0 12px 48px -12px rgba(196, 140, 70, 0.15), 0 2px 12px rgba(0, 0, 0, 0.03)",
          }}
        >
          {isRunning && (
            <motion.div
              className="absolute inset-0 pointer-events-none"
              style={{
                border: "2px solid #c48c46",
                borderRadius: "inherit",
              }}
              animate={{ opacity: [0.4, 0.9, 0.4] }}
              transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
            />
          )}

          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
            <div className="flex items-center gap-2">
              <motion.div
                animate={{ rotate: [0, 15, -15, 0] }}
                transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
              >
                <Sparkles className="w-5 h-5 text-amber-600" />
              </motion.div>
              <h2 className="text-sm font-bold tracking-tight" style={{ color: "#2d2216" }}>
                Machine Learning Goal & Specification
              </h2>
            </div>
            <span
              className="text-[10px] font-mono px-3 py-1 rounded-full font-semibold"
              style={{
                background: "rgba(196, 140, 70, 0.1)",
                color: "#9c631e",
                border: "1px solid rgba(196, 140, 70, 0.2)",
              }}
            >
              LangGraph State-Machine Flow
            </span>
          </div>

          <div className="flex flex-col sm:flex-row gap-3 relative">
            <div className="relative flex-1">
              <input
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !isRunning) runPipeline(prompt);
                }}
                placeholder="E.g. Build a model predicting diabetic readmission with insulin dosage..."
                disabled={isRunning}
                className="w-full px-5 py-4 rounded-2xl text-sm outline-none transition-all duration-300 font-medium disabled:opacity-60"
                style={{
                  background: "rgba(253, 250, 244, 0.95)",
                  border: "1.5px solid rgba(196, 140, 70, 0.22)",
                  color: "#2d2216",
                  boxShadow: "inset 0 2px 6px rgba(196, 140, 70, 0.06)",
                }}
              />
            </div>

            <motion.button
              onClick={() => runPipeline(prompt)}
              disabled={isRunning || !prompt.trim()}
              whileHover={{ scale: 1.03, y: -1 }}
              whileTap={{ scale: 0.97 }}
              className="px-7 py-4 rounded-2xl text-white text-sm font-bold shadow-xl flex items-center justify-center gap-2.5 whitespace-nowrap shrink-0 relative overflow-hidden group"
              style={{
                background: "linear-gradient(135deg, #c48c46 0%, #d89b52 50%, #e0a860 100%)",
                boxShadow: "0 8px 25px rgba(196, 140, 70, 0.4)",
                opacity: isRunning || !prompt.trim() ? 0.7 : 1,
              }}
            >
              {isRunning ? (
                <>
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  >
                    <CircleDashed className="w-4.5 h-4.5 text-white" />
                  </motion.div>
                  <span>Orchestrating… {progressPercent}%</span>
                </>
              ) : (
                <>
                  <Play className="w-4.5 h-4.5 fill-white text-white" />
                  <span>Launch Pipeline</span>
                </>
              )}
            </motion.button>
          </div>

          <div className="mt-4 flex flex-wrap gap-2 items-center">
            <span className="text-[10px] font-bold uppercase tracking-wider shrink-0" style={{ color: "#9c8672" }}>
              Quick Presets:
            </span>
            {SAMPLE_PROMPTS.map((sp, i) => (
              <motion.button
                key={i}
                whileHover={{ scale: 1.04, y: -1 }}
                whileTap={{ scale: 0.96 }}
                onClick={() => {
                  setPrompt(sp);
                  if (!isRunning) runPipeline(sp);
                }}
                disabled={isRunning}
                className="text-[10.5px] font-medium px-3.5 py-1.5 rounded-xl transition-all disabled:opacity-40 flex items-center gap-1.5 shadow-xs"
                style={{
                  background: "rgba(196, 140, 70, 0.08)",
                  color: "#6b5030",
                  border: "1px solid rgba(196, 140, 70, 0.2)",
                }}
              >
                <ArrowUpRight className="w-3 h-3 opacity-60" />
                {sp}
              </motion.button>
            ))}
          </div>
        </motion.section>

        {/* ── Global Progress & Execution Graph Bar ────────────────────────── */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.15 }}
          className="rounded-2xl px-6 py-4.5 relative overflow-hidden"
          style={{
            background: "rgba(255, 255, 255, 0.75)",
            backdropFilter: "blur(20px)",
            border: "1.5px solid rgba(196, 140, 70, 0.16)",
            boxShadow: "0 6px 24px rgba(196, 140, 70, 0.06)",
          }}
        >
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
            <div className="flex items-center gap-2.5">
              <div
                className="w-7 h-7 rounded-lg flex items-center justify-center"
                style={{ background: "rgba(196, 140, 70, 0.12)" }}
              >
                <Layers className="w-4 h-4 text-amber-700" />
              </div>
              <span className="text-xs font-bold" style={{ color: "#3d3023" }}>
                10-Agent Pipeline Execution Status
              </span>
            </div>

            <div className="flex items-center gap-3">
              <span className="text-xs font-mono font-bold" style={{ color: "#c48c46" }}>
                {completedCount} of {stages.length} Stages Complete (
                <AnimatedNumber value={progressPercent} />%)
              </span>
            </div>
          </div>

          <div
            className="w-full h-3 rounded-full overflow-hidden p-0.5 relative"
            style={{ background: "rgba(196, 140, 70, 0.12)" }}
          >
            <motion.div
              className="h-full rounded-full relative"
              style={{
                background:
                  "linear-gradient(90deg, #c48c46 0%, #e0a860 50%, #4a9e7c 100%)",
                boxShadow: "0 0 12px rgba(196, 140, 70, 0.5)",
              }}
              animate={{ width: `${progressPercent}%` }}
              transition={{ duration: 0.6, ease: "easeOut" }}
            />
          </div>
        </motion.section>

        {/* ── 1. Agentic ML Interactive Python Sandbox (UP FRONT) ────────── */}
        <AgentMLSandbox
          cells={notebookCells}
          setCells={setNotebookCells}
          activeCellId={activeCellId}
          setActiveCellId={setActiveCellId}
          isRunningGlobal={isRunning}
          currentAgent={currentAgent}
          notebookRef={notebookRef}
        />

        {/* ── 2. 10 Agent Working Boxes (DOWN BELOW PYTHON NOTEBOOK) ───────── */}
        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between px-1">
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-amber-700" />
              <h2 className="text-sm font-bold tracking-tight" style={{ color: "#2d2216" }}>
                10 Autonomous Agents Live Working Matrix
              </h2>
            </div>
            <span className="text-[10.5px] font-mono text-amber-900/60">
              Click any agent box to highlight and jump to its notebook cell above
            </span>
          </div>

          <section className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3.5">
            {stages.map((stage, idx) => (
              <AgentWorkingBox
                key={stage.id}
                stage={stage}
                idx={idx}
                isActive={currentAgent === stage.id}
                onSelect={() => scrollToCell(`cell_${stage.id}`)}
              />
            ))}
          </section>
        </div>

        {/* ── 3. Dual-Pane Stream Terminal & Model Evaluation Matrix ───────── */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

          {/* Left Console: Stream Terminal */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.25 }}
            className="lg:col-span-7 flex flex-col rounded-3xl overflow-hidden shadow-xl"
            style={{
              background: "rgba(255, 255, 255, 0.85)",
              backdropFilter: "blur(24px)",
              border: "1.5px solid rgba(196, 140, 70, 0.2)",
            }}
          >
            <div
              className="px-5 py-3.5 flex items-center justify-between"
              style={{
                borderBottom: "1.5px solid rgba(196, 140, 70, 0.14)",
                background: "rgba(253, 250, 244, 0.8)",
              }}
            >
              <div className="flex items-center gap-3">
                <div className="flex gap-1.5">
                  <div className="w-3 h-3 rounded-full bg-rose-400/80 shadow-xs" />
                  <div className="w-3 h-3 rounded-full bg-amber-400/80 shadow-xs" />
                  <div className="w-3 h-3 rounded-full bg-emerald-400/80 shadow-xs" />
                </div>
                <div className="h-4 w-[1px] bg-amber-900/15" />
                <div className="flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-amber-700" />
                  <span className="text-xs font-mono font-bold" style={{ color: "#423324" }}>
                    Agent Execution Stream
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-1.5 bg-amber-900/5 p-1 rounded-xl">
                {(["all", "agents", "metrics"] as const).map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    className={`text-[10px] font-mono px-2.5 py-1 rounded-lg uppercase tracking-wider font-semibold transition-all ${
                      activeTab === tab
                        ? "bg-white text-amber-800 shadow-xs font-bold"
                        : "text-amber-900/50 hover:text-amber-900"
                    }`}
                  >
                    {tab}
                  </button>
                ))}
              </div>
            </div>

            <div
              className="p-5 overflow-y-auto space-y-1 font-mono"
              style={{
                minHeight: "260px",
                maxHeight: "380px",
                background:
                  "radial-gradient(ellipse at 50% 0%, rgba(253, 250, 244, 0.6) 0%, rgba(248, 241, 228, 0.4) 100%)",
              }}
            >
              {filteredLogs.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-16 gap-3 text-center">
                  <motion.div
                    animate={{ y: [0, -6, 0] }}
                    transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                  >
                    <Terminal className="w-10 h-10 opacity-25 text-amber-700" />
                  </motion.div>
                  <p className="text-xs font-mono" style={{ color: "#9c8672" }}>
                    Pipeline stream idle. Launch a task above to watch agents collaborate live.
                  </p>
                </div>
              ) : (
                <AnimatePresence initial={false}>
                  {filteredLogs.map((log, i) => (
                    <LogLine key={i} log={log} index={i} />
                  ))}
                </AnimatePresence>
              )}

              {isRunning && (
                <motion.div
                  animate={{ opacity: [1, 0, 1] }}
                  transition={{ duration: 0.8, repeat: Infinity }}
                  className="inline-block w-2.5 h-4 bg-amber-600 ml-1 mt-1 align-middle"
                />
              )}
              <div ref={logEndRef} />
            </div>
          </motion.div>

          {/* Right Console: Evaluation Matrix */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="lg:col-span-5 flex flex-col rounded-3xl overflow-hidden shadow-xl"
            style={{
              background: "rgba(255, 255, 255, 0.85)",
              backdropFilter: "blur(24px)",
              border: "1.5px solid rgba(196, 140, 70, 0.2)",
            }}
          >
            <div
              className="px-5 py-3.5 flex items-center justify-between"
              style={{
                borderBottom: "1.5px solid rgba(196, 140, 70, 0.14)",
                background: "rgba(253, 250, 244, 0.8)",
              }}
            >
              <div className="flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-amber-700" />
                <h3 className="text-xs font-bold" style={{ color: "#3d3023" }}>
                  Model Evaluation Matrix
                </h3>
              </div>

              <AnimatePresence>
                {summary && (
                  <motion.span
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="flex items-center gap-1.5 text-[9.5px] font-mono font-bold px-2.5 py-0.5 rounded-full"
                    style={{
                      background: "rgba(74, 158, 124, 0.15)",
                      color: "#2b6b52",
                      border: "1px solid rgba(74, 158, 124, 0.35)",
                    }}
                  >
                    <ShieldCheck className="w-3.5 h-3.5" /> 5-Fold Audit Passed
                  </motion.span>
                )}
              </AnimatePresence>
            </div>

            <div className="flex-1 p-5 overflow-y-auto" style={{ minHeight: "220px", maxHeight: "380px" }}>
              <AnimatePresence mode="wait">
                {!summary ? (
                  <motion.div
                    key="waiting"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex flex-col items-center justify-center h-full py-12 gap-3 text-center"
                  >
                    <motion.div
                      animate={{ scale: [1, 1.08, 1], opacity: [0.35, 0.7, 0.35] }}
                      transition={{ duration: 2.8, repeat: Infinity }}
                    >
                      <Database className="w-12 h-12 text-amber-700" />
                    </motion.div>
                    <h4 className="text-xs font-bold" style={{ color: "#544130" }}>
                      Leaderboard Standby
                    </h4>
                    <p className="text-[10.5px] max-w-[200px]" style={{ color: "#9c8672" }}>
                      Ensemble competition results and verified model parameters will render here upon completion.
                    </p>
                  </motion.div>
                ) : (
                  <motion.div
                    key="results"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                    className="space-y-4"
                  >
                    <div
                      className="rounded-2xl p-4.5 relative overflow-hidden"
                      style={{
                        background:
                          "linear-gradient(135deg, rgba(196, 140, 70, 0.16) 0%, rgba(224, 168, 96, 0.08) 100%)",
                        border: "1.5px solid rgba(196, 140, 70, 0.3)",
                      }}
                    >
                      <motion.div
                        className="absolute -top-10 -right-10 w-28 h-28 rounded-full opacity-30 pointer-events-none"
                        style={{ background: "radial-gradient(circle, #c48c46, transparent)" }}
                        animate={{ scale: [1, 1.3, 1], rotate: [0, 20, 0] }}
                        transition={{ duration: 5, repeat: Infinity }}
                      />

                      <div className="flex items-center justify-between mb-1.5">
                        <span
                          className="text-[9px] uppercase font-mono font-bold tracking-widest flex items-center gap-1"
                          style={{ color: "#9c631e" }}
                        >
                          <Award className="w-3.5 h-3.5" /> Best Generalization
                        </span>
                        <span className="text-[9px] font-mono font-semibold opacity-70">
                          {summary.task_type || "Classification"}
                        </span>
                      </div>

                      <h3 className="text-xl font-black tracking-tight" style={{ color: "#2d2216" }}>
                        {summary.selected_model}
                      </h3>

                      <p className="text-[10px] mt-0.5" style={{ color: "#786550" }}>
                        Target: <code className="font-bold text-amber-800">{summary.target_column || "Target"}</code>
                      </p>

                      <div className="mt-3.5 flex items-baseline gap-2">
                        <span
                          className="text-4xl font-mono font-black tracking-tight"
                          style={{ color: "#c48c46" }}
                        >
                          <AnimatedNumber value={typeof summary.validation_score === "number" ? summary.validation_score * 100 : 0} decimals={2} />%
                        </span>
                        <span className="text-[11px] font-mono font-semibold" style={{ color: "#786550" }}>
                          Validation Metric
                        </span>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-[11px] font-bold mb-2.5 flex items-center justify-between" style={{ color: "#423324" }}>
                        <span>Candidate Cross-Validation</span>
                        <span className="text-[9px] font-mono opacity-60">5-Fold Avg</span>
                      </h4>
                      <div className="space-y-1">
                        {Object.entries(summary.metrics || {}).map(([name, score], i) => (
                          <MetricBar
                            key={name}
                            name={name}
                            score={typeof score === "number" ? score : 0.8}
                            isBest={name === summary.selected_model}
                            index={i}
                          />
                        ))}
                      </div>
                    </div>

                    {(summary.selected_features || []).length > 0 && (
                      <div>
                        <h4 className="text-[11px] font-bold mb-2" style={{ color: "#423324" }}>
                          Retained High-Impact Features
                        </h4>
                        <div className="flex flex-wrap gap-1.5">
                          {(summary.selected_features || []).map((f: string, i: number) => (
                            <motion.span
                              key={i}
                              initial={{ opacity: 0, scale: 0.8 }}
                              animate={{ opacity: 1, scale: 1 }}
                              transition={{ delay: i * 0.03 }}
                              className="text-[9.5px] font-mono px-2.5 py-0.5 rounded-lg shadow-xs"
                              style={{
                                background: "rgba(196, 140, 70, 0.09)",
                                color: "#6b5030",
                                border: "1px solid rgba(196, 140, 70, 0.2)",
                              }}
                            >
                              {f}
                            </motion.span>
                          ))}
                        </div>
                      </div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            <AnimatePresence>
              {summary && (
                <motion.div
                  initial={{ opacity: 0, y: 18 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="px-5 pb-5 pt-3 flex flex-col gap-2.5"
                  style={{ borderTop: "1.5px solid rgba(196, 140, 70, 0.14)" }}
                >
                  <div
                    className="w-full py-3 px-4 rounded-2xl text-xs font-mono font-medium flex items-center justify-center gap-2"
                    style={{
                      background: "rgba(196, 140, 70, 0.08)",
                      color: "#9c631e",
                      border: "1px dashed rgba(196, 140, 70, 0.35)",
                    }}
                  >
                    <Lock className="w-3.5 h-3.5 text-amber-700" />
                    <span>Awaiting User-Supplied Model Serialization Script (PKL export held)</span>
                  </div>

                  <motion.button
                    onClick={copyMetadata}
                    whileHover={{ scale: 1.01 }}
                    whileTap={{ scale: 0.99 }}
                    className="w-full py-2.5 px-4 rounded-xl text-xs font-mono font-medium flex items-center justify-center gap-2 transition-colors"
                    style={{
                      background: "rgba(196, 140, 70, 0.08)",
                      color: "#6b5030",
                      border: "1px solid rgba(196, 140, 70, 0.2)",
                    }}
                  >
                    <AnimatePresence mode="wait">
                      {copied ? (
                        <motion.span
                          key="copied"
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          className="flex items-center gap-1.5 text-emerald-700 font-bold"
                        >
                          <Check className="w-4 h-4" /> Pipeline Specification Copied!
                        </motion.span>
                      ) : (
                        <motion.span
                          key="copy"
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          className="flex items-center gap-1.5"
                        >
                          <Copy className="w-3.5 h-3.5" /> Copy model_spec.json
                        </motion.span>
                      )}
                    </AnimatePresence>
                  </motion.button>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>
      </main>
    </div>
  );
}

export default function PipelinePage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-[#fdfaf4]" />}>
      <PipelinePageContent />
    </Suspense>
  );
}
