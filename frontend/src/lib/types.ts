export interface PipelineRun {
  runId?: string;
  taskType?: string;
  status?: "pending" | "running" | "completed" | "failed" | "hitl_required";
  createdAt?: string;
  bestModelName?: string;
  bestModelMetrics?: Record<string, number>;
}
