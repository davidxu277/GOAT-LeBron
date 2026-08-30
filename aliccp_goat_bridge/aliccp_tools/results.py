"""Experiment result registry and final table generation."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def record_result(
    metrics_path: Path,
    experiment: str,
    results_dir: Path,
    gpu_hours: float = 0.0,
    llm_tokens: int = 0,
    human_interventions: int = 0,
    notes: str = "",
) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    record = {
        "experiment": experiment,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ctr_auc": metrics.get("ctr_auc"),
        "cvr_auc": metrics.get("cvr_auc"),
        "rows": metrics.get("rows"),
        "clicked_rows": metrics.get("clicked_rows"),
        "gpu_hours": float(gpu_hours),
        "llm_tokens": int(llm_tokens),
        "human_interventions": int(human_interventions),
        "metrics_path": str(metrics_path.resolve()),
        "notes": notes,
    }
    if record["ctr_auc"] is not None and record["cvr_auc"] is not None:
        record["objective"] = (float(record["ctr_auc"]) + float(record["cvr_auc"])) / 2.0
    else:
        record["objective"] = None
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{experiment}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def build_results_table(
    results_dir: Path,
    output_dir: Path,
    baseline_ctr: float | None = None,
    baseline_cvr: float | None = None,
) -> dict[str, Any]:
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(results_dir.glob("*.json"))]
    for record in records:
        record["delta_ctr"] = float(record["ctr_auc"]) - baseline_ctr if baseline_ctr is not None and record.get("ctr_auc") is not None else None
        record["delta_cvr"] = float(record["cvr_auc"]) - baseline_cvr if baseline_cvr is not None and record.get("cvr_auc") is not None else None
        record["delta_mean"] = (
            (record["delta_ctr"] + record["delta_cvr"]) / 2.0
            if record["delta_ctr"] is not None and record["delta_cvr"] is not None else None
        )
    eligible = [record for record in records if record.get("objective") is not None]
    best = max(eligible, key=lambda record: record["objective"]) if eligible else None
    summary = {
        "experiments": len(records),
        "best_experiment": best["experiment"] if best else None,
        "best_metrics": {"ctr_auc": best["ctr_auc"], "cvr_auc": best["cvr_auc"]} if best else None,
        "baseline": {"ctr_auc": baseline_ctr, "cvr_auc": baseline_cvr},
        "total_gpu_hours": sum(float(record.get("gpu_hours", 0)) for record in records),
        "total_llm_tokens": sum(int(record.get("llm_tokens", 0)) for record in records),
        "total_human_interventions": sum(int(record.get("human_interventions", 0)) for record in records),
        "records": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "final_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    columns = ["experiment", "ctr_auc", "cvr_auc", "objective", "delta_ctr", "delta_cvr", "delta_mean", "gpu_hours", "llm_tokens", "human_interventions", "notes"]
    with (output_dir / "final_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    lines = ["# Final Results", "", "| Experiment | CTR AUC | CVR AUC | ΔCTR | ΔCVR | GPU hours | LLM tokens | Human interventions |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for record in records:
        display = lambda value: "N/A" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)
        lines.append(f"| {record['experiment']} | {display(record.get('ctr_auc'))} | {display(record.get('cvr_auc'))} | {display(record.get('delta_ctr'))} | {display(record.get('delta_cvr'))} | {record.get('gpu_hours', 0)} | {record.get('llm_tokens', 0)} | {record.get('human_interventions', 0)} |")
    lines.extend(["", f"Best experiment: **{summary['best_experiment'] or 'N/A'}**", ""])
    (output_dir / "final_results.md").write_text("\n".join(lines), encoding="utf-8")
    return summary

