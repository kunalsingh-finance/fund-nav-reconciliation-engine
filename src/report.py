from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _prepare_export_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_datetime64_any_dtype(output[column]):
            output[column] = output[column].dt.strftime("%Y-%m-%d")
    return output


def export_output_pack(
    positions: pd.DataFrame,
    nav: pd.DataFrame,
    reconciliation: pd.DataFrame,
    output_dir: Path,
    asof_date: pd.Timestamp,
    fund_id: str,
) -> Path:
    asof = pd.Timestamp(asof_date).strftime("%Y-%m-%d")
    export_dir = output_dir / f"{asof}_{fund_id}"
    export_dir.mkdir(parents=True, exist_ok=True)

    positions_path = export_dir / "positions.csv"
    nav_path = export_dir / "nav.csv"
    reconciliation_path = export_dir / "reconciliation.csv"
    break_report_path = export_dir / "break_report.csv"
    summary_path = export_dir / "summary.json"

    _prepare_export_frame(positions).to_csv(positions_path, index=False)
    _prepare_export_frame(nav).to_csv(nav_path, index=False)
    _prepare_export_frame(reconciliation).to_csv(reconciliation_path, index=False)
    _prepare_export_frame(reconciliation).to_csv(break_report_path, index=False)

    nav_row = nav.iloc[0].to_dict() if not nav.empty else {}
    summary: dict[str, Any] = {
        "asof_date": asof,
        "fund_id": fund_id,
        "total_nav": float(nav_row.get("total_nav", 0.0)),
        "nav_per_share": float(nav_row.get("nav_per_share", 0.0)),
        "break_count": int(len(reconciliation)),
        "high_severity_breaks": int((reconciliation.get("severity") == "HIGH").sum()) if not reconciliation.empty else 0,
        "status": "REVIEW_REQUIRED" if not reconciliation.empty else "SUCCESS",
        "exports": {
            "positions": positions_path.name,
            "nav": nav_path.name,
            "reconciliation": reconciliation_path.name,
            "break_report": break_report_path.name,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return export_dir
