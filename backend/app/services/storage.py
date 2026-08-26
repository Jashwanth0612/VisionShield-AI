from __future__ import annotations

import json
import mimetypes
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings


class StorageService:
    """Local production-safe storage boundary.

    Metadata is SQLite-backed and media is stored on disk. The API is intentionally
    small so the implementation can later be replaced by MongoDB + S3/object storage.
    """

    def __init__(self) -> None:
        self.artifact_root = Path(settings.artifact_dir).expanduser()
        self.db_path = Path(settings.database_path).expanduser()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS inference_runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    nafnet_enabled INTEGER NOT NULL,
                    confidence_threshold REAL NOT NULL,
                    detections INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    fps REAL NOT NULL,
                    model_status TEXT NOT NULL,
                    model_config TEXT NOT NULL,
                    artifacts_json TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_inference_timestamp ON inference_runs(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_inference_filename ON inference_runs(filename);

                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    iterations INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    min_latency_ms REAL NOT NULL,
                    max_latency_ms REAL NOT NULL,
                    fps REAL NOT NULL,
                    detections REAL NOT NULL,
                    model_config TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_benchmark_timestamp ON benchmark_runs(timestamp ASC);
                """
            )

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def save_artifact(self, data: bytes, run_id: str, kind: str, filename: str | None, content_type: str | None) -> dict[str, Any]:
        suffix = Path(filename or "artifact.bin").suffix.lower()
        if not suffix:
            suffix = mimetypes.guess_extension(content_type or "") or ".bin"
        artifact_id = f"{run_id}_{kind}_{uuid.uuid4().hex[:8]}"
        path = self.artifact_root / f"{artifact_id}{suffix}"
        path.write_bytes(data)
        return {
            "artifact_id": artifact_id,
            "kind": kind,
            "filename": path.name,
            "url": f"/artifacts/{artifact_id}",
            "content_type": content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            "size_bytes": len(data),
        }

    def artifact_path(self, artifact_id: str) -> Path | None:
        matches = list(self.artifact_root.glob(f"{artifact_id}.*"))
        return matches[0] if matches else None

    def record_inference(self, record: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO inference_runs
                (run_id,timestamp,media_type,filename,nafnet_enabled,confidence_threshold,detections,latency_ms,fps,model_status,model_config,artifacts_json,details_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record["run_id"], record["timestamp"], record["media_type"], record["filename"],
                    int(record["nafnet_enabled"]), record["confidence_threshold"], record["detections"],
                    record["latency_ms"], record["fps"], record["model_status"], record["model_config"],
                    json.dumps(record.get("artifacts", {})), json.dumps(record.get("details", {})),
                ),
            )

    def list_inferences(self, search: str = "", media_type: str = "all", limit: int = 50) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if media_type in {"image", "video"}:
            clauses.append("media_type = ?")
            params.append(media_type)
        if search.strip():
            clauses.append("filename LIKE ?")
            params.append(f"%{search.strip()}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM inference_runs {where} ORDER BY timestamp DESC LIMIT ?",
                (*params, max(1, min(limit, 100))),
            ).fetchall()
        return [self._inference_row(row) for row in rows]

    def get_inference(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM inference_runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._inference_row(row) if row else None

    def _inference_row(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["nafnet_enabled"] = bool(item["nafnet_enabled"])
        item["artifacts"] = json.loads(item.pop("artifacts_json"))
        item["details"] = json.loads(item.pop("details_json"))
        return item

    def record_benchmark(self, record: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO benchmark_runs
                (run_id,timestamp,media_type,filename,iterations,latency_ms,min_latency_ms,max_latency_ms,fps,detections,model_config,details_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record["run_id"], record["timestamp"], record["media_type"], record["filename"], record["iterations"],
                    record["latency_ms"], record["min_latency_ms"], record["max_latency_ms"], record["fps"],
                    record["detections"], record["model_config"], json.dumps(record.get("details", {})),
                ),
            )

    def list_benchmarks(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM benchmark_runs ORDER BY timestamp ASC").fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["details"] = json.loads(item.pop("details_json"))
            result.append(item)
        return result

    def benchmark_summary(self) -> dict[str, Any]:
        runs = self.list_benchmarks()
        if not runs:
            return {"runs": [], "summary": None, "message": "No measured benchmark runs available"}
        latencies = [float(run["latency_ms"]) for run in runs]
        fps = [float(run["fps"]) for run in runs]
        detections = [float(run["detections"]) for run in runs]
        return {
            "runs": runs,
            "summary": {
                "runs": len(runs),
                "average_latency_ms": sum(latencies) / len(latencies),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "average_fps": sum(fps) / len(fps),
                "average_detections": sum(detections) / len(detections),
            },
        }

    def inference_count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM inference_runs").fetchone()[0])


storage = StorageService()
