from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Final


# Public Google Drive file IDs for the six production checkpoints. The files are
# intentionally kept outside GitHub because the combined size is ~1.94 GB.
DEFAULT_DRIVE_IDS: Final[dict[str, str]] = {
    "NAFNet_FOG_25p15.pth": "1lLlOSoLw0bs7v6hjhu-CZhZ3Fe9EVTB-",
    "NAFNet_LOL_best.pth": "1QRf2-pyk4NTnNjQf_waVT0qWoZv5yVpK",
    "NAFNet_OTS_best.pth": "19o_HoeqhEJQH36A-OK8yDOT2l5rqFN90",
    "NAFNet_RAIN_best.pth": "1buwK7-BilGUz5tKAVGAcYUs-XqyhFz18",
    "NAFNet_SNOW_best.pth": "1T6ElcXm7QglY1qDZFaYUAQMSQ8lbB-I1",
    "RTDETRv2_AQM_SOS_best.pth": "15OogZ9dKjR9MOsmxa26rUE7QB9-RcLgs",
}


class DriveWeightError(RuntimeError):
    """Raised when a configured remote checkpoint cannot be materialized."""


class DriveWeightService:
    """Materialize public Google Drive checkpoints into Render's ephemeral disk.

    Downloads are lazy: a checkpoint is fetched only when its model is first used.
    This avoids downloading all ~1.94 GB during every Render cold start.
    """

    def __init__(self, root: str | Path = "data/models") -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, filename: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(filename, threading.Lock())

    @staticmethod
    def _drive_id(filename: str) -> str | None:
        env_key = "DRIVE_" + filename.replace(".", "_").replace("-", "_").upper()
        return os.getenv(env_key) or DEFAULT_DRIVE_IDS.get(filename)

    def is_configured(self, filename: str) -> bool:
        return bool(self._drive_id(filename))

    def local_path(self, filename: str) -> Path:
        return self.root / filename

    def ensure(self, filename: str) -> Path:
        path = self.local_path(filename)
        if path.is_file() and path.stat().st_size > 1024 * 1024:
            return path

        file_id = self._drive_id(filename)
        if not file_id:
            raise DriveWeightError(f"No Google Drive file ID configured for {filename}.")

        lock = self._lock_for(filename)
        with lock:
            if path.is_file() and path.stat().st_size > 1024 * 1024:
                return path

            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".part")
            try:
                import gdown

                if temporary.exists():
                    temporary.unlink()
                downloaded = gdown.download(
                    id=file_id,
                    output=str(temporary),
                    quiet=False,
                    fuzzy=False,
                )
                if not downloaded or not temporary.is_file():
                    raise DriveWeightError("Google Drive did not return a checkpoint file.")
                if temporary.stat().st_size <= 1024 * 1024:
                    raise DriveWeightError(
                        "Downloaded checkpoint is unexpectedly small; the Drive link may "
                        "not be publicly downloadable or Google may have returned an error page."
                    )
                temporary.replace(path)
                return path
            except DriveWeightError:
                raise
            except Exception as exc:
                raise DriveWeightError(f"Unable to download {filename} from Google Drive: {exc}") from exc
            finally:
                if temporary.exists():
                    temporary.unlink(missing_ok=True)

    def status(self) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for filename in DEFAULT_DRIVE_IDS:
            path = self.local_path(filename)
            result[filename] = {
                "configured": self.is_configured(filename),
                "downloaded": path.is_file() and path.stat().st_size > 1024 * 1024,
                "path": str(path),
                "size_bytes": path.stat().st_size if path.is_file() else 0,
            }
        return result


# Shared service instance used by model adapters.
drive_weights = DriveWeightService()
