import json
import time
from pathlib import Path
from typing import Optional


class ExperimentLogger:
    def __init__(self, log_path: str | Path, print_every: Optional[float] = None):
        self.log_path = Path(log_path)
        self.start_time = time.time()
        self.records: list[dict] = []
        self.print_every = print_every
        self.last_print_time = 0.0

        # Create log directory if it doesn't exist
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_existing_records()

    def log(self, step: int, train_loss: float, val_loss: float):
        """Record one log entry."""
        elapsed = time.time() - self.start_time
        entry = {
            "step": step,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "elapsed_sec": round(elapsed, 2)
        }
        self.records.append(entry)

        # Optionally print to console
        if self.print_every is not None:
            if time.time() - self.last_print_time > self.print_every:
                print(
                    f"[step {step:5d}] "
                    f"train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}, "
                    f"time = {elapsed:.1f}s"
                )
                self.last_print_time = time.time()

    def save(self):
        """Write all logs to a JSON file."""
        with open(self.log_path, "w") as f:
            json.dump(self.records, f, indent=2)
        print(f"Experiment log saved to {self.log_path.resolve()}")

    def latest(self) -> Optional[dict]:
        """Return the latest record (or None if empty)."""
        return self.records[-1] if self.records else None

    def _load_existing_records(self) -> None:
        """Load existing log records if the file already exists."""
        if not self.log_path.exists():
            return

        try:
            with open(self.log_path, "r") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: unable to parse existing log file at {self.log_path}, starting a new log.")
            return

        if isinstance(existing, list):
            self.records = [entry for entry in existing if isinstance(entry, dict)]
            if self.records:
                last_elapsed = self.records[-1].get("elapsed_sec")
                if isinstance(last_elapsed, (int, float)):
                    # Offset start time so elapsed seconds continue increasing when resuming.
                    self.start_time = time.time() - float(last_elapsed)
            # Avoid spamming console immediately after resume.
            self.last_print_time = time.time()
