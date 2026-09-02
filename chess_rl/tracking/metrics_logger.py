"""
Metrics logging utility for streaming experiment telemetry.
Provides concurrent and append-safe logging to CSV and JSONL formats.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


class MetricsLogger:
    """
    Logs metric dictionaries to CSV and JSONL files simultaneously.
    Ensures immediate disk flush after every write for live monitoring.
    """

    def __init__(self, output_dir: Path, csv_filename: str = "training_metrics.csv", jsonl_filename: str = "metrics.jsonl"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.output_dir / csv_filename
        self.jsonl_path = self.output_dir / jsonl_filename
        
        self._csv_headers_written = self.csv_path.exists()
        self._fieldnames: Optional[List[str]] = None

    def log(self, metrics: Dict[str, Any]) -> None:
        """
        Log a single row/event of metrics.

        Args:
            metrics: Key-value mapping of metric names and numeric/string values.
        """
        # Append to JSON Lines
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
            f.flush()

        # Append to CSV
        if not self._csv_headers_written or self._fieldnames is None:
            self._fieldnames = list(metrics.keys())
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                if not self._csv_headers_written:
                    writer.writeheader()
                    self._csv_headers_written = True
                writer.writerow(metrics)
                f.flush()
        else:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
                writer.writerow(metrics)
                f.flush()
