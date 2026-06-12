import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any

class EvalTracker:
    def __init__(self, storage_dir: str = "eval_runs"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def save_run(self, model_version: str, dataset_name: str, results: List[Dict[str, Any]], aggregate_metrics: Dict[str, float]):
        """Saves an evaluation run to track history and drift."""
        run_id = int(time.time())
        timestamp = datetime.now().isoformat()
        
        run_data = {
            "run_id": run_id,
            "timestamp": timestamp,
            "model_version": model_version,
            "dataset_name": dataset_name,
            "aggregate_metrics": aggregate_metrics,
            "details": results
        }
        
        filename = os.path.join(self.storage_dir, f"run_{run_id}.json")
        with open(filename, 'w') as f:
            json.dump(run_data, f, indent=4)
        
        print(f"✅ Saved eval run to {filename}")

    def load_all_runs(self) -> List[Dict[str, Any]]:
        """Loads all historical runs for dashboard visualization."""
        runs = []
        if not os.path.exists(self.storage_dir):
            return runs
            
        for file in os.listdir(self.storage_dir):
            if file.endswith(".json"):
                with open(os.path.join(self.storage_dir, file), 'r') as f:
                    runs.append(json.load(f))
        
        # Sort by run_id (timestamp)
        return sorted(runs, key=lambda x: x["run_id"])
