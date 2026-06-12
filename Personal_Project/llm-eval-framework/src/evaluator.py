import time
from typing import List, Dict, Any, Callable
from .metrics import exact_match, valid_json, contains_keyword, llm_judge_relevance
from .tracker import EvalTracker

class Evaluator:
    def __init__(self, model_version: str, dataset_name: str):
        self.model_version = model_version
        self.dataset_name = dataset_name
        self.tracker = EvalTracker()

    def mock_llm_call(self, prompt: str) -> str:
        """Mock LLM generator. Replace with real API calls (OpenAI, Gemini, Anthropic)."""
        # For demonstration purposes, hardcoded responses
        if "capital of France" in prompt:
            return "Paris"
        elif "JSON" in prompt:
            return '{"name": "Alice", "age": 30}'
        else:
            return "This is a generic LLM response."

    def evaluate_dataset(self, dataset: List[Dict[str, str]]):
        """Runs the benchmark and calculates metrics."""
        results = []
        
        # Tracking metrics accumulators
        total_exact_match = 0
        total_valid_json = 0
        total_relevance = 0
        total_latency = 0
        
        print(f"🚀 Starting evaluation for {self.model_version} on {self.dataset_name}...")
        
        for item in dataset:
            prompt = item["prompt"]
            expected = item.get("expected", "")
            
            # 1. Measure Latency
            start_time = time.time()
            prediction = self.mock_llm_call(prompt)
            latency = time.time() - start_time
            
            # 2. Compute Metrics
            em_score = exact_match(prediction, expected)
            json_score = valid_json(prediction)
            relevance_score = llm_judge_relevance(prompt, prediction)
            
            # 3. Store item results
            results.append({
                "prompt": prompt,
                "expected": expected,
                "prediction": prediction,
                "latency_sec": latency,
                "metrics": {
                    "exact_match": em_score,
                    "valid_json": json_score,
                    "relevance": relevance_score
                }
            })
            
            # Accumulate
            total_exact_match += em_score
            total_valid_json += json_score
            total_relevance += relevance_score
            total_latency += latency
            
        # Calculate Aggregates
        num_items = len(dataset)
        agg_metrics = {
            "avg_exact_match": total_exact_match / num_items,
            "avg_valid_json": total_valid_json / num_items,
            "avg_relevance": total_relevance / num_items,
            "avg_latency_sec": total_latency / num_items
        }
        
        print(f"📊 Results: {agg_metrics}")
        
        # Save to Tracker
        self.tracker.save_run(self.model_version, self.dataset_name, results, agg_metrics)
        return agg_metrics
