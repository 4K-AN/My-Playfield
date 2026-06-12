from src.evaluator import Evaluator
import time

dataset = [
    {"prompt": "What is the capital of France?", "expected": "Paris"},
    {"prompt": "Generate a user JSON object.", "expected": ""},
    {"prompt": "Explain quantum computing in one sentence.", "expected": ""}
]

# Simulate a series of model runs over time to show drift

print("--- RUN 1 (Baseline) ---")
evaluator_v1 = Evaluator(model_version="gpt-3.5-v1", dataset_name="General-QA")
# Monkeypatch mock latency
evaluator_v1.evaluate_dataset(dataset)
time.sleep(1) # Ensure timestamp difference

print("\n--- RUN 2 (Regression in JSON generation) ---")
evaluator_v2 = Evaluator(model_version="gpt-3.5-v2", dataset_name="General-QA")
# We'll artificially break the mock response to simulate a regression
evaluator_v2.mock_llm_call = lambda prompt: "Paris" if "France" in prompt else "Sorry, I can't do JSON anymore."
evaluator_v2.evaluate_dataset(dataset)
time.sleep(1)

print("\n--- RUN 3 (Fix applied, but latency increased) ---")
evaluator_v3 = Evaluator(model_version="gpt-4-v1", dataset_name="General-QA")
def slow_mock(prompt):
    time.sleep(0.5) # Simulate higher latency
    if "France" in prompt: return "Paris"
    if "JSON" in prompt: return '{"name": "Bob", "status": "fixed"}'
    return "Quantum computing is complex."
evaluator_v3.mock_llm_call = slow_mock
evaluator_v3.evaluate_dataset(dataset)

print("\n✅ Simulated runs completed. Now start the dashboard:")
print("streamlit run dashboard.py")
