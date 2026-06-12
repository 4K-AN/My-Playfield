import json
import re

def exact_match(prediction: str, target: str) -> float:
    """Checks if the prediction exactly matches the target (case-insensitive)."""
    if not prediction or not target:
        return 0.0
    return 1.0 if prediction.strip().lower() == target.strip().lower() else 0.0

def contains_keyword(prediction: str, target: str) -> float:
    """Checks if the target keyword exists in the prediction."""
    if not prediction or not target:
        return 0.0
    return 1.0 if target.strip().lower() in prediction.strip().lower() else 0.0

def valid_json(prediction: str) -> float:
    """Checks if the prediction is valid JSON."""
    try:
        # Sometimes models wrap json in markdown blocks
        clean_pred = re.sub(r'```json|```', '', prediction).strip()
        json.loads(clean_pred)
        return 1.0
    except json.JSONDecodeError:
        return 0.0

# LLM-as-a-judge stub
def llm_judge_relevance(prompt: str, prediction: str) -> float:
    """
    Evaluates how relevant the prediction is to the prompt.
    In a real system, you'd call an API like OpenAI/Anthropic here to grade it on a 0-1 scale.
    For demonstration, we return a mock value.
    """
    # Mock logic: if prediction is longer than prompt, consider it detailed (mock)
    if len(prediction) > 10:
        return 0.9
    return 0.2
