# LLM Evaluation Metrics

To build a robust LLM evaluation framework that tracks regressions and drift, we need to define metrics across different dimensions. Here is a starting list of metrics we can track:

## 1. Quality & Semantic Metrics (LLM-as-a-Judge)
These metrics require another LLM (like GPT-4 or Claude 3) to evaluate the output.
* **Faithfulness / Hallucination Rate**: Is the generated answer completely supported by the provided context?
* **Answer Relevance**: Does the answer directly address the user's prompt without unnecessary tangents?
* **Context Precision & Recall** (For RAG): Did the retrieval system fetch the right documents, and did the model use them effectively?
* **Tone & Formatting Compliance**: Did the model follow specific instructions (e.g., "Answer in JSON", "Be polite")?

## 2. Deterministic / Lexical Metrics
These are fast, cheap, and easily computable using traditional code.
* **Exact Match (EM)**: Did the model output the exact expected string? (Good for multiple choice or short QA).
* **Regex / JSON Parsing**: Did the model output valid, parseable JSON or code blocks?
* **Sub-string Presence**: Does the output contain required keywords?

## 3. Task-Specific Metrics
* **Code Pass@k**: For code generation, does the generated code pass a set of unit tests?
* **Function Calling Accuracy**: Did the model select the correct tool and provide the correct arguments?

## 4. Operational & Drift Metrics
Crucial for tracking regressions across model versions or system updates.
* **Latency**: Time to First Token (TTFT) and Total Generation Time.
* **Throughput**: Tokens generated per second.
* **Cost**: Estimated cost per 1,000 queries based on input/output token counts.
* **Error Rate**: Rate of API timeouts, malformed outputs, or context window limits.

---
### Next Steps
Which of these metric categories are most important for your specific use case? (e.g., Are you building a RAG chatbot, a coding assistant, or an extraction tool?)
