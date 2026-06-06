"""
EXAMPLE 2: Research Agent
Agent yang melakukan research multi-step dengan memory persistence
Menunjukkan: multi-step planning, memory reuse, continuation
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import Agent
from src.tools import ToolRegistry, ToolParameter
from src.utils import setup_logging, print_result


# Custom tools untuk research
def web_search(query: str) -> dict:
    """Simulate web search"""
    return {
        "query": query,
        "results": [
            {
                "title": f"Article 1: {query}",
                "snippet": f"Information about {query}...",
                "url": "https://example.com/1"
            },
            {
                "title": f"Article 2: Advanced {query}",
                "snippet": f"Advanced information about {query}...",
                "url": "https://example.com/2"
            }
        ],
        "count": 2
    }


def summarize_text(text: str) -> dict:
    """Simulate text summarization"""
    words = text.split()
    summary = " ".join(words[:min(30, len(words))])
    
    return {
        "original_length": len(text),
        "summary": summary + "...",
        "summary_length": len(summary)
    }


def analyze_sentiment(text: str) -> dict:
    """Simulate sentiment analysis"""
    # Simple heuristic: count positive/negative words
    positive_words = ["good", "great", "excellent", "amazing", "best"]
    negative_words = ["bad", "poor", "terrible", "worst", "awful"]
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        sentiment = "positive"
    elif neg_count > pos_count:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return {
        "sentiment": sentiment,
        "confidence": 0.8,
        "positive_score": pos_count,
        "negative_score": neg_count
    }


def extract_key_facts(text: str) -> dict:
    """Extract key information dari text"""
    sentences = text.split(".")
    key_facts = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
    
    return {
        "key_facts": key_facts,
        "total_facts": len(key_facts)
    }


def create_research_agent() -> Agent:
    """Create agent dengan custom research tools"""
    
    # Create registry dengan tools
    registry = ToolRegistry()
    
    # Register research tools
    registry.register(
        name="web_search",
        description="Search web untuk informasi tentang topic",
        implementation=web_search,
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                required=True,
                description="Search query"
            )
        ]
    )
    
    registry.register(
        name="summarize",
        description="Summarize panjang text menjadi ringkasan singkat",
        implementation=summarize_text,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                required=True,
                description="Text untuk summarize"
            )
        ]
    )
    
    registry.register(
        name="sentiment_analysis",
        description="Analisis sentiment dari text",
        implementation=analyze_sentiment,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                required=True,
                description="Text untuk analyze"
            )
        ]
    )
    
    registry.register(
        name="extract_facts",
        description="Extract key facts dari text",
        implementation=extract_key_facts,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                required=True,
                description="Text untuk extract facts"
            )
        ]
    )
    
    # Create agent dengan registry
    agent = Agent(
        model="gpt-4",
        memory_path="./memory/research",
        max_iterations=8,
        token_budget=3000,
        use_persistent_memory=True,
        tool_registry=registry
    )
    
    return agent


def demonstrate_continuation():
    """
    Demonstrate continuation dari previous session
    """
    
    print("\n" + "="*60)
    print("RESEARCH AGENT - CONTINUATION EXAMPLE")
    print("="*60)
    
    logger = setup_logging(level="INFO")
    
    # Session 1: Initial research
    print("\n--- SESSION 1: Initial Research ---")
    
    agent = create_research_agent()
    
    result1 = agent.run(
        task="Research about quantum computing",
        session_id="research_session_001",
        max_retries=2
    )
    
    print_result(result1)
    
    print("\nSession 1 Memory State:")
    print(f"  Conversation messages: {len(agent.memory.conversation_history)}")
    print(f"  Facts stored: {len(agent.memory.facts)}")
    print(f"  Failed attempts: {len(agent.memory.failed_attempts)}")
    
    # Show saved facts
    if agent.memory.facts:
        print("\n  Saved Facts:")
        for fact in agent.memory.get_facts()[:3]:
            print(f"    - {fact['key']}: {str(fact['value'])[:50]}...")
    
    # Session 2: Continuation
    print("\n--- SESSION 2: Continue Research (Different Angle) ---")
    
    agent2 = create_research_agent()
    
    result2 = agent2.run(
        task="Expand on quantum computing applications",
        session_id="research_session_001",  # Same session ID!
        max_retries=2
    )
    
    print_result(result2)
    
    print("\nSession 2 Memory State:")
    print(f"  Conversation messages: {len(agent2.memory.conversation_history)}")
    print(f"  Facts stored: {len(agent2.memory.facts)}")
    print(f"  Previous facts loaded: {len(agent2.memory.get_facts()) > 0}")
    
    # Comparison
    print("\n--- COMPARISON ---")
    print(f"Session 1: {result1.iterations} iterations, {result1.total_tokens} tokens")
    print(f"Session 2: {result2.iterations} iterations, {result2.total_tokens} tokens")
    
    if result1.success and result2.success:
        print("✓ Both sessions successful!")
    
    return result1, result2


def demonstrate_memory_reuse():
    """
    Demonstrate memory reuse dalam execution
    """
    
    print("\n" + "="*60)
    print("RESEARCH AGENT - MEMORY REUSE EXAMPLE")
    print("="*60)
    
    logger = setup_logging(level="INFO")
    
    agent = create_research_agent()
    
    # Manually add facts untuk simulate learning dari previous sessions
    agent.memory.add_fact(
        key="known_quantum_experts",
        value=["Richard Feynman", "David Deutsch"],
        category="fact"
    )
    
    agent.memory.add_fact(
        key="quantum_algorithms",
        value=["Shor's algorithm", "Grover's algorithm"],
        category="fact",
        relevance_score=0.95
    )
    
    print("Manually added facts dari 'previous learning'")
    print("  - Known quantum experts")
    print("  - Quantum algorithms")
    
    # Run task yang bisa benefit dari facts
    result = agent.run(
        task="Summarize quantum computing breakthroughs and key figures",
        session_id="research_with_context",
        max_retries=2
    )
    
    print_result(result)
    
    print("\nMemory Context Used:")
    print(agent.memory.get_context_summary())
    
    return result


def main():
    """Main demo function"""
    
    demo_type = sys.argv[1] if len(sys.argv) > 1 else "continuation"
    
    if demo_type == "continuation":
        return demonstrate_continuation()
    elif demo_type == "memory_reuse":
        return demonstrate_memory_reuse()
    else:
        print("Usage: python research_agent.py [continuation|memory_reuse]")
        sys.exit(1)


if __name__ == "__main__":
    main()
