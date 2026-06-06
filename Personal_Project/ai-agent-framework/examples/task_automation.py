"""
EXAMPLE 4: Task Automation Agent
Menunjukkan: practical task automation, combining multiple services
"""

import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import Agent
from src.tools import ToolRegistry, ToolParameter
from src.utils import setup_logging, print_result


# Task automation tools

def read_email(email_id: str) -> dict:
    """Read email dari inbox"""
    return {
        "email_id": email_id,
        "from": "customer@example.com",
        "subject": "Question about product",
        "body": "I have a question about your AI agent product...",
        "timestamp": "2024-01-15T10:30:00Z"
    }


def classify_email(subject: str, body: str) -> dict:
    """Classify email sebagai support, sales, feedback, etc"""
    
    keywords = {
        "support": ["help", "problem", "bug", "error", "not working"],
        "sales": ["pricing", "buy", "enterprise", "license"],
        "feedback": ["good", "great", "suggestion", "feature"]
    }
    
    text = (subject + " " + body).lower()
    
    for category, words in keywords.items():
        if any(word in text for word in words):
            return {"category": category, "confidence": 0.85}
    
    return {"category": "general", "confidence": 0.5}


def send_response(recipient: str, subject: str, body: str) -> dict:
    """Send email response"""
    return {
        "sent_to": recipient,
        "subject": subject,
        "status": "sent",
        "message_id": f"msg_{hash(body) % 10000}"
    }


def create_ticket(title: str, description: str, priority: str = "normal") -> dict:
    """Create support ticket"""
    return {
        "ticket_id": f"TKT-{hash(title) % 100000}",
        "title": title,
        "priority": priority,
        "status": "created",
        "assigned_to": "support_team"
    }


def fetch_kb_article(query: str) -> dict:
    """Fetch dari knowledge base"""
    return {
        "query": query,
        "articles": [
            {
                "title": f"How to use {query}",
                "content": f"This article explains how to use {query}...",
                "relevance": 0.9
            }
        ],
        "count": 1
    }


def add_note_to_crm(contact_id: str, note: str) -> dict:
    """Add note to customer CRM record"""
    return {
        "contact_id": contact_id,
        "note_id": f"note_{hash(note) % 10000}",
        "note": note,
        "status": "saved"
    }


def schedule_callback(contact_id: str, time_slot: str) -> dict:
    """Schedule callback dengan customer"""
    return {
        "contact_id": contact_id,
        "scheduled_time": time_slot,
        "status": "scheduled",
        "confirmation_sent": True
    }


def create_automation_agent() -> Agent:
    """Create agent untuk task automation"""
    
    registry = ToolRegistry()
    
    # Email tools
    registry.register(
        name="read_email",
        description="Read email dari inbox",
        implementation=read_email,
        parameters=[
            ToolParameter(
                name="email_id",
                type="string",
                required=True,
                description="Email ID to read"
            )
        ]
    )
    
    registry.register(
        name="classify_email",
        description="Classify email category",
        implementation=classify_email,
        parameters=[
            ToolParameter(
                name="subject",
                type="string",
                required=True,
                description="Email subject"
            ),
            ToolParameter(
                name="body",
                type="string",
                required=True,
                description="Email body"
            )
        ]
    )
    
    registry.register(
        name="send_response",
        description="Send email response",
        implementation=send_response,
        parameters=[
            ToolParameter(
                name="recipient",
                type="string",
                required=True,
                description="Recipient email"
            ),
            ToolParameter(
                name="subject",
                type="string",
                required=True,
                description="Response subject"
            ),
            ToolParameter(
                name="body",
                type="string",
                required=True,
                description="Response body"
            )
        ]
    )
    
    # Ticketing tools
    registry.register(
        name="create_ticket",
        description="Create support ticket",
        implementation=create_ticket,
        parameters=[
            ToolParameter(
                name="title",
                type="string",
                required=True,
                description="Ticket title"
            ),
            ToolParameter(
                name="description",
                type="string",
                required=True,
                description="Ticket description"
            ),
            ToolParameter(
                name="priority",
                type="string",
                required=False,
                description="Priority: low, normal, high"
            )
        ]
    )
    
    # Knowledge base
    registry.register(
        name="search_kb",
        description="Search knowledge base",
        implementation=fetch_kb_article,
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                required=True,
                description="Search query"
            )
        ]
    )
    
    # CRM tools
    registry.register(
        name="add_crm_note",
        description="Add note to CRM",
        implementation=add_note_to_crm,
        parameters=[
            ToolParameter(
                name="contact_id",
                type="string",
                required=True,
                description="Contact ID"
            ),
            ToolParameter(
                name="note",
                type="string",
                required=True,
                description="Note to add"
            )
        ]
    )
    
    registry.register(
        name="schedule_callback",
        description="Schedule customer callback",
        implementation=schedule_callback,
        parameters=[
            ToolParameter(
                name="contact_id",
                type="string",
                required=True,
                description="Contact ID"
            ),
            ToolParameter(
                name="time_slot",
                type="string",
                required=True,
                description="Time slot for callback"
            )
        ]
    )
    
    # Create agent
    agent = Agent(
        model="gpt-4",
        memory_path="./memory/automation",
        max_iterations=8,
        token_budget=3000,
        use_persistent_memory=True,
        tool_registry=registry
    )
    
    return agent


def demonstrate_email_triage():
    """
    Demonstrate email triage workflow
    """
    
    print("\n" + "="*60)
    print("TASK AUTOMATION - EMAIL TRIAGE")
    print("="*60)
    
    logger = setup_logging(level="INFO")
    
    agent = create_automation_agent()
    
    task = """
    Automate email triage workflow:
    1. Read incoming email
    2. Classify email category (support, sales, feedback)
    3. Based on category:
       - If support: search KB, create ticket if needed
       - If sales: add note to CRM, schedule callback
       - If feedback: add note to CRM
    4. Send appropriate response
    5. Log actions untuk audit trail
    """
    
    result = agent.run(
        task=task,
        session_id="email_triage_automation",
        max_retries=2
    )
    
    print_result(result, verbose=True)
    
    # Show workflow execution
    print("\nWorkflow Execution Chain:")
    history = agent.tool_executor.execution_history
    
    for i, execution in enumerate(history, 1):
        status = "✓" if execution.success else "✗"
        print(f"  {i}. {status} {execution.tool} ({execution.duration_seconds:.2f}s)")
    
    return result


def demonstrate_data_processing():
    """
    Demonstrate data processing automation
    """
    
    print("\n" + "="*60)
    print("TASK AUTOMATION - DATA PROCESSING")
    print("="*60)
    
    logger = setup_logging(level="INFO")
    
    agent = create_automation_agent()
    
    task = """
    Automate data processing workflow:
    1. Read customer emails batch
    2. Classify each email
    3. Extract key information
    4. Create/update CRM records
    5. Generate daily report
    
    Optimize untuk speed dan accuracy.
    """
    
    result = agent.run(
        task=task,
        session_id="data_processing_automation",
        max_retries=2
    )
    
    print_result(result, verbose=True)
    
    # Show metrics
    print("\nAutomation Metrics:")
    history = agent.tool_executor.execution_history
    
    total_time = sum(ex.duration_seconds for ex in history)
    success_count = sum(1 for ex in history if ex.success)
    
    print(f"  Total operations: {len(history)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {len(history) - success_count}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average per operation: {(total_time/len(history) if history else 0):.2f}s")
    
    return result


def demonstrate_custom_workflow():
    """
    Demonstrate custom workflow automation
    """
    
    print("\n" + "="*60)
    print("TASK AUTOMATION - CUSTOM WORKFLOW")
    print("="*60)
    
    logger = setup_logging(level="INFO")
    
    agent = create_automation_agent()
    
    # Manually add facts untuk customize behavior
    agent.memory.add_fact(
        key="sla_response_time",
        value="2 hours",
        category="policy"
    )
    
    agent.memory.add_fact(
        key="priority_escalation_threshold",
        value=["urgent", "critical"],
        category="policy",
        relevance_score=0.95
    )
    
    task = """
    Process high-priority customer request:
    1. Read and classify email
    2. Apply SLA policies
    3. Escalate if necessary
    4. Create ticket with appropriate priority
    5. Send immediate acknowledgment
    6. Schedule follow-up
    
    Follow internal policies untuk handling.
    """
    
    result = agent.run(
        task=task,
        session_id="custom_workflow",
        max_retries=2
    )
    
    print_result(result, verbose=True)
    
    print("\nPolicy Context Used:")
    print(agent.memory.get_context_summary())
    
    return result


def main():
    """Main demo function"""
    
    demo_type = sys.argv[1] if len(sys.argv) > 1 else "triage"
    
    if demo_type == "triage":
        return demonstrate_email_triage()
    elif demo_type == "processing":
        return demonstrate_data_processing()
    elif demo_type == "custom":
        return demonstrate_custom_workflow()
    else:
        print("Usage: python task_automation.py [triage|processing|custom]")
        print("\nExamples:")
        print("  python task_automation.py triage     # Email triage automation")
        print("  python task_automation.py processing  # Data processing automation")
        print("  python task_automation.py custom      # Custom workflow automation")
        sys.exit(1)


if __name__ == "__main__":
    main()
