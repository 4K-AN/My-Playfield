"""
Tests untuk Agent Framework
"""

import unittest
import sys
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import Agent
from src.memory import Memory
from src.tools import ToolRegistry, ToolExecutor, create_default_registry
from src.parser import OutputParser
from src.models import Plan, ExecutionStep


class TestAgent(unittest.TestCase):
    """Test Agent class"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.agent = Agent(
            model="gpt-4",
            memory_path=self.temp_dir,
            max_iterations=3,
            use_persistent_memory=True
        )
    
    def tearDown(self):
        """Cleanup"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_agent_initialization(self):
        """Test agent initializes correctly"""
        self.assertIsNotNone(self.agent)
        self.assertEqual(self.agent.model, "gpt-4")
        self.assertEqual(self.agent.max_iterations, 3)
    
    def test_add_tool(self):
        """Test adding custom tool"""
        def dummy_tool():
            return "result"
        
        self.agent.add_tool(
            name="dummy",
            implementation=dummy_tool,
            description="Dummy tool for testing"
        )
        
        self.assertIsNotNone(self.agent.tool_registry.get_tool("dummy"))
    
    def test_tool_registry(self):
        """Test tool registry"""
        registry = create_default_registry()
        
        tools = registry.list_tools()
        self.assertTrue(len(tools) > 0)
        self.assertTrue(any(t.name == "search" for t in tools))
    
    def test_context_building(self):
        """Test context building"""
        self.agent.memory.start_session("Test task")
        
        context = self.agent._observe()
        
        self.assertIn("task", context)
        self.assertIn("iteration", context)
        self.assertIn("available_tools", context)


class TestMemory(unittest.TestCase):
    """Test Memory system"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.memory = Memory(
            memory_path=self.temp_dir,
            use_persistent=True
        )
    
    def tearDown(self):
        """Cleanup"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_memory_initialization(self):
        """Test memory initializes"""
        self.assertIsNotNone(self.memory)
        self.assertIsNotNone(self.memory.in_memory)
    
    def test_session_start(self):
        """Test session start"""
        self.memory.start_session("Test task")
        
        self.assertIsNotNone(self.memory.session_id)
        self.assertIsNotNone(self.memory.current_session)
        self.assertEqual(self.memory.current_session.task, "Test task")
    
    def test_add_message(self):
        """Test adding message"""
        self.memory.start_session("Test")
        self.memory.add_message("user", "Hello")
        
        self.assertEqual(len(self.memory.conversation_history), 1)
        self.assertEqual(self.memory.conversation_history[0].role, "user")
    
    def test_add_fact(self):
        """Test adding facts"""
        self.memory.start_session("Test")
        self.memory.add_fact("key1", "value1", category="fact")
        
        facts = self.memory.get_facts()
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["key"], "key1")
    
    def test_save_and_load_session(self):
        """Test session persistence"""
        self.memory.start_session("Test task", session_id="test_001")
        self.memory.add_message("user", "Test message")
        self.memory.save_session()
        
        # Create new memory and load session
        memory2 = Memory(memory_path=self.temp_dir, use_persistent=True)
        memory2.start_session("Any", session_id="test_001")
        
        # Should have loaded messages
        self.assertTrue(len(memory2.conversation_history) > 0)


class TestToolExecutor(unittest.TestCase):
    """Test Tool Executor"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.registry = create_default_registry()
        self.executor = ToolExecutor(self.registry)
    
    def test_execute_tool(self):
        """Test tool execution"""
        result = self.executor.execute("search", {"query": "test"})
        
        self.assertTrue(result.success)
        self.assertEqual(result.tool, "search")
        self.assertIsNotNone(result.output)
    
    def test_invalid_tool(self):
        """Test invalid tool"""
        result = self.executor.execute("nonexistent", {})
        
        self.assertFalse(result.success)
        self.assertIn("not found", result.error.lower())
    
    def test_execution_history(self):
        """Test execution history"""
        self.executor.execute("search", {"query": "test"})
        self.executor.execute("calculate", {"expression": "2+2"})
        
        history = self.executor.get_execution_history()
        self.assertEqual(len(history), 2)


class TestParser(unittest.TestCase):
    """Test Output Parser"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.parser = OutputParser(strict=False)
    
    def test_parse_valid_json(self):
        """Test parsing valid JSON"""
        json_text = '{"key": "value", "number": 42}'
        result = self.parser.parse_json(json_text)
        
        self.assertEqual(result["key"], "value")
        self.assertEqual(result["number"], 42)
    
    def test_parse_json_with_markdown(self):
        """Test parsing JSON from markdown"""
        text = """
Here's the result:
```json
{
    "status": "success",
    "data": "test"
}
```
"""
        result = self.parser.parse_json(text)
        
        self.assertEqual(result["status"], "success")
    
    def test_parse_json_with_text(self):
        """Test parsing JSON mixed with text"""
        text = "The result is: {\"answer\": 42}"
        result = self.parser.parse_json(text)
        
        self.assertEqual(result["answer"], 42)
    
    def test_parse_invalid_json(self):
        """Test parsing invalid JSON"""
        with self.assertRaises(Exception):
            self.parser.parse_json("not json at all!")


class TestModels(unittest.TestCase):
    """Test data models"""
    
    def test_plan_model(self):
        """Test Plan model"""
        step = ExecutionStep(
            step=1,
            tool="search",
            params={"query": "test"},
            description="Search test"
        )
        
        plan = Plan(
            reasoning="Test reasoning",
            next_steps=[step],
            expected_outcome="Test outcome"
        )
        
        self.assertEqual(len(plan.next_steps), 1)
        self.assertEqual(plan.next_steps[0].tool, "search")


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestMemory))
    suite.addTests(loader.loadTestsFromTestCase(TestToolExecutor))
    suite.addTests(loader.loadTestsFromTestCase(TestParser))
    suite.addTests(loader.loadTestsFromTestCase(TestModels))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
