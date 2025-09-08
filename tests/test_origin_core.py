"""
Unit tests for Origin-Core micro-kernel agent framework.
"""

import unittest
from origin_core.tools import ToolRegistry
from origin_core.memory import AgentMemory
from origin_core.agent import OriginAgent


class TestOriginCore(unittest.TestCase):

    def test_tool_registry(self):
        registry = ToolRegistry()
        
        @registry.register(name="add", description="Adds two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        tools = registry.list_tools()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "add")

        res = registry.execute("add", a=5, b=10)
        self.assertEqual(res, 15)

    def test_agent_memory(self):
        memory = AgentMemory(max_turns=3)
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi there")
        memory.add_message("user", "How are you?")
        memory.add_message("assistant", "I am well")

        self.assertLessEqual(len(memory.get_messages()), 3)

    def test_origin_agent_execution(self):
        registry = ToolRegistry()
        @registry.register(name="search", description="Search web")
        def search(query: str) -> str:
            return "Search results for " + query

        agent = OriginAgent(system_prompt="You are a helpful assistant", tools=registry)
        result = agent.run("search for python news")

        self.assertTrue(result.success)
        self.assertGreaterEqual(result.steps_executed, 1)


if __name__ == "__main__":
    unittest.main()
