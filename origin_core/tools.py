"""
Tool Registry module for defining and executing typed agent tools.
"""

import inspect
from typing import Callable, Dict, Any, List
from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    """Metadata and execution function wrapper for an agent tool."""
    name: str
    description: str
    func: Callable
    parameters_schema: Dict[str, Any] = Field(default_factory=dict)


class ToolRegistry:
    """Registry managing available tools for Origin-Core AI Agents."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, name: str, description: str):
        """Decorator registering a function as an agent tool."""
        def decorator(func: Callable):
            sig = inspect.signature(func)
            schema = {param.name: str(param.annotation) for param in sig.parameters.values()}
            
            tool_def = ToolDefinition(
                name=name,
                description=description,
                func=func,
                parameters_schema=schema
            )
            self._tools[name] = tool_def
            return func
        return decorator

    def execute(self, tool_name: str, **kwargs) -> Any:
        """Executes a registered tool with keyword arguments."""
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' is not registered in ToolRegistry.")
        return self._tools[tool_name].func(**kwargs)

    def list_tools(self) -> List[Dict[str, str]]:
        """Lists registered tools for LLM system prompts."""
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools.values()
        ]
