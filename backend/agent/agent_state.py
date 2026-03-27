"""
Agent state management for agentic AI system
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class AgentState:
    """Manages state throughout the agentic reasoning loop"""
    query: str
    conversation_history: List[Dict] = field(default_factory=list)
    gathered_info: Dict[str, Any] = field(default_factory=dict)  # Tool results by tool name
    tools_used: List[str] = field(default_factory=list)
    reasoning_history: List[str] = field(default_factory=list)
    current_strategy: Optional[Dict] = None
    iteration_count: int = 0
    is_complete: bool = False
    
    def update(self, tool_name: str, results: Dict[str, Any], reasoning: str = ""):
        """Update state after tool execution"""
        self.tools_used.append(tool_name)
        self.gathered_info[tool_name] = results
        if reasoning:
            self.reasoning_history.append(reasoning)
        self.iteration_count += 1
    
    def get_summary(self) -> str:
        """Get summary of gathered information"""
        summary = []
        summary.append(f"Query: {self.query}")
        summary.append(f"Iterations: {self.iteration_count}")
        summary.append(f"Tools used: {', '.join(self.tools_used)}")
        for tool, results in self.gathered_info.items():
            if isinstance(results, dict):
                count = results.get('count', len(results.get('results', [])))
                summary.append(f"{tool}: {count} results")
        return "\n".join(summary)
    
    def has_info_from(self, tool_name: str) -> bool:
        """Check if we have information from a specific tool"""
        return tool_name in self.gathered_info and bool(self.gathered_info.get(tool_name))

