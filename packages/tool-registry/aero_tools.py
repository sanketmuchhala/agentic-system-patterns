"""
Modular Tool Registry and strict tool interfaces for aero-routing.

This module implements a framework-free tool registry system following the 
BaseTool interface design. Crucially, it adopts an 'Agentic Self-Correction' 
error design pattern, returning execution failures as string observations rather 
than crashing the agent thread.
"""

import sys
import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List

# Add packages/graph-memory to system path to import FlightGraph
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
graph_memory_dir = os.path.join(root_dir, "packages", "graph-memory")
if graph_memory_dir not in sys.path:
    sys.path.append(graph_memory_dir)

try:
    from airline_schema import FlightGraph
except ImportError:
    # Fallback to local import if the workspace is structured differently in execution env
    sys.path.append(os.path.join(root_dir, "packages"))
    from graph_memory.airline_schema import FlightGraph

# =====================================================================
# 1. Base Tool Interface
# =====================================================================

class BaseTool(ABC):
    """
    Abstract base class for all agentic tools.
    Enforces a standard schema interface (name and description) so that 
    the orchestrator can accurately present execution options to the LLM.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """The tool identifier used in the ReAct 'Action:' line."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Verbose description detailing arguments and outputs for the LLM prompt."""
        pass

    @abstractmethod
    def run(self, **kwargs) -> str:
        """Executes the tool's core logic. Must return a string observation."""
        pass


# =====================================================================
# 2. Tool 1: query_hub_graph
# =====================================================================

class QueryHubGraphTool(BaseTool):
    """
    Tool that interfaces with the persistent FlightGraph memory layer.
    Extracts local subgraphs to feed search spaces to the LLM.
    """
    @property
    def name(self) -> str:
        return "query_hub_graph"

    @property
    def description(self) -> str:
        return (
            "query_hub_graph(start_airport: str, max_hops: int = 2) -> str\n"
            "Queries the persistent Graph Memory layer for all airports and connecting flights "
            "reachable within `max_hops` from the `start_airport`.\n"
            "Arguments:\n"
            "  - start_airport: The 3-letter IATA code of the starting airport (e.g., 'DCA'). [Required]\n"
            "  - max_hops: Search depth limit (default=2). [Optional]"
        )

    def run(self, **kwargs) -> str:
        # Support multiple query field variations
        start = kwargs.get("start_airport") or kwargs.get("origin") or kwargs.get("origin_code")
        if not start:
            raise ValueError("Missing required parameter: 'start_airport'")
            
        max_hops = kwargs.get("max_hops", 2)
        try:
            max_hops = int(max_hops)
        except (ValueError, TypeError):
            raise TypeError("Parameter 'max_hops' must be an integer.")
            
        # Instantiate FlightGraph, load flight networks, and query subgraph
        fg = FlightGraph()
        fg.seed_data()
        return fg.get_local_subgraph(start, max_hops=max_hops)


# =====================================================================
# 3. Tool 2: calculate_layover_penalty
# =====================================================================

class CalculateLayoverPenaltyTool(BaseTool):
    """
    A mathematical utility tool that calculates travel fatigue scores 
    and layover cost overhead.
    """
    @property
    def name(self) -> str:
        return "calculate_layover_penalty"

    @property
    def description(self) -> str:
        return (
            "calculate_layover_penalty(flight_durations: list[int], layover_durations: list[int]) -> str\n"
            "Calculates total fatigue and monetary cost penalty metrics for multi-leg travel.\n"
            "Arguments:\n"
            "  - flight_durations: A JSON-style list of leg flight times in minutes. [Required]\n"
            "  - layover_durations: A JSON-style list of connection layover times in minutes. [Required]"
        )

    def run(self, **kwargs) -> str:
        flight_durations = kwargs.get("flight_durations")
        layover_durations = kwargs.get("layover_durations")
        
        if flight_durations is None or layover_durations is None:
            raise ValueError("Both 'flight_durations' and 'layover_durations' are required parameters.")
            
        # Coerce strings (e.g. from JSON output format or LLM formatting errors) to Python lists
        if isinstance(flight_durations, str):
            try:
                flight_durations = json.loads(flight_durations)
            except Exception:
                raise TypeError("'flight_durations' must be a valid JSON list format.")
        if isinstance(layover_durations, str):
            try:
                layover_durations = json.loads(layover_durations)
            except Exception:
                raise TypeError("'layover_durations' must be a valid JSON list format.")
                
        if not isinstance(flight_durations, list) or not isinstance(layover_durations, list):
            raise TypeError("Arguments must be lists of durations.")
            
        # Convert items to float and calculate fatigue
        try:
            flights = [float(x) for x in flight_durations]
            layovers = [float(x) for x in layover_durations]
        except (ValueError, TypeError):
            raise TypeError("List elements must be valid numerical values.")
            
        total_flight = sum(flights)
        total_layover = sum(layovers)
        num_layovers = len(layovers)
        
        # System Calculation:
        # - Fatigue = (Flight duration * 0.1) + (Layover duration * 0.15) + ($50 penalty per layover)
        # - Cost Penalty = ($25 flat fee per layover) + ($0.05 surcharge per layover minute)
        fatigue = (total_flight * 0.1) + (total_layover * 0.15) + (num_layovers * 50.0)
        cost_penalty = (num_layovers * 25.0) + (total_layover * 0.05)
        
        result = {
            "total_flight_mins": total_flight,
            "total_layover_mins": total_layover,
            "layovers_count": num_layovers,
            "fatigue_score": round(fatigue, 2),
            "cost_penalty_usd": round(cost_penalty, 2)
        }
        
        return json.dumps(result, indent=2)


# =====================================================================
# 4. Tool Registry Manager with Agentic Self-Correction
# =====================================================================

class ToolRegistry:
    """
    A safety registry that holds and runs tools.
    
    System Design & Agentic Self-Correction:
    ----------------------------------------
    - When building autonomous LLM runtimes, raising standard python exceptions 
      for syntax errors, parameter mismatches, or invalid arguments will crash the 
      entire application process.
    - Instead of throwing a traceback, the execute_tool method acts as an exception 
      boundary. It catches all errors and translates them into plain text observations.
    - Example: 'Error: Tool execution failed - ValueError: Missing required parameter...'
    - This error string is fed directly back into the ReAct loop as an 'Observation:'.
    - The LLM receives the error text in its context, understands what parameter it 
      got wrong, automatically corrects its plan, and tries again.
    """
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register_tool(self, tool: BaseTool):
        """Registers a tool inside the orchestrator sandbox."""
        self._tools[tool.name] = tool
        
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """
        Executes a registered tool by name with keyword parameters.
        Protects the agent process from crashes, supporting Agentic Self-Correction.
        """
        if tool_name not in self._tools:
            return (
                f"Error: Tool '{tool_name}' is not registered.\n"
                f"Available tools: {list(self._tools.keys())}"
            )
            
        try:
            tool = self._tools[tool_name]
            return tool.run(**kwargs)
        except Exception as e:
            # Shield the orchestrator from traceback exceptions.
            # Format and return the error message for in-context LLM self-correction.
            return f"Error: Tool '{tool_name}' execution failed - {type(e).__name__}: {str(e)}"

    def get_tool_descriptions(self) -> str:
        """Formats all registered tool names and usage guidelines for prompt injection."""
        descriptions = []
        for name, tool in self._tools.items():
            descriptions.append(f"Tool Name: {name}\nDescription:\n{tool.description}")
        return "\n\n---\n\n".join(descriptions)


# =====================================================================
# Manual Verification Block
# =====================================================================
if __name__ == "__main__":
    # Style helper
    def cyan(t): return f"\033[96m{t}\033[0m"
    def green(t): return f"\033[92m{t}\033[0m"
    def red(t): return f"\033[91m{t}\033[0m"

    print(cyan("\n[1] Initializing ToolRegistry and registering tools..."))
    registry = ToolRegistry()
    registry.register_tool(QueryHubGraphTool())
    registry.register_tool(CalculateLayoverPenaltyTool())
    print(green("Tools registered successfully!"))
    print("\n" + registry.get_tool_descriptions())

    print(cyan("\n[2] Executing query_hub_graph tool with valid parameters..."))
    res = registry.execute_tool("query_hub_graph", start_airport="DCA", max_hops=1)
    print(green("Result:"))
    print(res)

    print(cyan("\n[3] Executing calculate_layover_penalty with valid parameters..."))
    res2 = registry.execute_tool(
        "calculate_layover_penalty", 
        flight_durations=[120, 270], 
        layover_durations=[60]
    )
    print(green("Result:"))
    print(res2)

    print(cyan("\n[4] Triggering Agentic Self-Correction (Passing invalid parameters)..."))
    err_res = registry.execute_tool("query_hub_graph", wrong_param="DCA")
    print(red("Returned Observation (Safe Error Capture):"))
    print(err_res)
