"""
ReAct (Reason + Act) Agent Loop implementation in pure Python.

This module implements a production-grade, framework-free ReAct loop.
It demonstrates the core patterns of agentic systems:
1. Orchestration: Structuring system prompts to enforce reasoning steps.
2. State Management: Maintaining execution trace (short-term memory) across multi-turn API calls.
3. Execution Sandbox: Separating LLM plans from safe, deterministic python execution.

No external agentic frameworks (LangChain, LlamaIndex, etc.) are used here, 
ensuring a clear, transparent reference architecture.
"""

import os
import re
import json
from typing import List, Dict, Any, Callable

# =====================================================================
# 1. Styling Utilities for Beautiful Console Logs
# =====================================================================

def color_text(text: str, color_code: str) -> str:
    """Helper to add ANSI escape color codes for standard output."""
    colors = {
        "cyan": "\033[96m",
        "magenta": "\033[95m",
        "yellow": "\033[93m",
        "green": "\033[92m",
        "red": "\033[91m",
        "bold": "\033[1m",
        "reset": "\033[0m"
    }
    return f"{colors.get(color_code, '')}{text}{colors['reset']}"


# =====================================================================
# 2. Raw LLM Client Wrapper with Fallback Simulation
# =====================================================================

class SimpleOpenAIClient:
    """
    A clean wrapper for raw chat completion calls to OpenAI.
    Provides a fallback simulation mode if OPENAI_API_KEY is missing, 
    allowing users to run the pattern sandbox locally out-of-the-box.
    """
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.is_simulated = not self.api_key
        
        if self.is_simulated:
            print(color_text("[System Note] OPENAI_API_KEY environment variable not found. Running in simulated LLM mode.", "yellow"))
        else:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                print(color_text("[System Note] 'openai' library not installed. Falling back to simulated LLM mode.", "yellow"))
                self.is_simulated = True

    def complete(self, messages: List[Dict[str, str]]) -> str:
        """
        Retrieves completion text from the LLM or simulation.
        
        Note:
        - We set temperature=0.0 when calling real LLMs to force the model to adhere
          strictly to the structured ReAct format and avoid reasoning drift.
        """
        if self.is_simulated:
            return self._simulate(messages)
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(color_text(f"[LLM API Error] Failed to call OpenAI API: {e}. Falling back to simulation.", "red"))
            return self._simulate(messages)

    def _simulate(self, messages: List[Dict[str, str]]) -> str:
        """
        Mock LLM simulator that inspects conversation history state and mimics 
        a standard ReAct assistant response step-by-step.
        """
        # Find the user's initial prompt
        user_query = ""
        for msg in messages:
            if msg["role"] == "user" and not msg["content"].startswith("Observation:"):
                user_query = msg["content"]
                break
                
        # Collect tool outputs (observations) from conversation history
        observations = []
        for msg in messages:
            if msg["role"] == "user" and msg["content"].startswith("Observation:"):
                obs_val = msg["content"].replace("Observation:", "").strip()
                observations.append(obs_val)
                
        # Scenario 1: JFK to CDG and CDG to LHR
        if "JFK" in user_query.upper() and "CDG" in user_query.upper() and "LHR" in user_query.upper():
            if len(observations) == 0:
                return (
                    "Thought: The user wants to calculate the total flight distance from JFK to CDG, and then from CDG to LHR.\n"
                    "I need to query the distance for the first leg: JFK to CDG.\n"
                    "Action: calculate_flight_distance\n"
                    "Action Input: {\"origin_code\": \"JFK\", \"dest_code\": \"CDG\"}"
                )
            elif len(observations) == 1:
                return (
                    f"Thought: JFK to CDG is {observations[0]} miles. Now I need to find the distance for the second leg: CDG to LHR.\n"
                    "Action: calculate_flight_distance\n"
                    "Action Input: {\"origin_code\": \"CDG\", \"dest_code\": \"LHR\"}"
                )
            elif len(observations) == 2:
                try:
                    d1 = float(observations[0])
                    d2 = float(observations[1])
                    total = d1 + d2
                except ValueError:
                    d1, d2, total = 3635.0, 216.0, 3851.0
                return (
                    f"Thought: I now have the distance for both flight legs. JFK->CDG is {d1} miles and CDG->LHR is {d2} miles.\n"
                    f"The total distance is {d1} + {d2} = {total} miles.\n"
                    f"Final Answer: The total flight distance from JFK to CDG ({d1:,.0f} miles) and CDG to LHR ({d2:,.0f} miles) is {total:,.0f} miles."
                )

        # Scenario 2: Simple JFK to LAX
        if "JFK" in user_query.upper() and "LAX" in user_query.upper():
            if len(observations) == 0:
                return (
                    "Thought: The user wants the flight distance from JFK to LAX. I will call the calculate_flight_distance tool.\n"
                    "Action: calculate_flight_distance\n"
                    "Action Input: {\"origin_code\": \"JFK\", \"dest_code\": \"LAX\"}"
                )
            elif len(observations) == 1:
                return (
                    f"Thought: JFK to LAX is {observations[0]} miles. I have the complete answer.\n"
                    f"Final Answer: The flight distance from JFK to LAX is {observations[0]} miles."
                )
                
        # Default simple simulation handler
        if len(observations) == 0:
            codes = re.findall(r"\b[A-Z]{3}\b", user_query.upper())
            if len(codes) >= 2:
                return (
                    f"Thought: I detected airport codes {codes[0]} and {codes[1]}. Let me calculate the flight distance.\n"
                    f"Action: calculate_flight_distance\n"
                    f"Action Input: {{\n  \"origin_code\": \"{codes[0]}\",\n  \"dest_code\": \"{codes[1]}\"\n}}"
                )
            return (
                "Thought: The user query is general. I will fetch the distance between JFK and LHR.\n"
                "Action: calculate_flight_distance\n"
                "Action Input: {\"origin_code\": \"JFK\", \"dest_code\": \"LHR\"}"
            )
        else:
            return (
                f"Thought: I have obtained the flight distance observation: {observations[-1]} miles.\n"
                f"Final Answer: The flight distance is {observations[-1]} miles."
            )


# =====================================================================
# 3. Tool Definition & Registry
# =====================================================================

def calculate_flight_distance(origin_code: str, dest_code: str) -> float:
    """
    Mock GIS calculations.
    Returns the great-circle distance in miles between airport codes.
    """
    origin = origin_code.strip().upper()
    dest = dest_code.strip().upper()
    
    # Core distance database
    distances = {
        ("JFK", "CDG"): 3635.0,
        ("CDG", "LHR"): 216.0,
        ("LHR", "NRT"): 5974.0,
        ("JFK", "LAX"): 2475.0,
        ("LAX", "NRT"): 5451.0,
        ("JFK", "LHR"): 3451.0,
    }
    
    # Try forward and reverse pairs
    if (origin, dest) in distances:
        return distances[(origin, dest)]
    if (dest, origin) in distances:
        return distances[(dest, origin)]
        
    # Fallback to a deterministic pseudo-random distance based on string hash
    hash_val = sum(ord(c) for c in (origin + dest))
    return float(1500 + (hash_val % 4500))


# =====================================================================
# 4. Action Input Parser Helper
# =====================================================================

def parse_action_input(input_str: str) -> Dict[str, Any]:
    """
    Robust parsing engine for Action Input strings.
    Handles raw JSON, single-quoted representations, or standard key-value assignments.
    """
    input_str = input_str.strip()
    
    # 1. Clean markdown code blocks if the LLM outputted ```json ... ```
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", input_str, flags=re.MULTILINE).strip()
    
    # 2. Attempt standard JSON parsing
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
        
    # 3. Attempt to convert Python-like single quote dict to double quotes
    try:
        return json.loads(cleaned.replace("'", '"'))
    except Exception:
        pass
        
    # 4. Fallback to key-value regex extracting (e.g. key="value" or key='value')
    pairs = re.findall(r"(\w+)\s*=\s*['\"]([^'\"]+)['\"]", input_str)
    if pairs:
        return {k: v for k, v in pairs}
        
    # 5. Fallback to comma-separated list
    parts = [p.strip().replace('"', '').replace("'", "") for p in input_str.split(",")]
    if len(parts) == 2:
        return {"origin_code": parts[0], "dest_code": parts[1]}
        
    return {"raw_input": input_str}


# =====================================================================
# 5. ReAct Orchestrator & Execution Loop
# =====================================================================

class ReActAgent:
    """
    The orchestrator implementing the ReAct (Reason + Act) loop pattern.
    Manages the conversational trace as short-term state, parses plans, 
    and executes external tools.
    """
    def __init__(self, system_prompt: str, tools: Dict[str, Callable], client: SimpleOpenAIClient):
        self.system_prompt = system_prompt
        self.tools = tools
        self.client = client
        self.history: List[Dict[str, str]] = []
        
    def reset(self):
        """Resets agent conversation history back to the base system prompt."""
        self.history = [{"role": "system", "content": self.system_prompt}]
        
    def run(self, user_query: str, max_iterations: int = 5) -> str:
        """
        Executes the main ReAct loop.
        
        State Management Details:
        -------------------------
        - LLMs are completely stateless. They have no built-in retention of previous steps.
        - To simulate 'thinking' over multiple turns, the execution engine acts as the state manager.
        - We maintain a chronological list of messages in self.history.
        - LLM thoughts and action calls are appended as 'assistant' messages.
        - Execution outputs (observations) are injected back into the thread as 'user' observations.
        - The next execution step feeds the entire historical chain back into the LLM, enabling the 
          model to perform in-context learning and build upon its past steps.
        """
        self.reset()
        self.history.append({"role": "user", "content": user_query})
        
        print(color_text(f"\n[🚀 Starting ReAct Agent Loop for query: '{user_query}']", "bold"))
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            print(color_text(f"\n--- Iteration {iteration} ---", "cyan"))
            
            # 1. Ask the stateless LLM client what to do next based on the full conversation state
            response_text = self.client.complete(self.history)
            
            # Log the thoughts and actions outputted by the LLM
            self._print_llm_response(response_text)
            
            # 2. Save the LLM reasoning/planning output in our state history
            self.history.append({"role": "assistant", "content": response_text})
            
            # Check for termination condition
            if "Final Answer:" in response_text:
                final_answer_match = re.search(r"Final Answer:\s*(.+)", response_text, re.DOTALL)
                if final_answer_match:
                    return final_answer_match.group(1).strip()
                return response_text.split("Final Answer:")[-1].strip()
                
            # 3. Parse action to execute
            action_match = re.search(r"Action:\s*(\w+)", response_text)
            action_input_match = re.search(r"Action Input:\s*(.+)", response_text, re.DOTALL)
            
            if not action_match or not action_input_match:
                # Handle edge case where LLM fails to structure output or gets stuck
                print(color_text("[⚠️ Format Error] Failed to parse Action/Action Input. Requesting formatting retry.", "red"))
                retry_instruction = (
                    "Error: Your response must include an 'Action:' and 'Action Input:' block to execute a tool, "
                    "or a 'Final Answer:' block to reply to the user. Please try again."
                )
                self.history.append({"role": "user", "content": retry_instruction})
                continue
                
            tool_name = action_match.group(1).strip()
            raw_input_str = action_input_match.group(1).strip()
            
            # Parse parameters into dictionary arguments
            args = parse_action_input(raw_input_str)
            
            # 4. Execute the tool in the safe registry environment
            if tool_name in self.tools:
                print(color_text(f"[⚙️ Executing Tool] {tool_name} with params: {args}", "yellow"))
                try:
                    tool_func = self.tools[tool_name]
                    # Attempt keyword argument unpacking. Fallback to positional if needed.
                    if "raw_input" in args and len(args) == 1:
                        observation = tool_func(args["raw_input"])
                    else:
                        observation = tool_func(**args)
                    obs_str = f"Observation: {observation}"
                except TypeError as te:
                    print(color_text(f"[❌ Parameters Mismatch] Tool arguments error: {te}", "red"))
                    obs_str = f"Observation: Error - Argument mismatch when calling tool {tool_name}. Details: {te}"
                except Exception as e:
                    print(color_text(f"[❌ Execution Error] Tool failed: {e}", "red"))
                    obs_str = f"Observation: Error executing tool {tool_name}: {str(e)}"
            else:
                print(color_text(f"[⚠️ Registry Error] Tool '{tool_name}' not found.", "red"))
                obs_str = f"Observation: Error - Tool '{tool_name}' is not registered. Available: {list(self.tools.keys())}"
                
            print(color_text(f"[📥 Result] {obs_str}", "green"))
            
            # 5. Append the observation output back to the state history.
            # This completes one reasoning loop step. The next loop iteration will pass
            # this observation to the LLM, prompting it to reason (Thought:) about the output.
            self.history.append({"role": "user", "content": obs_str})
            
        print(color_text(f"\n[❌ Iteration Limit] Reached max steps ({max_iterations}) without resolving.", "red"))
        return "Failure: Max iteration limit exceeded."
        
    def _print_llm_response(self, text: str):
        """Beautify the multi-line LLM response text in the terminal."""
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("Thought:"):
                print(color_text(line, "magenta"))
            elif line.startswith("Action:"):
                print(color_text(line, "cyan"))
            elif line.startswith("Action Input:"):
                print(color_text(line, "cyan"))
            elif line.startswith("Final Answer:"):
                print(color_text(line, "bold"))
            else:
                print(line)


# =====================================================================
# 6. Execution Block / Sandbox Demo
# =====================================================================

if __name__ == "__main__":
    # Create the ReAct system prompt explaining how to format and which tools are available
    tools_list = {
        "calculate_flight_distance": calculate_flight_distance
    }
    
    tool_descriptions_prompt = (
        "- calculate_flight_distance(origin_code: str, dest_code: str) -> float\n"
        "  Calculates the great-circle distance in miles between two three-letter airport codes."
    )
    
    from jinja2 import Template
    # We define the strict system instructions here
    react_instructions = (
        "You are an AI research assistant capable of solving queries by using tools step-by-step.\n"
        "You must strictly follow the ReAct (Reason + Act) loop pattern.\n\n"
        "You have access to the following tools:\n"
        f"{tool_descriptions_prompt}\n\n"
        "At each step, you must think about what to do next and respond in the exact format:\n\n"
        "Thought: <reasoning about the next step>\n"
        "Action: <tool_name_only>\n"
        "Action Input: <tool_arguments_as_json_or_key_value>\n\n"
        "After you output Action and Action Input, the execution engine will run the tool and return the output in the format:\n"
        "Observation: <tool_output>\n\n"
        "You must repeat this sequence until you have collected enough information to solve the query.\n"
        "When you are ready to answer the user, you must output:\n\n"
        "Thought: <final reasoning summarizing the findings>\n"
        "Final Answer: <your complete response to the user>\n\n"
        "Rules:\n"
        "1. Every turn must start with 'Thought:'.\n"
        "2. You must output 'Action:' and 'Action Input:' ONLY if you need to run a tool.\n"
        "3. You must not invent observations; wait for the execution engine to supply them.\n"
        "4. When you have the final answer, do NOT output 'Action:' or 'Action Input:'. Instead, output 'Final Answer:'.\n"
        "5. Output only one Action and Action Input per turn. Do not stack multiple actions."
    )
    
    # Initialize components
    client = SimpleOpenAIClient()
    agent = ReActAgent(system_prompt=react_instructions, tools=tools_list, client=client)
    
    # Test Scenario: Flight distance JFK -> CDG -> LHR
    query = "What is the total flight distance from JFK to CDG and then CDG to LHR?"
    final_ans = agent.run(query)
    
    print(color_text(f"\n[🏁 Agent Executed Successfully!]", "bold"))
    print(color_text(f"Final Answer:\n{final_ans}", "green"))
