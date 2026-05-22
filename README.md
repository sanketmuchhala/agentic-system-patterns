# Agentic System Patterns 🌐

A reference implementation and practice sandbox for building production-ready, autonomous AI agents. This repository focuses on separating reasoning logic from state management, advanced retrieval (GraphRAG), and scalable system design.

**Author:** Sanket Muchhala

## 🎯 Core Principles

1. **Separation of Concerns:** LLM orchestration is strictly isolated from tool execution and memory storage.
2. **Persistent Graph Memory:** Utilizing graph databases to manage agent state, map complex relationships, and drastically reduce redundant token usage.
3. **Docs as Code:** System architecture and agent flows are documented natively using Mermaid.js.

## 🏗 System Architecture

The core pattern relies on an iterative reflection loop. The agent observes its environment, plans a sequence of actions, queries the tool registry, and updates its graph memory.

```mermaid
sequenceDiagram
    participant User
    participant RouterAgent
    participant GraphMemory
    participant ToolRegistry

    User->>RouterAgent: Submit complex mission
    loop Reflection & Execution
        RouterAgent->>GraphMemory: Query past context (GraphRAG)
        GraphMemory-->>RouterAgent: Return sub-graph context
        RouterAgent->>ToolRegistry: Execute specialized tool
        ToolRegistry-->>RouterAgent: Return tool output
        RouterAgent->>GraphMemory: Write new state/findings
    end
    RouterAgent->>User: Deliver final objective