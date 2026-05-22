"""
GraphRAG memory layer for agentic airline routing.

This module implements `FlightGraph` using the `networkx` library to model
airports (nodes) and flights (edges) in a directed graph structure.
It highlights the core GraphRAG principle of local subgraph extraction
for state management and prompt construction.
"""

import json
from typing import Dict, Any, List
import networkx as nx

class FlightGraph:
    """
    Manages a directed graph of airports and connecting flights.
    Provides utility methods to seed data and retrieve local subgraphs.
    """
    def __init__(self):
        # Initialize a directed graph (nx.DiGraph) because flights are directional
        self.graph = nx.DiGraph()

    def seed_data(self):
        """
        Seeds the graph with major US hubs and flights between them.
        Crucially:
        - Establishes no direct flights from DCA to SEA.
        - Forces agents to discover layovers through either ORD, DFW, or ATL.
        """
        self.graph.clear()
        
        # 1. Add Nodes (Airports) with attributes
        airports = [
            ("DCA", {"airport_code": "DCA", "city": "Washington D.C.", "hub_status": "Regional Hub"}),
            ("SEA", {"airport_code": "SEA", "city": "Seattle", "hub_status": "Major Hub"}),
            ("ORD", {"airport_code": "ORD", "city": "Chicago", "hub_status": "Mega Hub"}),
            ("DFW", {"airport_code": "DFW", "city": "Dallas-Fort Worth", "hub_status": "Mega Hub"}),
            ("ATL", {"airport_code": "ATL", "city": "Atlanta", "hub_status": "Mega Hub"}),
        ]
        self.graph.add_nodes_from(airports)
        
        # 2. Add Edges (Flights) with attributes
        # Notice there is no direct edge from DCA to SEA.
        flights = [
            # Flights originating from DCA (our starting point)
            ("DCA", "ORD", {"carrier": "American Airlines", "cost": 150.0, "duration_mins": 120}),
            ("DCA", "DFW", {"carrier": "American Airlines", "cost": 180.0, "duration_mins": 190}),
            ("DCA", "ATL", {"carrier": "Delta Air Lines", "cost": 130.0, "duration_mins": 100}),
            
            # Flights connecting ORD to SEA and other hubs
            ("ORD", "SEA", {"carrier": "United Airlines", "cost": 250.0, "duration_mins": 270}),
            ("ORD", "DFW", {"carrier": "United Airlines", "cost": 120.0, "duration_mins": 140}),
            
            # Flights connecting DFW to SEA and other hubs
            ("DFW", "SEA", {"carrier": "American Airlines", "cost": 280.0, "duration_mins": 250}),
            ("DFW", "ATL", {"carrier": "Delta Air Lines", "cost": 110.0, "duration_mins": 120}),
            
            # Flights connecting ATL to SEA and other hubs
            ("ATL", "SEA", {"carrier": "Delta Air Lines", "cost": 310.0, "duration_mins": 300}),
            ("ATL", "ORD", {"carrier": "Southwest Airlines", "cost": 95.0, "duration_mins": 115}),
            
            # Return/Alternative routes to enrich graph topology
            ("SEA", "ORD", {"carrier": "United Airlines", "cost": 240.0, "duration_mins": 260}),
            ("SEA", "DFW", {"carrier": "American Airlines", "cost": 270.0, "duration_mins": 240}),
        ]
        
        for u, v, attrs in flights:
            self.graph.add_edge(u, v, **attrs)

    def get_local_subgraph(self, start_node: str, max_hops: int = 2) -> str:
        """
        Extracts a localized neighborhood of nodes and edges up to `max_hops` 
        from `start_node` and returns it as a formatted JSON string.
        
        System Design Benefits of Local Subgraphs (GraphRAG):
        ----------------------------------------------------
        1. Context Optimization & Cost Control:
           In production airline scheduling or knowledge networks, the global graph
           may scale to millions of nodes (flight legs, users, products) and billions 
           of relationships. Injecting a flat, serialized list of all routes into 
           an LLM prompt is infeasible, expensive, and leads to token-bloat.
           
        2. Prevention of Attention Degradation ("Lost in the Middle"):
           LLMs suffer from recall loss when prompts are over-saturated with irrelevant
           data. By extracting only the ego-network (e.g., flights reachable within 
           2 hops of the source airport), we focus the LLM's attention strictly on the 
           relevant path-finding routes.
           
        3. Hallucination Reduction:
           Confining the facts available to the LLM to a deterministic, queried subgraph
           acts as a hard boundary. The LLM cannot hallucinate paths that do not exist, 
           as it is forced to reason strictly over the provided sub-graph topology.
        """
        start = start_node.strip().upper()
        if start not in self.graph:
            return json.dumps({
                "error": f"Airport code '{start}' not found in database.",
                "available_airports": list(self.graph.nodes)
            }, indent=2)
            
        # Find all nodes within max_hops using single source shortest path length
        # (This is equivalent to a bounded BFS traversal)
        distances = nx.single_source_shortest_path_length(self.graph, source=start, cutoff=max_hops)
        
        # Get the induced subgraph
        subg = self.graph.subgraph(distances.keys())
        
        # Format nodes and edges to standard dictionary
        nodes_data = []
        for node, data in subg.nodes(data=True):
            node_info = data.copy()
            node_info["hops_from_source"] = distances[node]
            nodes_data.append(node_info)
            
        edges_data = []
        for u, v, data in subg.edges(data=True):
            edge_info = data.copy()
            edge_info["origin"] = u
            edge_info["destination"] = v
            edges_data.append(edge_info)
            
        result = {
            "source_airport": start,
            "max_search_depth_hops": max_hops,
            "airports_in_subgraph": nodes_data,
            "available_flights": edges_data
        }
        
        return json.dumps(result, indent=2)

# =====================================================================
# Main execution for manual verification
# =====================================================================
if __name__ == "__main__":
    # Style helper
    def cyan(t): return f"\033[96m{t}\033[0m"
    def green(t): return f"\033[92m{t}\033[0m"
    
    print(cyan("\n[1] Initializing and Seeding FlightGraph..."))
    fg = FlightGraph()
    fg.seed_data()
    print(green(f"Successfully seeded graph with {fg.graph.number_of_nodes()} airports and {fg.graph.number_of_edges()} flights."))
    
    print(cyan("\n[2] Performing local subgraph retrieval (GraphRAG context extraction) for source 'DCA' with max_hops=2..."))
    subgraph_json = fg.get_local_subgraph("DCA", max_hops=2)
    print(green("Retrieved Subgraph Context (to be injected into LLM prompt):"))
    print(subgraph_json)
