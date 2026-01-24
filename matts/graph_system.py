#!/usr/bin/env python3
"""
Basic Graph System for Relationship Modeling

This module provides a basic graph system for modeling relationships between
contexts, generators, observers, and other components in the library.
"""

import uuid
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque

# =============================================================================
# GRAPH ENUMS AND TYPES
# =============================================================================

class RelationshipType(Enum):
    """Types of relationships in the graph."""
    DEPENDS_ON = "depends_on"
    CONTAINS = "contains"
    TRIGGERS = "triggers"
    OBSERVES = "observes"
    INHERITS_FROM = "inherits_from"
    COMPOSES_WITH = "composes_with"
    CUSTOM = "custom"


@dataclass
class GraphNode:
    """Node in the relationship graph."""
    node_id: str
    node_type: str  # "context", "generator", "observer", "signal"
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """Edge representing relationship between nodes."""
    edge_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    relationship_type: RelationshipType = RelationshipType.CUSTOM
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


class BasicRelationshipGraph:
    """Basic graph system for modeling relationships."""
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self.adjacency_list: Dict[str, List[str]] = defaultdict(list)
        
    def add_node(self, node: GraphNode):
        """Add node to graph."""
        self.nodes[node.node_id] = node
        if node.node_id not in self.adjacency_list:
            self.adjacency_list[node.node_id] = []
    
    def add_edge(self, edge: GraphEdge):
        """Add edge to graph."""
        self.edges[edge.edge_id] = edge
        self.adjacency_list[edge.source_id].append(edge.target_id)
    
    def find_related_nodes(self, node_id: str, relationship_type: RelationshipType = None,
                          max_depth: int = 3) -> List[GraphNode]:
        """Find nodes related to given node."""
        if node_id not in self.nodes:
            return []
        
        visited = set()
        queue = deque([(node_id, 0)])
        related_nodes = []
        
        while queue:
            current_id, depth = queue.popleft()
            
            if depth > max_depth or current_id in visited:
                continue
            
            visited.add(current_id)
            
            if current_id != node_id:
                related_nodes.append(self.nodes[current_id])
            
            for edge in self.edges.values():
                if (edge.source_id == current_id and 
                    (relationship_type is None or edge.relationship_type == relationship_type)):
                    queue.append((edge.target_id, depth + 1))
        
        return related_nodes
    
    def get_relationship_path(self, from_node: str, to_node: str) -> Optional[List[GraphEdge]]:
        """Find path of relationships between two nodes."""
        if from_node not in self.nodes or to_node not in self.nodes:
            return None
        
        visited = set()
        queue = deque([(from_node, [])])
        
        while queue:
            current_id, path = queue.popleft()
            
            if current_id == to_node:
                return path
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            for edge in self.edges.values():
                if edge.source_id == current_id and edge.target_id not in visited:
                    new_path = path + [edge]
                    queue.append((edge.target_id, new_path))
        
        return None