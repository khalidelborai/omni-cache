"""
Dependency Graph model for cache invalidation.

This module defines the dependency graph model for managing cache invalidation
relationships between cache entries with graph operations.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Set, Tuple
from enum import Enum
import time
import json
from collections import defaultdict, deque


class DependencyType(Enum):
    """Types of dependencies between cache entries."""
    STRONG = "strong"      # Invalidation propagates immediately
    WEAK = "weak"          # Invalidation is delayed/batched
    COMPUTED = "computed"  # Derived data dependencies
    TEMPORAL = "temporal"  # Time-based dependencies
    CONDITIONAL = "conditional"  # Conditional dependencies


class InvalidationStrategy(Enum):
    """Invalidation propagation strategies."""
    IMMEDIATE = "immediate"      # Invalidate immediately
    BATCHED = "batched"         # Batch invalidations
    LAZY = "lazy"               # Invalidate on next access
    CASCADING = "cascading"     # Cascade through all levels
    SELECTIVE = "selective"     # Selective based on conditions


@dataclass
class DependencyEdge:
    """Represents a dependency edge between two cache keys."""
    source: str
    target: str
    dependency_type: DependencyType = DependencyType.STRONG
    weight: float = 1.0
    condition: Optional[str] = None  # Condition for conditional dependencies
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate edge data."""
        if not self.source:
            raise ValueError("Source key cannot be empty")
        if not self.target:
            raise ValueError("Target key cannot be empty")
        if self.weight < 0:
            raise ValueError("Weight cannot be negative")

    @property
    def is_strong(self) -> bool:
        """Check if this is a strong dependency."""
        return self.dependency_type == DependencyType.STRONG

    @property
    def is_weak(self) -> bool:
        """Check if this is a weak dependency."""
        return self.dependency_type == DependencyType.WEAK

    @property
    def is_conditional(self) -> bool:
        """Check if this is a conditional dependency."""
        return self.dependency_type == DependencyType.CONDITIONAL

    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "dependency_type": self.dependency_type.value,
            "weight": self.weight,
            "condition": self.condition,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DependencyEdge':
        """Create edge from dictionary."""
        return cls(
            source=data["source"],
            target=data["target"],
            dependency_type=DependencyType(data.get("dependency_type", "strong")),
            weight=data.get("weight", 1.0),
            condition=data.get("condition"),
            created_at=data.get("created_at", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class InvalidationPlan:
    """Plan for cache invalidation execution."""
    root_keys: Set[str] = field(default_factory=set)
    levels: List[Set[str]] = field(default_factory=list)
    total_keys: int = 0
    estimated_cost: float = 0.0
    strategy: InvalidationStrategy = InvalidationStrategy.IMMEDIATE
    created_at: float = field(default_factory=time.time)

    def add_level(self, keys: Set[str]) -> None:
        """Add a level to the invalidation plan."""
        if keys:
            self.levels.append(keys)
            self.total_keys += len(keys)

    def get_all_keys(self) -> Set[str]:
        """Get all keys in the invalidation plan."""
        all_keys = set()
        for level in self.levels:
            all_keys.update(level)
        return all_keys

    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary."""
        return {
            "root_keys": list(self.root_keys),
            "levels": [list(level) for level in self.levels],
            "total_keys": self.total_keys,
            "estimated_cost": self.estimated_cost,
            "strategy": self.strategy.value,
            "created_at": self.created_at,
        }


@dataclass
class DependencyGraph:
    """
    Dependency graph model for cache invalidation.

    Manages relationships between cache entries and provides graph operations
    for efficient cache invalidation planning and execution.
    """

    name: str = "default"
    description: str = ""

    # Graph storage
    _adjacency_list: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    _reverse_adjacency_list: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    _edges: Dict[Tuple[str, str], DependencyEdge] = field(default_factory=dict)

    # Configuration
    max_depth: int = 10
    max_invalidation_keys: int = 1000
    default_strategy: InvalidationStrategy = InvalidationStrategy.IMMEDIATE

    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: int = 1

    def __post_init__(self):
        """Post-initialization setup."""
        if not self.name:
            raise ValueError("Graph name is required")

    @property
    def node_count(self) -> int:
        """Get number of nodes in the graph."""
        all_nodes = set(self._adjacency_list.keys())
        all_nodes.update(self._reverse_adjacency_list.keys())
        return len(all_nodes)

    @property
    def edge_count(self) -> int:
        """Get number of edges in the graph."""
        return len(self._edges)

    @property
    def is_empty(self) -> bool:
        """Check if graph is empty."""
        return self.edge_count == 0

    def add_dependency(self, source: str, target: str,
                      dependency_type: DependencyType = DependencyType.STRONG,
                      weight: float = 1.0, condition: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a dependency edge between two cache keys.

        Args:
            source: Source cache key
            target: Target cache key (depends on source)
            dependency_type: Type of dependency
            weight: Edge weight for prioritization
            condition: Optional condition for conditional dependencies
            metadata: Additional edge metadata

        Returns:
            True if edge was added, False if it already exists
        """
        if source == target:
            raise ValueError("Self-dependencies are not allowed")

        edge_key = (source, target)
        if edge_key in self._edges:
            return False

        # Check for cycles
        if self._would_create_cycle(source, target):
            raise ValueError(f"Adding edge {source} -> {target} would create a cycle")

        # Create edge
        edge = DependencyEdge(
            source=source,
            target=target,
            dependency_type=dependency_type,
            weight=weight,
            condition=condition,
            metadata=metadata or {}
        )

        # Add to graph
        self._edges[edge_key] = edge
        self._adjacency_list[source].add(target)
        self._reverse_adjacency_list[target].add(source)

        self.updated_at = time.time()
        self.version += 1

        return True

    def remove_dependency(self, source: str, target: str) -> bool:
        """
        Remove a dependency edge.

        Args:
            source: Source cache key
            target: Target cache key

        Returns:
            True if edge was removed, False if it didn't exist
        """
        edge_key = (source, target)
        if edge_key not in self._edges:
            return False

        # Remove edge
        del self._edges[edge_key]
        self._adjacency_list[source].discard(target)
        self._reverse_adjacency_list[target].discard(source)

        # Clean up empty entries
        if not self._adjacency_list[source]:
            del self._adjacency_list[source]
        if not self._reverse_adjacency_list[target]:
            del self._reverse_adjacency_list[target]

        self.updated_at = time.time()
        self.version += 1

        return True

    def remove_node(self, key: str) -> int:
        """
        Remove a node and all its dependencies.

        Args:
            key: Cache key to remove

        Returns:
            Number of edges removed
        """
        removed_count = 0

        # Remove outgoing edges
        targets = list(self._adjacency_list.get(key, []))
        for target in targets:
            if self.remove_dependency(key, target):
                removed_count += 1

        # Remove incoming edges
        sources = list(self._reverse_adjacency_list.get(key, []))
        for source in sources:
            if self.remove_dependency(source, key):
                removed_count += 1

        return removed_count

    def get_dependencies(self, key: str) -> List[DependencyEdge]:
        """Get all outgoing dependencies for a key."""
        dependencies = []
        for target in self._adjacency_list.get(key, []):
            edge = self._edges.get((key, target))
            if edge:
                dependencies.append(edge)
        return dependencies

    def get_dependents(self, key: str) -> List[DependencyEdge]:
        """Get all incoming dependencies for a key."""
        dependents = []
        for source in self._reverse_adjacency_list.get(key, []):
            edge = self._edges.get((source, key))
            if edge:
                dependents.append(edge)
        return dependents

    def get_transitive_dependencies(self, key: str, max_depth: Optional[int] = None) -> Set[str]:
        """
        Get all transitive dependencies for a key using DFS.

        Args:
            key: Starting cache key
            max_depth: Maximum depth to traverse

        Returns:
            Set of all dependent keys
        """
        if max_depth is None:
            max_depth = self.max_depth

        visited = set()
        stack = [(key, 0)]

        while stack:
            current_key, depth = stack.pop()

            if current_key in visited or depth >= max_depth:
                continue

            visited.add(current_key)

            # Add direct dependencies
            for target in self._adjacency_list.get(current_key, []):
                if target not in visited:
                    stack.append((target, depth + 1))

        visited.discard(key)  # Remove the starting key
        return visited

    def get_transitive_dependents(self, key: str, max_depth: Optional[int] = None) -> Set[str]:
        """
        Get all transitive dependents for a key using DFS.

        Args:
            key: Starting cache key
            max_depth: Maximum depth to traverse

        Returns:
            Set of all keys that depend on this key
        """
        if max_depth is None:
            max_depth = self.max_depth

        visited = set()
        stack = [(key, 0)]

        while stack:
            current_key, depth = stack.pop()

            if current_key in visited or depth >= max_depth:
                continue

            visited.add(current_key)

            # Add direct dependents
            for source in self._reverse_adjacency_list.get(current_key, []):
                if source not in visited:
                    stack.append((source, depth + 1))

        visited.discard(key)  # Remove the starting key
        return visited

    def create_invalidation_plan(self, root_keys: Set[str],
                                strategy: Optional[InvalidationStrategy] = None) -> InvalidationPlan:
        """
        Create an invalidation plan for the given root keys.

        Args:
            root_keys: Keys that were initially invalidated
            strategy: Invalidation strategy to use

        Returns:
            InvalidationPlan with ordered levels of keys to invalidate
        """
        if strategy is None:
            strategy = self.default_strategy

        plan = InvalidationPlan(
            root_keys=root_keys.copy(),
            strategy=strategy
        )

        if strategy == InvalidationStrategy.IMMEDIATE:
            return self._create_immediate_plan(root_keys, plan)
        elif strategy == InvalidationStrategy.BATCHED:
            return self._create_batched_plan(root_keys, plan)
        elif strategy == InvalidationStrategy.LAZY:
            return self._create_lazy_plan(root_keys, plan)
        elif strategy == InvalidationStrategy.CASCADING:
            return self._create_cascading_plan(root_keys, plan)
        elif strategy == InvalidationStrategy.SELECTIVE:
            return self._create_selective_plan(root_keys, plan)
        else:
            return self._create_immediate_plan(root_keys, plan)

    def _create_immediate_plan(self, root_keys: Set[str], plan: InvalidationPlan) -> InvalidationPlan:
        """Create immediate invalidation plan using BFS."""
        visited = set()
        current_level = root_keys.copy()
        depth = 0

        while current_level and depth < self.max_depth:
            if plan.total_keys + len(current_level) > self.max_invalidation_keys:
                break

            plan.add_level(current_level)
            visited.update(current_level)

            # Find next level
            next_level = set()
            for key in current_level:
                for target in self._adjacency_list.get(key, []):
                    edge = self._edges.get((key, target))
                    if edge and target not in visited and edge.is_strong:
                        next_level.add(target)

            current_level = next_level
            depth += 1

        return plan

    def _create_batched_plan(self, root_keys: Set[str], plan: InvalidationPlan) -> InvalidationPlan:
        """Create batched invalidation plan."""
        # Group by dependency type and weight
        strong_deps = set()
        weak_deps = set()

        for key in root_keys:
            for target in self._adjacency_list.get(key, []):
                edge = self._edges.get((key, target))
                if edge:
                    if edge.is_strong:
                        strong_deps.add(target)
                    else:
                        weak_deps.add(target)

        if strong_deps:
            plan.add_level(strong_deps)
        if weak_deps:
            plan.add_level(weak_deps)

        return plan

    def _create_lazy_plan(self, root_keys: Set[str], plan: InvalidationPlan) -> InvalidationPlan:
        """Create lazy invalidation plan (only direct dependencies)."""
        direct_deps = set()
        for key in root_keys:
            direct_deps.update(self._adjacency_list.get(key, []))

        if direct_deps:
            plan.add_level(direct_deps)

        return plan

    def _create_cascading_plan(self, root_keys: Set[str], plan: InvalidationPlan) -> InvalidationPlan:
        """Create cascading invalidation plan (all transitive dependencies)."""
        all_deps = set()
        for key in root_keys:
            all_deps.update(self.get_transitive_dependencies(key))

        # Sort by dependency distance for level assignment
        levels_dict = defaultdict(set)
        for dep in all_deps:
            distance = self._calculate_distance(root_keys, dep)
            levels_dict[distance].add(dep)

        for distance in sorted(levels_dict.keys()):
            plan.add_level(levels_dict[distance])

        return plan

    def _create_selective_plan(self, root_keys: Set[str], plan: InvalidationPlan) -> InvalidationPlan:
        """Create selective invalidation plan based on conditions."""
        selected_keys = set()

        for key in root_keys:
            for target in self._adjacency_list.get(key, []):
                edge = self._edges.get((key, target))
                if edge and self._should_invalidate(edge):
                    selected_keys.add(target)

        if selected_keys:
            plan.add_level(selected_keys)

        return plan

    def _would_create_cycle(self, source: str, target: str) -> bool:
        """Check if adding an edge would create a cycle."""
        if target not in self._adjacency_list:
            return False

        # DFS from target to see if we can reach source
        visited = set()
        stack = [target]

        while stack:
            current = stack.pop()
            if current == source:
                return True

            if current in visited:
                continue

            visited.add(current)
            stack.extend(self._adjacency_list.get(current, []))

        return False

    def _calculate_distance(self, sources: Set[str], target: str) -> int:
        """Calculate minimum distance from any source to target."""
        if target in sources:
            return 0

        visited = set()
        queue = deque([(source, 0) for source in sources])

        while queue:
            current, distance = queue.popleft()

            if current == target:
                return distance

            if current in visited:
                continue

            visited.add(current)

            for neighbor in self._adjacency_list.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))

        return float('inf')

    def _should_invalidate(self, edge: DependencyEdge) -> bool:
        """Determine if an edge should trigger invalidation."""
        if edge.is_conditional and edge.condition:
            # Here you would evaluate the condition
            # For now, return True for all conditions
            return True
        return True

    def detect_cycles(self) -> List[List[str]]:
        """Detect all cycles in the graph."""
        visited = set()
        rec_stack = set()
        cycles = []

        def dfs(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self._adjacency_list.get(node, []):
                if neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
                elif neighbor not in visited:
                    dfs(neighbor, path)

            rec_stack.remove(node)
            path.pop()

        for node in self._adjacency_list:
            if node not in visited:
                dfs(node, [])

        return cycles

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        cycles = self.detect_cycles()

        # Calculate degree statistics
        in_degrees = [len(self._reverse_adjacency_list.get(node, []))
                     for node in self._adjacency_list]
        out_degrees = [len(deps) for deps in self._adjacency_list.values()]

        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "cycle_count": len(cycles),
            "avg_in_degree": sum(in_degrees) / len(in_degrees) if in_degrees else 0,
            "avg_out_degree": sum(out_degrees) / len(out_degrees) if out_degrees else 0,
            "max_in_degree": max(in_degrees) if in_degrees else 0,
            "max_out_degree": max(out_degrees) if out_degrees else 0,
            "dependency_types": self._count_dependency_types(),
        }

    def _count_dependency_types(self) -> Dict[str, int]:
        """Count edges by dependency type."""
        type_counts = defaultdict(int)
        for edge in self._edges.values():
            type_counts[edge.dependency_type.value] += 1
        return dict(type_counts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "edges": [edge.to_dict() for edge in self._edges.values()],
            "max_depth": self.max_depth,
            "max_invalidation_keys": self.max_invalidation_keys,
            "default_strategy": self.default_strategy.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "statistics": self.get_statistics(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DependencyGraph':
        """Create graph from dictionary representation."""
        graph = cls(
            name=data.get("name", "default"),
            description=data.get("description", ""),
            max_depth=data.get("max_depth", 10),
            max_invalidation_keys=data.get("max_invalidation_keys", 1000),
            default_strategy=InvalidationStrategy(data.get("default_strategy", "immediate")),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            version=data.get("version", 1),
        )

        # Add edges
        for edge_data in data.get("edges", []):
            edge = DependencyEdge.from_dict(edge_data)
            graph.add_dependency(
                edge.source,
                edge.target,
                edge.dependency_type,
                edge.weight,
                edge.condition,
                edge.metadata
            )

        return graph

    def to_json(self) -> str:
        """Convert graph to JSON string."""
        return json.dumps(self.to_dict(), default=str, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'DependencyGraph':
        """Create graph from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __str__(self) -> str:
        """String representation of the graph."""
        return f"DependencyGraph({self.name}, {self.node_count} nodes, {self.edge_count} edges)"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"DependencyGraph(name='{self.name}', nodes={self.node_count}, "
                f"edges={self.edge_count}, version={self.version})")

    def __eq__(self, other) -> bool:
        """Check equality based on name and version."""
        if not isinstance(other, DependencyGraph):
            return False
        return self.name == other.name and self.version == other.version

    def __hash__(self) -> int:
        """Hash based on name."""
        return hash(self.name)