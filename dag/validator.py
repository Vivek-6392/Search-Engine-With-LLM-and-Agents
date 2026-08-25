from collections import defaultdict, deque
import re
from typing import Any, Dict, List, Set, Tuple, Union

from dag.models import DAGNode, ResearchDAG


def _normalize_task(task_text: str) -> str:
    """Normalize task text for duplicate detection."""
    clean = re.sub(r"[^a-zA-Z0-9\s]", "", task_text.lower())
    return " ".join(clean.split())


def _is_redundant_synthesis_task(task_text: str, is_dependent: bool) -> bool:
    """
    Identify formatting-only, trivial aggregation, or duplicate synthesis nodes
    that duplicate the global synthesis engine's role.
    """
    if not is_dependent:
        return False

    norm = _normalize_task(task_text)
    synthesis_patterns = [
        "synthesize",
        "aggregate",
        "format as table",
        "format table",
        "summarize all findings",
        "summarize and compare",
        "combine all findings",
        "combine results",
        "final summary",
        "final synthesis",
        "compile final report",
    ]
    return any(p in norm for p in synthesis_patterns)


def validate_and_sanitize_dag(
    dag: Union[ResearchDAG, List[DAGNode], List[dict], dict],
    max_nodes: int = 4,
    max_depth: int = 3,
    max_fan_out: int = 4,
) -> ResearchDAG:
    """
    Validate, sanitize, and prune a ResearchDAG following 'minimum sufficient decomposition':
    1. Reject empty or whitespace tasks.
    2. Enforce unique node IDs.
    3. Remove self-dependencies and invalid dependency references.
    4. Detect and prune duplicate research tasks (merging dependency links).
    5. Prune redundant synthesis / formatting-only nodes.
    6. Break cycles (guarantee acyclic execution).
    7. Prune excessive nodes to enforce mode maximums.
    8. Enforce maximum depth (<= 3) and maximum fan-out (<= 4).
    """
    if not dag:
        return ResearchDAG()

    if isinstance(dag, ResearchDAG):
        node_list = list(dag.nodes.values())
    elif isinstance(dag, dict):
        if "nodes" in dag:
            node_list = dag["nodes"]
        else:
            node_list = list(dag.values())
    elif isinstance(dag, (list, tuple)):
        node_list = list(dag)
    else:
        return ResearchDAG()

    if not node_list:
        return ResearchDAG()

    clean_dag = ResearchDAG()
    seen_ids: Set[str] = set()
    seen_tasks: Dict[str, str] = {}  # norm_task -> canonical_node_id
    id_remap: Dict[str, str] = {}    # duplicate_id -> canonical_id

    # 1. Empty Task Rejection, Unique Node IDs, & Duplicate Detection
    for item in node_list:
        if isinstance(item, DAGNode):
            orig_id = (item.id or "").strip()
            task = (item.task or "").strip()
            deps = list(item.dependencies)
            t_type = getattr(item, "task_type", "llm")
            t_name = getattr(item, "tool_name", None)
            t_params = dict(getattr(item, "tool_params", {}) or {})
        elif isinstance(item, dict):
            orig_id = str(item.get("id", "")).strip()
            task = str(item.get("task", "")).strip()
            deps = list(item.get("dependencies", []))
            t_type = str(item.get("task_type", "llm")).strip().lower()
            t_name = item.get("tool") or item.get("tool_name")
            t_params = dict(item.get("tool_params", {}) or {})
            if "query" in item and "query" not in t_params:
                t_params["query"] = item["query"]
        else:
            continue

        if not task and not t_params.get("query"):
            continue  # Reject empty task

        if not task and t_params.get("query"):
            task = f"Execute tool {t_name or 'search'}: {t_params['query']}"

        norm = _normalize_task(task)
        if norm in seen_tasks:
            # Duplicate research task detected -> remap to canonical node
            canonical_id = seen_tasks[norm]
            id_remap[orig_id] = canonical_id
            continue

        # Ensure unique node ID
        node_id = orig_id if orig_id else f"node_{len(seen_ids) + 1}"
        if node_id in seen_ids:
            idx = 1
            while f"{node_id}_{idx}" in seen_ids:
                idx += 1
            new_id = f"{node_id}_{idx}"
            id_remap[orig_id] = new_id
            node_id = new_id

        seen_ids.add(node_id)
        seen_tasks[norm] = node_id
        clean_dag.add_node(
            DAGNode(
                id=node_id,
                task=task,
                dependencies=deps,
                task_type=t_type if t_type in ("tool", "llm") else "llm",
                tool_name=t_name,
                tool_params=t_params,
            )
        )

    if not clean_dag.nodes:
        return ResearchDAG()

    # 2. Dependency Cleaning (Self-deps, Nonexistent deps, and Remapped IDs)
    for node_id, node in list(clean_dag.nodes.items()):
        cleaned_deps = []
        for dep in node.dependencies:
            dep_actual = id_remap.get(dep, dep)
            if dep_actual != node_id and dep_actual in clean_dag.nodes and dep_actual not in cleaned_deps:
                cleaned_deps.append(dep_actual)
        node.dependencies = cleaned_deps

    # 3. Prune Redundant Synthesis / Formatting Nodes
    # If the DAG has > 1 node and one node is purely "synthesize/aggregate all other findings",
    # remove it since global synthesis already executes on all findings.
    if len(clean_dag.nodes) > 1:
        nodes_to_remove = []
        for node_id, node in clean_dag.nodes.items():
            if _is_redundant_synthesis_task(node.task, is_dependent=bool(node.dependencies)):
                nodes_to_remove.append(node_id)

        # Only remove if other research nodes remain
        if len(clean_dag.nodes) - len(nodes_to_remove) >= 1:
            for rem_id in nodes_to_remove:
                del clean_dag.nodes[rem_id]
            # Clean dependencies referencing removed nodes
            for node in clean_dag.nodes.values():
                node.dependencies = [d for d in node.dependencies if d in clean_dag.nodes]

    # 4. Cycle Detection and Resolution (Kahn's Algorithm / DFS Edge Pruning)
    clean_dag = _break_cycles(clean_dag)

    # 5. Excessive-Node Pruning (Enforce max_nodes cap)
    if len(clean_dag.nodes) > max_nodes:
        # Keep roots and early layers first
        layers = clean_dag.topological_layers()
        retained_nodes = []
        for layer in layers:
            for node in layer:
                if len(retained_nodes) < max_nodes:
                    retained_nodes.append(node.id)
                else:
                    break

        pruned_nodes = {nid: clean_dag.nodes[nid] for nid in retained_nodes}
        clean_dag.nodes = pruned_nodes
        # Clean dependencies for retained nodes
        for node in clean_dag.nodes.values():
            node.dependencies = [d for d in node.dependencies if d in clean_dag.nodes]

    # 6. Depth and Fan-Out Pruning
    clean_dag = _enforce_depth_and_fan_out(clean_dag, max_depth=max_depth, max_fan_out=max_fan_out)

    return clean_dag


def _break_cycles(dag: ResearchDAG) -> ResearchDAG:
    """Detect and remove edges creating cycles to guarantee a strict DAG."""
    if not dag.nodes:
        return dag

    # Tarjan / DFS Cycle Detection
    visited = {}  # 0=unvisited, 1=visiting, 2=visited
    for nid in dag.nodes:
        visited[nid] = 0

    def dfs(node_id: str):
        visited[node_id] = 1  # visiting
        node = dag.nodes[node_id]
        safe_deps = []
        for dep in node.dependencies:
            if dep not in dag.nodes:
                continue
            if visited[dep] == 1:
                # Cycle detected! Break edge from node_id -> dep
                continue
            if visited[dep] == 0:
                dfs(dep)
            safe_deps.append(dep)
        node.dependencies = safe_deps
        visited[node_id] = 2  # finished

    for nid in list(dag.nodes.keys()):
        if visited[nid] == 0:
            dfs(nid)

    return dag


def _enforce_depth_and_fan_out(
    dag: ResearchDAG,
    max_depth: int = 3,
    max_fan_out: int = 4,
) -> ResearchDAG:
    """Enforce max dependency depth and max parent fan-out."""
    if not dag.nodes:
        return dag

    # Enforce Depth: Flatten chains deeper than max_depth
    layers = dag.topological_layers()
    if len(layers) > max_depth:
        # For nodes in layers >= max_depth, flatten dependencies to top-level layer 0
        allowed_parents = set(n.id for n in layers[0])
        for layer_idx in range(max_depth, len(layers)):
            for node in layers[layer_idx]:
                node.dependencies = [d for d in node.dependencies if d in allowed_parents]

    # Enforce Fan-out: Restrict parent dependencies if too many children attach
    parent_child_count = defaultdict(int)
    for node in dag.nodes.values():
        valid_deps = []
        for dep in node.dependencies:
            if parent_child_count[dep] < max_fan_out:
                parent_child_count[dep] += 1
                valid_deps.append(dep)
        node.dependencies = valid_deps

    return dag
