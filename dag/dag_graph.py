"""
Renders a ResearchDAG as a tiered node-and-edge graph (HTML + inline SVG).
Displays connection curves/arrows between parent and child nodes, and highlights
which tool was called for each node (e.g. node_1 (calculator), node_2 (web search)).
"""

from collections import defaultdict
import textwrap

ROW_HEIGHT = 118   # px per DAG tier
BOX_HEIGHT = 62    # px node box
PAD_TOP = 14       # px

STATUS_STYLES = {
    "PENDING": {
        "border": "#2D3748",
        "bg": "#141414",
        "icon_color": "#64748B",
        "icon": "\u25cb",  # ○
    },
    "RUNNING": {
        "border": "#D97706",
        "bg": "rgba(245,158,11,0.08)",
        "icon_color": "#F59E0B",
        "icon": None,  # animated ring
    },
    "COMPLETED": {
        "border": "#10B981",
        "bg": "rgba(16,185,129,0.08)",
        "icon_color": "#10B981",
        "icon": "\u2713",  # ✓
    },
    "FAILED": {
        "border": "#EF4444",
        "bg": "rgba(239,68,68,0.08)",
        "icon_color": "#EF4444",
        "icon": "\u2715",  # ✕
    },
}


def _compute_depths(dag):
    """Longest-path depth of each node from its roots, for tiered layout."""
    depths = {}

    def depth_of(node_id, stack=()):
        if node_id in depths:
            return depths[node_id]
        if node_id in stack:
            return 0  # cycle guard
        node = dag.nodes[node_id]
        deps = [d for d in node.dependencies if d in dag.nodes]
        depths[node_id] = (
            0 if not deps else 1 + max(depth_of(d, stack + (node_id,)) for d in deps)
        )
        return depths[node_id]

    for node_id in dag.nodes:
        depth_of(node_id)
    return depths


def _row_top(row: int) -> int:
    return PAD_TOP + row * ROW_HEIGHT


def _center_x(i: int, n: int) -> float:
    return (i + 0.5) / n * 100.0


def _box_width(n: int) -> float:
    return min(55.0, 88.0 / n)


def render_dag_graph(dag, node_statuses: dict) -> str:
    """Build the HTML/SVG for the DAG panel with visible connector lines and tool call badges."""
    if not dag.nodes:
        return '<p style="color:#718096;font-size:0.85rem;">No nodes yet.</p>'

    depths = _compute_depths(dag)
    rows = defaultdict(list)
    for node_id, d in depths.items():
        rows[d].append(node_id)

    max_depth = max(rows)
    total_height = 2 * PAD_TOP + max_depth * ROW_HEIGHT + BOX_HEIGHT + 10

    position = {}
    for row, node_ids in rows.items():
        for i, node_id in enumerate(sorted(node_ids)):
            position[node_id] = (row, i, len(node_ids))

    # SVG Connectors between nodes (percentage X, pixel Y)
    edges_svg = []
    for node in dag.nodes.values():
        node_row, node_i, node_n = position[node.id]
        x2 = _center_x(node_i, node_n)
        y2 = float(_row_top(node_row))
        status = node_statuses.get(node.id, "PENDING")

        for dep_id in node.dependencies:
            if dep_id not in position:
                continue
            dep_row, dep_i, dep_n = position[dep_id]
            x1 = _center_x(dep_i, dep_n)
            y1 = float(_row_top(dep_row) + BOX_HEIGHT)
            my = (y1 + y2) / 2.0

            if status == "COMPLETED":
                stroke_color = "#10B981"
                marker = "url(#arrow-completed)"
                dash = ""
            elif status == "RUNNING":
                stroke_color = "#F59E0B"
                marker = "url(#arrow-running)"
                dash = 'stroke-dasharray="3,2"'
            else:
                stroke_color = "#64748B"
                marker = "url(#arrow)"
                dash = ""

            edges_svg.append(
                f'<path d="M {x1:.2f} {y1:.1f} C {x1:.2f} {my:.1f}, {x2:.2f} {my:.1f}, {x2:.2f} {y2:.1f}" '
                f'fill="none" stroke="{stroke_color}" stroke-width="2.5" vector-effect="non-scaling-stroke" {dash} marker-end="{marker}"/>'
            )

    # Node Cards with Tool Call Badges
    boxes_html = []
    for node in dag.nodes.values():
        row, i, n = position[node.id]
        status = node_statuses.get(node.id, "PENDING")
        style = STATUS_STYLES.get(status, STATUS_STYLES["PENDING"])
        w = _box_width(n)
        left = _center_x(i, n) - w / 2.0
        top = _row_top(row)

        if status == "RUNNING":
            icon_html = '<span class="dag-spinner"></span>'
        else:
            icon_html = (
                f'<span style="color:{style["icon_color"]};font-weight:700;'
                f'font-size:0.78rem;">{style["icon"]}</span>'
            )

        # Tool Badge (e.g. calculator, web search, finance)
        tool_label = getattr(node, "tool_used", None)
        if not tool_label:
            if status == "RUNNING":
                tool_label = "running..."
            elif status == "COMPLETED":
                tool_label = "direct"
            else:
                tool_label = None

        tool_badge_html = ""
        if tool_label:
            tool_badge_html = (
                f'<span style="background:rgba(59,130,246,0.15); color:#93C5FD; '
                f'font-size:0.65rem; font-weight:600; padding:1px 6px; border-radius:4px; '
                f'border:1px solid rgba(147,197,253,0.25); white-space:nowrap; text-transform:lowercase;">'
                f'{tool_label}</span>'
            )

        task_preview = node.task if len(node.task) <= 55 else node.task[:52] + "\u2026"

        boxes_html.append(
            f'<div style="position:absolute; left:{left:.2f}%; top:{top}px; width:{w:.2f}%; '
            f'box-sizing:border-box; background:{style["bg"]}; '
            f'border:1px solid {style["border"]}; border-radius:8px; padding:7px 10px; '
            f'box-shadow:0 2px 8px rgba(0,0,0,0.35);">'
            f'<div style="display:flex; align-items:center; justify-content:space-between; gap:4px; margin-bottom:3px;">'
            f'<div style="display:flex; align-items:center; gap:5px;">'
            f'{icon_html}'
            f'<span style="font-weight:600; font-size:0.84rem; color:#FFFFFF;">{node.id}</span>'
            f'</div>'
            f'{tool_badge_html}'
            f'</div>'
            f'<div style="font-size:0.71rem; color:#94A3B8; line-height:1.3; overflow:hidden; text-overflow:ellipsis;">{task_preview}</div>'
            f'</div>'
        )

    return textwrap.dedent(f'''
    <style>
    @keyframes dag-spin {{ to {{ transform: rotate(360deg); }} }}
    .dag-spinner {{
        width: 11px; height: 11px; border-radius: 50%;
        border: 2px solid rgba(217,119,6,0.25); border-top-color: #D97706;
        display: inline-block; animation: dag-spin 0.8s linear infinite;
    }}
    </style>
    <div style="position:relative; height:{total_height}px; width:100%; margin-bottom:8px;">
        <svg viewBox="0 0 100 {total_height}" preserveAspectRatio="none"
             style="position:absolute; inset:0; width:100%; height:100%; pointer-events:none;">
            <defs>
                <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                    <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#64748B" />
                </marker>
                <marker id="arrow-completed" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                    <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#10B981" />
                </marker>
                <marker id="arrow-running" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                    <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#F59E0B" />
                </marker>
            </defs>
            {"".join(edges_svg)}
        </svg>
        {"".join(boxes_html)}
    </div>
    ''').strip()


def render_synthesis_skeleton() -> str:
    """Shimmering placeholder shown in the Synthesis Canvas while the DAG executes."""
    bars = "".join(
        f'<div class="dag-shimmer" style="height:11px;border-radius:4px;'
        f'background:rgba(140,140,140,0.12);margin-bottom:10px;width:{w}%;"></div>'
        for w in (95, 88, 92, 60)
    )
    return textwrap.dedent(f'''
    <style>
    @keyframes dag-shimmer {{ 0%, 100% {{ opacity: 0.5; }} 50% {{ opacity: 1; }} }}
    .dag-shimmer {{ animation: dag-shimmer 1.6s ease-in-out infinite; }}
    </style>
    <div style="padding-top:4px;">{bars}</div>
    ''').strip()