"""
Renders a ResearchDAG as a tiered node-and-edge graph (HTML + inline SVG)
instead of a flat vertical list of cards.

Nodes are grouped into rows by dependency depth (longest path from a root
node), and dependency edges are drawn as SVG lines between rows. Pure
HTML/CSS/SVG via st.markdown(unsafe_allow_html=True) - no extra frontend
component dependency (streamlit-agraph exists but is effectively
unmaintained, and a custom component adds an iframe/JS failure surface
that isn't worth it for a diagram this simple).
"""

from collections import defaultdict
import textwrap

ROW_HEIGHT = 108   # px per DAG tier
BOX_HEIGHT = 58    # px, two-line node box
PAD_TOP = 16       # px

STATUS_STYLES = {
    "PENDING": {
        "border": "rgba(140,140,140,0.35)",
        "bg": "rgba(140,140,140,0.06)",
        "icon_color": "#888",
        "icon": "\u25cb",  # ○
    },
    "RUNNING": {
        "border": "#D97706",
        "bg": "rgba(245,158,11,0.08)",
        "icon_color": "#D97706",
        "icon": None,  # rendered as a spinning ring, see _ICON_HTML
    },
    "COMPLETED": {
        "border": "#059669",
        "bg": "rgba(16,185,129,0.08)",
        "icon_color": "#059669",
        "icon": "\u2713",  # ✓
    },
    "FAILED": {
        "border": "#DC2626",
        "bg": "rgba(239,68,68,0.08)",
        "icon_color": "#DC2626",
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
            return 0  # cycle guard - shouldn't happen, planner disallows cycles
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
    return (i + 0.5) / n * 100


def _box_width(n: int) -> float:
    return min(60.0, 90.0 / n)


def render_dag_graph(dag, node_statuses: dict) -> str:
    """Build the HTML/SVG for the DAG panel. Call via st.markdown(..., unsafe_allow_html=True)."""

    if not dag.nodes:
        return '<p style="color:#718096;font-size:0.85rem;">No nodes yet.</p>'

    depths = _compute_depths(dag)
    rows = defaultdict(list)
    for node_id, d in depths.items():
        rows[d].append(node_id)

    max_depth = max(rows)
    total_height = 2 * PAD_TOP + max_depth * ROW_HEIGHT + BOX_HEIGHT

    position = {}
    for row, node_ids in rows.items():
        for i, node_id in enumerate(sorted(node_ids)):
            position[node_id] = (row, i, len(node_ids))

    edges_svg = []
    for node in dag.nodes.values():
        node_row, node_i, node_n = position[node.id]
        x2, y2 = _center_x(node_i, node_n), _row_top(node_row)
        for dep_id in node.dependencies:
            if dep_id not in position:
                continue
            dep_row, dep_i, dep_n = position[dep_id]
            x1 = _center_x(dep_i, dep_n)
            y1 = _row_top(dep_row) + BOX_HEIGHT
            edges_svg.append(
                f'<line x1="{x1:.2f}" y1="{y1}" x2="{x2:.2f}" y2="{y2}" '
                f'stroke="rgba(140,140,140,0.5)" stroke-width="1"/>'
            )

    boxes_html = []
    for node in dag.nodes.values():
        row, i, n = position[node.id]
        status = node_statuses.get(node.id, "PENDING")
        style = STATUS_STYLES.get(status, STATUS_STYLES["PENDING"])
        w = _box_width(n)
        left = _center_x(i, n) - w / 2
        top = _row_top(row)

        if status == "RUNNING":
            icon_html = '<span class="dag-spinner"></span>'
        else:
            icon_html = (
                f'<span style="color:{style["icon_color"]};font-weight:700;'
                f'font-size:0.75rem;">{style["icon"]}</span>'
            )

        task_preview = node.task if len(node.task) <= 60 else node.task[:60] + "\u2026"

        boxes_html.append(
            f'<div style="position:absolute; left:{left:.2f}%; top:{top}px; width:{w:.2f}%; '
            f'box-sizing:border-box; background:{style["bg"]}; '
            f'border:1px solid {style["border"]}; border-radius:8px; padding:8px 10px;">'
            f'<div style="display:flex; align-items:center; gap:6px; margin-bottom:2px;">'
            f'{icon_html}'
            f'<span style="font-weight:600; font-size:0.85rem;">{node.id}</span>'
            f'</div>'
            f'<div style="font-size:0.72rem; color:#718096; line-height:1.3;">{task_preview}</div>'
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
             style="position:absolute; inset:0; width:100%; height:100%;">
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