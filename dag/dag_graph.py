"""
Renders a ResearchDAG as a tiered node-and-edge graph (HTML + colorful CSS neon connectors).
Displays vibrant glowing gradient connector lines with arrowheads between parent and child nodes,
and highlights which tool was called for each node (e.g. node_1 (calculator), node_2 (web search)).
"""

from collections import defaultdict
import textwrap

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
            return 0
        node = dag.nodes[node_id]
        deps = [d for d in node.dependencies if d in dag.nodes]
        depths[node_id] = (
            0 if not deps else 1 + max(depth_of(d, stack + (node_id,)) for d in deps)
        )
        return depths[node_id]

    for node_id in dag.nodes:
        depth_of(node_id)
    return depths


def _render_node_card(node, status: str) -> str:
    style = STATUS_STYLES.get(status, STATUS_STYLES["PENDING"])

    if status == "RUNNING":
        icon_html = '<span class="dag-spinner"></span>'
    else:
        icon_html = (
            f'<span style="color:{style["icon_color"]};font-weight:700;'
            f'font-size:0.8rem;">{style["icon"]}</span>'
        )

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
            f'font-size:0.68rem; font-weight:600; padding:1px 6px; border-radius:4px; '
            f'border:1px solid rgba(147,197,253,0.25); white-space:nowrap; text-transform:lowercase;">'
            f'{tool_label}</span>'
        )

    task_preview = node.task if len(node.task) <= 65 else node.task[:62] + "\u2026"

    return f'''
    <div style="background:{style["bg"]}; border:1px solid {style["border"]}; border-radius:8px;
                padding:8px 12px; flex:1 1 140px; min-width:130px; max-width:260px; box-sizing:border-box;
                box-shadow:0 2px 8px rgba(0,0,0,0.35);">
        <div style="display:flex; align-items:center; justify-content:space-between; gap:4px; margin-bottom:4px;">
            <div style="display:flex; align-items:center; gap:5px;">
                {icon_html}
                <span style="font-weight:600; font-size:0.86rem; color:#FFFFFF;">{node.id}</span>
            </div>
            {tool_badge_html}
        </div>
        <div style="font-size:0.72rem; color:#94A3B8; line-height:1.35; overflow:hidden; text-overflow:ellipsis;">{task_preview}</div>
    </div>
    '''


def _render_colorful_connector(parent_count: int, child_count: int) -> str:
    """Render a vibrant, colorful glowing gradient line connecting tiers."""
    if parent_count == 1 and child_count == 1:
        return '''
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; width:100%; margin:5px 0;">
            <div style="width:3px; height:24px; background:linear-gradient(180deg, #3B82F6 0%, #8B5CF6 50%, #10B981 100%); border-radius:2px; box-shadow:0 0 10px rgba(59,130,246,0.8), 0 0 4px rgba(16,185,129,0.8);"></div>
            <div style="width:0; height:0; border-left:6px solid transparent; border-right:6px solid transparent; border-top:7px solid #10B981; filter:drop-shadow(0 2px 5px rgba(16,185,129,0.9)); margin-top:-1px;"></div>
        </div>
        '''
    elif parent_count == 1 and child_count >= 2:
        return '''
        <div style="display:flex; flex-direction:column; align-items:center; width:100%; margin:4px 0;">
            <div style="width:3px; height:12px; background:linear-gradient(180deg, #3B82F6, #8B5CF6); border-radius:2px; box-shadow:0 0 8px rgba(59,130,246,0.7);"></div>
            <div style="width:52%; height:3px; background:linear-gradient(90deg, #3B82F6 0%, #8B5CF6 50%, #10B981 100%); border-radius:2px; box-shadow:0 0 10px rgba(139,92,246,0.7); position:relative;">
                <div style="position:absolute; left:0; top:0; width:3px; height:12px; background:#3B82F6; box-shadow:0 0 6px rgba(59,130,246,0.8);"></div>
                <div style="position:absolute; left:-4.5px; top:11px; width:0; height:0; border-left:6px solid transparent; border-right:6px solid transparent; border-top:7px solid #3B82F6; filter:drop-shadow(0 2px 4px rgba(59,130,246,0.8));"></div>
                <div style="position:absolute; right:0; top:0; width:3px; height:12px; background:#10B981; box-shadow:0 0 6px rgba(16,185,129,0.8);"></div>
                <div style="position:absolute; right:-4.5px; top:11px; width:0; height:0; border-left:6px solid transparent; border-right:6px solid transparent; border-top:7px solid #10B981; filter:drop-shadow(0 2px 4px rgba(16,185,129,0.8));"></div>
            </div>
            <div style="height:16px;"></div>
        </div>
        '''
    elif parent_count >= 2 and child_count == 1:
        return '''
        <div style="display:flex; flex-direction:column; align-items:center; width:100%; margin:4px 0;">
            <div style="height:14px; position:relative; width:52%;">
                <div style="position:absolute; left:0; top:0; width:3px; height:14px; background:#3B82F6; box-shadow:0 0 6px rgba(59,130,246,0.8);"></div>
                <div style="position:absolute; right:0; top:0; width:3px; height:14px; background:#10B981; box-shadow:0 0 6px rgba(16,185,129,0.8);"></div>
            </div>
            <div style="width:52%; height:3px; background:linear-gradient(90deg, #3B82F6 0%, #8B5CF6 50%, #10B981 100%); border-radius:2px; box-shadow:0 0 10px rgba(139,92,246,0.7);"></div>
            <div style="width:3px; height:12px; background:linear-gradient(180deg, #8B5CF6, #10B981); border-radius:2px; box-shadow:0 0 8px rgba(16,185,129,0.7);"></div>
            <div style="width:0; height:0; border-left:6px solid transparent; border-right:6px solid transparent; border-top:7px solid #10B981; filter:drop-shadow(0 2px 5px rgba(16,185,129,0.9)); margin-top:-1px;"></div>
        </div>
        '''
    else:
        return '''
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; width:100%; margin:5px 0;">
            <div style="width:3px; height:24px; background:linear-gradient(180deg, #3B82F6 0%, #8B5CF6 50%, #10B981 100%); border-radius:2px; box-shadow:0 0 10px rgba(59,130,246,0.8);"></div>
            <div style="width:0; height:0; border-left:6px solid transparent; border-right:6px solid transparent; border-top:7px solid #10B981; filter:drop-shadow(0 2px 5px rgba(16,185,129,0.9)); margin-top:-1px;"></div>
        </div>
        '''


def render_dag_graph(dag, node_statuses: dict) -> str:
    """Build the HTML for the DAG panel with vibrant glowing gradient connector lines and tool badges."""
    if not dag.nodes:
        return '<p style="color:#718096;font-size:0.85rem;">No nodes yet.</p>'

    depths = _compute_depths(dag)
    rows = defaultdict(list)
    for node_id, d in depths.items():
        rows[d].append(node_id)

    tier_keys = sorted(rows.keys())
    output_html_parts = []

    for tier_idx, depth in enumerate(tier_keys):
        node_ids = rows[depth]
        row_cards = []
        for nid in sorted(node_ids):
            node = dag.nodes[nid]
            status = node_statuses.get(nid, "PENDING")
            row_cards.append(_render_node_card(node, status))

        output_html_parts.append(
            f'<div style="display:flex; justify-content:center; gap:10px; width:100%; flex-wrap:wrap; box-sizing:border-box;">'
            f'{"".join(row_cards)}'
            f'</div>'
        )

        # Render colorful connector line to the next tier if another tier follows
        if tier_idx < len(tier_keys) - 1:
            next_node_ids = rows[tier_keys[tier_idx + 1]]
            connector_html = _render_colorful_connector(len(node_ids), len(next_node_ids))
            output_html_parts.append(connector_html)

    return textwrap.dedent(f'''
    <style>
    @keyframes dag-spin {{ to {{ transform: rotate(360deg); }} }}
    .dag-spinner {{
        width: 11px; height: 11px; border-radius: 50%;
        border: 2px solid rgba(217,119,6,0.25); border-top-color: #D97706;
        display: inline-block; animation: dag-spin 0.8s linear infinite;
    }}
    </style>
    <div style="display:flex; flex-direction:column; align-items:center; width:100%; padding:6px 0; box-sizing:border-box;">
        {"".join(output_html_parts)}
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