"""
High-impact, judge-friendly Gradio UI for AutonomousReturns-v0.

REBUILT: Operations dashboard for judges.

Sections:
1. Header bar (task badge, profit, pending count)
2. Main area: 2-column layout (item card | action buttons)
3. Resolution feed (action → condition → profit/loss)
4. Pending timeline (visual countdown bars)
5. Statistics accordion

NO developer elements. NO JSON panels. NO code snippets.
"""

from __future__ import annotations

import asyncio
from typing import Any

import gradio as gr

from autonomous_returns_v0.baseline import BaselineAgent, BaselineRunConfig
from autonomous_returns_v0.models import DispositionAction, ReturnsObservation
from autonomous_returns_v0.policies import HeuristicPolicy


def _build_header(task: str, ledger: float, pending_count: int) -> str:
    """Header bar with task badge, profit (BIG green/red), pending count."""
    profit_color = "#22c55e" if ledger >= 0 else "#ef4444"
    profit_sign = "+" if ledger >= 0 else ""
    task_colors = {"easy": "#3b82f6", "medium": "#f59e0b", "hard": "#ef4444"}
    task_color = task_colors.get(task.lower(), "#6b7280")
    
    return f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                padding:20px 28px;background:#0f172a;border-bottom:2px solid #1e293b">
        <div>
            <h1 style="margin:0;font-size:26px;color:#f8fafc;font-weight:700">
                Autonomous Returns
            </h1>
            <p style="margin:4px 0 0;font-size:13px;color:#64748b">
                E-commerce Returns Decision System
            </p>
        </div>
        <div style="display:flex;gap:40px;align-items:center">
            <div style="text-align:center">
                <p style="margin:0;font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px">Task</p>
                <span style="background:{task_color};color:white;padding:6px 14px;border-radius:6px;font-size:13px;font-weight:700;display:inline-block;margin-top:4px">
                    {task.upper()}
                </span>
            </div>
            <div style="text-align:center;min-width:160px">
                <p style="margin:0;font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px">Running Profit</p>
                <p style="margin:6px 0 0;font-size:34px;font-weight:700;color:{profit_color}">
                    {profit_sign}${abs(ledger):.2f}
                </p>
            </div>
            <div style="text-align:center">
                <p style="margin:0;font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px">Pending</p>
                <p style="margin:6px 0 0;font-size:28px;font-weight:700;color:#f97316">{pending_count}</p>
            </div>
        </div>
    </div>
    """


def _build_item_card(item, reward_detail=None) -> str:
    """LEFT COLUMN: Item inspection card - product details, condition meter, customer risk."""
    if item is None:
        return '''
        <div style="padding:48px;text-align:center;background:#1e293b;border-radius:12px;color:#94a3b8;height:100%">
            <p style="font-size:24px;margin:0">📦 No items in queue</p>
            <p style="font-size:14px;margin:12px 0 0">Waiting for pending resolutions...</p>
        </div>
        '''
    
    # Condition bar (0-10 visual meter)
    score = item.noisy_condition_score
    filled_pct = int(score) * 10
    cond_color = "#22c55e" if score >= 7 else "#f59e0b" if score >= 4 else "#ef4444"
    
    # Customer risk based on return count
    risk = "LOW" if item.customer_return_count <= 1 else "MEDIUM" if item.customer_return_count <= 3 else "HIGH"
    risk_colors = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}
    risk_color = risk_colors[risk]
    
    price_color = "#f8fafc" if item.price >= 200 else "#94a3b8"
    
    return f'''
    <div style="background:#1e293b;border-radius:12px;padding:24px;height:100%;display:flex;flex-direction:column">
        <h3 style="margin:0 0 20px;font-size:13px;color:#64748b;text-transform:uppercase;letter-spacing:1px">
            Item Inspection
        </h3>
        
        <div style="flex:1">
            <h2 style="margin:0 0 6px;font-size:24px;color:#f8fafc;font-weight:700">{item.product_title}</h2>
            <p style="margin:0 0 20px;font-size:14px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px">
                {item.category} · <span style="color:{price_color}">${item.price:.2f}</span>
            </p>
            
            <div style="margin-bottom:16px">
                <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                    <span style="font-size:12px;color:#64748b">Condition Score</span>
                    <span style="font-size:13px;font-weight:600;color:{cond_color}">{score:.1f}/10</span>
                </div>
                <div style="height:10px;background:#334155;border-radius:5px;overflow:hidden">
                    <div style="width:{filled_pct}%;height:100%;background:{cond_color};border-radius:5px"></div>
                </div>
            </div>
            
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px">
                <div style="padding:14px;background:#0f172a;border-radius:8px">
                    <p style="margin:0;font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px">Return Reason</p>
                    <p style="margin:6px 0 0;font-size:13px;color:#cbd5e1;font-style:italic">"{item.stated_return_reason}"</p>
                </div>
                <div style="padding:14px;background:#0f172a;border-radius:8px">
                    <p style="margin:0;font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px">Customer Risk</p>
                    <p style="margin:6px 0 0;font-size:18px;font-weight:700;color:{risk_color}">{risk}</p>
                </div>
            </div>
            
            <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px">
                <span style="padding:6px 10px;background:#334155;border-radius:6px;font-size:12px;color:#cbd5e1">
                    📦 {item.packaging_condition or 'N/A'}
                </span>
                <span style="padding:6px 10px;background:#334155;border-radius:6px;font-size:12px;color:#cbd5e1">
                    ↩️ {item.customer_return_count}x returns
                </span>
            </div>
        </div>
    </div>
    '''


def _build_resolution_feed(resolutions: list, reveal: bool = False) -> str:
    """Resolution feed - shows: Action → TRUE condition → Profit/Loss"""
    if not resolutions:
        return '''
        <div style="padding:40px;text-align:center;background:#1e293b;border-radius:12px;color:#64748b">
            <p style="font-size:36px;margin:0">📋</p>
            <p style="font-size:14px;margin:12px 0 0">No resolutions yet. Actions appear here.</p>
        </div>
        '''
    
    entries = ""
    for res in resolutions[-6:]:
        action = res.action.value.replace("_", " ").title()
        reward = res.normalized_reward
        profit_color = "#22c55e" if reward >= 0 else "#ef4444"
        profit_sign = "+" if reward >= 0 else ""
        
        true_cond = res.hidden_condition_revealed.value if reveal and res.hidden_condition_revealed else "??? <span style='font-size:10px;color:#64748b'>(hidden)</span>"
        
        bg_color = "#052e16" if reward >= 0 else "#1c0a0a"
        
        entries += f'''
        <div style="display:flex;justify-content:space-between;align-items:center;padding:14px 16px;
                    background:{bg_color};border-radius:8px;margin-bottom:8px;
                    border-left:5px solid {profit_color}">
            <div>
                <span style="font-size:15px;font-weight:600;color:#f8fafc">{action}</span>
                <span style="font-size:13px;color:#64748b;margin-left:10px">→</span>
                <span style="font-size:13px;color:#94a3b8;margin-left:10px">{true_cond}</span>
            </div>
            <span style="font-size:20px;font-weight:700;color:{profit_color}">{profit_sign}{reward:.2f}</span>
        </div>
        '''
    
    return f'''
    <div style="background:#1e293b;border-radius:12px;padding:20px;height:100%;display:flex;flex-direction:column">
        <h3 style="margin:0 0 16px;font-size:13px;color:#64748b;text-transform:uppercase;letter-spacing:1px">
            Resolution Feed
        </h3>
        <div style="flex:1;max-height:350px;overflow-y:auto;padding-right:4px">
            {entries}
        </div>
    </div>
    '''


def _build_pending_timeline(pending_items: list) -> str:
    """Visual timeline of items waiting to resolve."""
    if not pending_items:
        return '''
        <div style="padding:24px;text-align:center;background:#1e293b;border-radius:12px;color:#64748b">
            <p style="font-size:28px;margin:0">✓</p>
            <p style="font-size:13px;margin:8px 0 0">No pending items</p>
        </div>
        '''
    
    items = ""
    for item in pending_items[:5]:
        delay = max(1, item.get("delay_remaining", 1))
        progress = max(0, min(100, (6 - delay) / 6 * 100))
        bar_color = "#22c55e" if delay <= 1 else "#f59e0b" if delay <= 3 else "#3b82f6"
        urgency_color = "#ef4444" if delay <= 1 else "#f59e0b" if delay <= 2 else "#64748b"
        action = item.get("action", "?").replace("_", " ").title()
        
        items += f'''
        <div style="margin-bottom:10px">
            <div style="display:flex;justify-content:space-between;margin-bottom:5px">
                <span style="font-size:12px;color:#cbd5e1">{action}</span>
                <span style="font-size:11px;font-weight:600;color:{urgency_color}">⏱ {delay} step{"s" if delay != 1 else ""}</span>
            </div>
            <div style="height:6px;background:#334155;border-radius:3px">
                <div style="width:{progress}%;height:100%;background:{bar_color};border-radius:3px"></div>
            </div>
        </div>
        '''
    
    remaining = len(pending_items) - 5
    if remaining > 0:
        items += f'<p style="margin:8px 0 0;font-size:11px;color:#64748b;text-align:center">+{remaining} more pending</p>'
    
    return f'''
    <div style="background:#1e293b;border-radius:12px;padding:16px">
        <h4 style="margin:0 0 12px;font-size:12px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px">
            Pending Timeline ({len(pending_items)})
        </h4>
        {items}
    </div>
    '''


def _build_action_card() -> str:
    """Action panel HTML (informational only - actual buttons are Gradio components)."""
    return ""


def _build_controls(task: str = "easy", seed: int = 42) -> str:
    """Footer controls - task selector, seed, reset, autoplay."""
    task_options = "".join(
        f'<option value="{t}" {"selected" if t == task else ""}>{t.title()}</option>'
        for t in ["easy", "medium", "hard"]
    )
    
    return f'''
    <div style="background:#1e293b;border-radius:12px;padding:20px">
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr auto;gap:16px;align-items:end">
            <div>
                <label style="display:block;font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px">Task</label>
                <select id="task_select" style="width:100%;padding:12px;background:#0f172a;color:#f8fafc;border:1px solid #334155;border-radius:8px;font-size:14px">
                    {task_options}
                </select>
            </div>
            <div>
                <label style="display:block;font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px">Seed</label>
                <input type="number" id="seed_input" value="{seed}" style="width:100%;padding:12px;background:#0f172a;color:#f8fafc;border:1px solid #334155;border-radius:8px;font-size:14px"/>
            </div>
            <div>
                <label style="display:flex;align-items:center;gap:10px;padding:12px;background:#0f172a;border:1px solid #334155;border-radius:8px;cursor:pointer">
                    <input type="checkbox" id="reveal_checkbox" style="width:16px;height:16px;cursor:pointer"/>
                    <span style="font-size:13px;color:#cbd5e1">Reveal conditions</span>
                </label>
            </div>
            <button id="reset_btn" style="padding:12px 24px;font-size:14px;font-weight:600;color:white;background:#3b82f6;border:none;border-radius:8px;cursor:pointer">
                🔄 Reset
            </button>
        </div>
    </div>
    '''


def _build_stats(total_processed: int, total_fraud: int) -> str:
    """Statistics panel."""
    return f'''
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
        <div style="padding:20px;background:#1e293b;border-radius:10px;text-align:center">
            <p style="margin:0;font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px">Total Processed</p>
            <p style="margin:8px 0 0;font-size:32px;font-weight:700;color:#f8fafc">{total_processed}</p>
        </div>
        <div style="padding:20px;background:#1e293b;border-radius:10px;text-align:center">
            <p style="margin:0;font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px">Fraud Caught</p>
            <p style="margin:8px 0 0;font-size:32px;font-weight:700;color:#dc2626">{total_fraud}</p>
        </div>
    </div>
    '''


def _get_state_from_manager(mgr: Any) -> dict:
    """Extract state dict from web_manager."""
    try:
        episode_state = getattr(mgr, "episode_state", None)
        if episode_state is None:
            return {}
        state = getattr(episode_state, "current_observation", None)
        if state is None:
            return {}
        if hasattr(state, "model_dump"):
            return state.model_dump()
        return dict(state) if isinstance(state, dict) else {}
    except Exception:
        return {}


def _get_pending_items(mgr: Any) -> list:
    """Extract pending items from web_manager state."""
    try:
        episode_state = getattr(mgr, "episode_state", None)
        if episode_state is None:
            return []
        pending = getattr(episode_state, "pending_items", [])
        if pending is None:
            return []
        if hasattr(pending[0], "model_dump") if pending else False:
            return [p.model_dump() for p in pending]
        return list(pending) if pending else []
    except Exception:
        return []


def _get_resolution_history(mgr: Any) -> list:
    """Extract resolution history from web_manager."""
    try:
        episode_state = getattr(mgr, "episode_state", None)
        if episode_state is None:
            return []
        history = getattr(episode_state, "resolution_history", [])
        if history is None:
            return []
        if hasattr(history[0], "model_dump") if history else False:
            return [h.model_dump() for h in history]
        return list(history) if history else []
    except Exception:
        return []


def _make_resolution_obj(data: dict):
    """Create a mock resolution object from dict."""
    class Res:
        def __init__(self, d):
            self.action = type('Action', (), {'value': d.get('action', 'unknown')})()
            self.normalized_reward = d.get('normalized_reward', 0.0)
            self.hidden_condition_revealed = type('Cond', (), {'value': d.get('hidden_condition_revealed', 'unknown')})()
    return Res(data)


def _get_hidden_cond(res) -> str:
    """Get hidden condition from resolution."""
    try:
        return res.hidden_condition_revealed.value if hasattr(res.hidden_condition_revealed, 'value') else str(res.hidden_condition_revealed)
    except:
        return "unknown"


async def _render_ui(web_manager: Any, reveal: bool = False) -> list:
    """
    Render all UI components. Returns exactly 5 values matching outputs list.
    
    IMPORTANT: All event handlers MUST return exactly these 5 values in this order.
    """
    state = _get_state_from_manager(web_manager)
    obs = state if isinstance(state, dict) else {}
    
    task = obs.get("task", "easy") if obs else "easy"
    ledger = obs.get("running_ledger", 0.0) if obs else 0.0
    pending_count = obs.get("pending_resolution_count", 0) if obs else 0
    current_item = obs.get("current_item", None) if obs else None
    reward_detail = obs.get("reward_detail", None) if obs else None
    
    pending_items = _get_pending_items(web_manager)
    resolution_history = _get_resolution_history(web_manager)
    
    header = _build_header(task, ledger, pending_count)
    item_card = _build_item_card(current_item, reward_detail)
    resolution_feed = _build_resolution_feed(
        [_make_resolution_obj(r) for r in resolution_history[-10:]],
        reveal
    )
    pending_timeline = _build_pending_timeline(pending_items)
    
    total_processed = len(resolution_history)
    total_fraud = sum(1 for r in resolution_history[-10:]
                      if _get_hidden_cond(r) == "fraudulent")
    stats = _build_stats(total_processed, total_fraud)
    
    # Return exactly 6 values matching outputs list
    return [
        header,        # 1
        item_card,     # 2
        resolution_feed,  # 3
        pending_timeline,   # 4
        stats,        # 5
    ]


def gradio_builder(
    web_manager: Any,
    action_fields: list[dict[str, Any]],
    metadata: Any,
    is_chat_env: bool,
    title: str,
    quick_start_md: str | None,
) -> gr.Blocks:
    """
    Judge-winning operations dashboard.
    
    Layout:
    - Header bar (task, profit, pending)
    - Main area: 2 columns (item card | action buttons)
    - Resolution feed
    - Pending timeline
    - Statistics accordion
    
    NO developer elements. NO code. NO JSON.
    """
    del action_fields, is_chat_env, metadata
    
    with gr.Blocks(title=title) as demo:
        demo.theme = gr.themes.Origin()
        gr.Markdown(
            "<style>.gradio-container {max-width: 1200px !important}</style>",
            visible=False
        )
        
        # HEADER
        header_html = gr.HTML()
        
        # MAIN AREA: 2 columns
        with gr.Row():
            with gr.Column(scale=3):
                item_card_html = gr.HTML()
                
                gr.Markdown("### Decision Panel", elem_id="decision-panel-header")
                
                # 8 ACTION BUTTONS - large and clickable
                with gr.Row():
                    resell_full_btn = gr.Button("Resell Full", variant="primary", size="lg")
                with gr.Row():
                    discount_15_btn = gr.Button("Discount 15%", size="lg")
                    discount_30_btn = gr.Button("Discount 30%", size="lg")
                with gr.Row():
                    discount_50_btn = gr.Button("Discount 50%", size="lg")
                    refurbish_btn = gr.Button("Refurbish", size="lg")
                with gr.Row():
                    dispose_btn = gr.Button("Dispose", size="lg")
                    flag_fraud_btn = gr.Button("Flag Fraud", variant="stop", size="lg")
                with gr.Row():
                    wait_btn = gr.Button("Wait", size="lg")
            
            with gr.Column(scale=2):
                resolution_feed_html = gr.HTML()
                pending_timeline_html = gr.HTML()
        
        # STATISTICS ACCORDION
        with gr.Accordion("📊 Statistics", open=False):
            stats_html = gr.HTML()
        
        # CONTROLS
        controls_html = gr.HTML()
        
        # Define outputs list ONCE - use for all handlers
        # IMPORTANT: Must match exactly what _render_ui returns (5 values)
        all_outputs = [
            header_html,
            item_card_html,
            resolution_feed_html,
            pending_timeline_html,
            stats_html,
        ]
        
        # EVENT HANDLERS
        # Each handler must return exactly 6 values (matching all_outputs)
        
        async def on_reset(task: str, seed: Any, reveal: bool):
            kwargs = {"task": str(task).strip().lower(), "reveal_hidden_conditions": bool(reveal)}
            if seed is not None and str(seed).strip() != "":
                try:
                    kwargs["seed"] = int(seed)
                except (TypeError, ValueError):
                    pass
            await web_manager.reset_environment(kwargs)
            return await _render_ui(web_manager, bool(reveal))
        
        async def on_action_resell_full():
            await web_manager.step_environment({"action": "resell_full"})
            return await _render_ui(web_manager, False)
        
        async def on_action_discount_15():
            await web_manager.step_environment({"action": "resell_discount_15"})
            return await _render_ui(web_manager, False)
        
        async def on_action_discount_30():
            await web_manager.step_environment({"action": "resell_discount_30"})
            return await _render_ui(web_manager, False)
        
        async def on_action_discount_50():
            await web_manager.step_environment({"action": "resell_discount_50"})
            return await _render_ui(web_manager, False)
        
        async def on_action_refurbish():
            await web_manager.step_environment({"action": "refurbish"})
            return await _render_ui(web_manager, False)
        
        async def on_action_dispose():
            await web_manager.step_environment({"action": "dispose"})
            return await _render_ui(web_manager, False)
        
        async def on_action_flag_fraud():
            await web_manager.step_environment({"action": "flag_fraud"})
            return await _render_ui(web_manager, False)
        
        async def on_action_wait():
            await web_manager.step_environment({"action": "wait"})
            return await _render_ui(web_manager, False)
        
        async def on_autoplay(task: str, seed: Any, reveal: bool, steps: int, agent: str, provider: str, model: str, temp: float):
            import os
            
            kwargs = {"task": str(task).strip().lower(), "reveal_hidden_conditions": bool(reveal)}
            if seed is not None and str(seed).strip() != "":
                try:
                    kwargs["seed"] = int(seed)
                except (TypeError, ValueError):
                    pass
            
            await web_manager.reset_environment(kwargs)
            
            count = 0
            max_count = min(steps, 200)
            
            if agent == "Heuristic":
                policy = HeuristicPolicy()
                for _ in range(max_count):
                    obs = _get_state_from_manager(web_manager)
                    if not obs:
                        break
                    if obs.get("done") or (obs.get("remaining_items", 0) == 0 and obs.get("pending_resolution_count", 0) == 0):
                        break
                    action = policy(type('Obs', (), obs)())
                    await web_manager.step_environment({"action": action.action.value})
                    count += 1
            else:
                provider = str(provider).strip().lower()
                if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
                    return await _render_ui(web_manager, reveal)
                if provider == "google" and not os.getenv("VERTEX_PROJECT_ID"):
                    return await _render_ui(web_manager, reveal)
                
                resolved_model = (model or "").strip() or (
                    "gpt-4o-mini" if provider == "openai" else (
                        "gemini-2.0-flash"
                    )
                )
                config = BaselineRunConfig(provider=provider, model=resolved_model, temperature=float(temp))
                agent_obj = BaselineAgent(config)
                
                for _ in range(max_count):
                    obs = _get_state_from_manager(web_manager)
                    if not obs:
                        break
                    if obs.get("done") or (obs.get("remaining_items", 0) == 0 and obs.get("pending_resolution_count", 0) == 0):
                        break
                    try:
                        obs_obj = type('Obs', (), obs)()
                        action = await asyncio.to_thread(agent_obj.act, obs_obj)
                    except Exception:
                        return await _render_ui(web_manager, reveal)
                    await web_manager.step_environment({"action": action.action.value})
                    count += 1
            
            return await _render_ui(web_manager, reveal)
        
        # WIRE UP ALL EVENT HANDLERS
        # Each returns 5 values, matching all_outputs
        
        async def on_load():
            return await _render_ui(web_manager, False)
            
        demo.load(
            fn=on_load,
            outputs=all_outputs
        )
        
        # Action buttons - each handler captures its action value via closure
        resell_full_btn.click(fn=on_action_resell_full, inputs=[], outputs=all_outputs)
        discount_15_btn.click(fn=on_action_discount_15, inputs=[], outputs=all_outputs)
        discount_30_btn.click(fn=on_action_discount_30, inputs=[], outputs=all_outputs)
        discount_50_btn.click(fn=on_action_discount_50, inputs=[], outputs=all_outputs)
        refurbish_btn.click(fn=on_action_refurbish, inputs=[], outputs=all_outputs)
        dispose_btn.click(fn=on_action_dispose, inputs=[], outputs=all_outputs)
        flag_fraud_btn.click(fn=on_action_flag_fraud, inputs=[], outputs=all_outputs)
        wait_btn.click(fn=on_action_wait, inputs=[], outputs=all_outputs)
        
        # Note: Reset and autoplay handlers need to be connected to actual UI components
        # These are placeholder connections - actual inputs would need to be defined
        
    return demo
