import time
import json
import streamlit as st
import plotly.graph_objects as go
from google import genai
from google.genai import types
from search_engine import (
    LANDMARKS, DEFAULT_WALLS,
    run_astar, run_bfs, run_dfs, run_hill_climbing, run_simulated_annealing
)

st.set_page_config(
    page_title="PathSense AI | Campus Navigation & Penguin Assistant", page_icon="logo.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 3.6rem !important; padding-bottom: 1rem !important; }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #f8fafc;
        margin-bottom: 0.15rem;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 1.25rem;
    }
    .stat-card {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
    }
    .stat-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    .stat-value {
        font-size: 1.35rem;
        font-weight: 600;
        color: #38bdf8;
        margin-top: 0.15rem;
    }
    .stat-sub {
        font-size: 0.8rem;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)

if "start_loc" not in st.session_state:
    st.session_state.start_loc = "Main Gate"
if "goal_loc" not in st.session_state:
    st.session_state.goal_loc = "Central Library"
if "custom_walls" not in st.session_state:
    st.session_state.custom_walls = set()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "Hello! I am Pip, your PathSense AI Penguin Assistant. Ask me anything—from planning optimal campus routes and comparing A*, BFS, DFS, Hill Climbing, and Simulated Annealing, to general AI concepts or coding questions."
        }
    ]

api_key = ""
try:
    api_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    api_key = ""

with st.sidebar:
    st.markdown("### Navigation Parameters")
    landmark_names = list(LANDMARKS.keys())
    start_choice = st.selectbox(
        "Origin Landmark",
        landmark_names,
        index=landmark_names.index(st.session_state.start_loc)
    )
    goal_choice = st.selectbox(
        "Destination Landmark",
        landmark_names,
        index=landmark_names.index(st.session_state.goal_loc)
    )

    if start_choice != st.session_state.start_loc or goal_choice != st.session_state.goal_loc:
        st.session_state.start_loc = start_choice
        st.session_state.goal_loc = goal_choice

    st.markdown("---")
    st.markdown("### Path Overlays")
    show_astar = st.checkbox("A* Optimal Path (Manhattan)", value=True)
    show_bfs = st.checkbox("Breadth-First Search (BFS)", value=True)
    show_dfs = st.checkbox("Depth-First Search (DFS)", value=False)
    show_hc = st.checkbox("Hill Climbing", value=True)
    show_sa = st.checkbox("Simulated Annealing", value=False)

    st.markdown("---")
    st.markdown("### Dynamic Obstacles")
    extra_block = st.toggle("Library Corridor Blockade", value=False)
    custom_coord = st.selectbox(
        "Add Custom Barrier Cell (Row, Col)",
        ["None"] + [f"({r}, {c})" for r in range(12) for c in range(12) if (r, c) not in LANDMARKS.values()]
    )
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("Add Barrier", use_container_width=True) and custom_coord != "None":
            r_val, c_val = eval(custom_coord)
            st.session_state.custom_walls.add((r_val, c_val))
            st.rerun()
    with col_b2:
        if st.button("Reset Grid", use_container_width=True):
            st.session_state.custom_walls = set()
            st.rerun()

    st.markdown("---")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.chat_history = [st.session_state.chat_history[0]]
        st.rerun()

active_walls = set(DEFAULT_WALLS) | st.session_state.custom_walls
if extra_block:
    active_walls.update({(3, 8), (2, 9)})

start_coords = LANDMARKS[st.session_state.start_loc]
goal_coords = LANDMARKS[st.session_state.goal_loc]

res_astar = run_astar(start_coords, goal_coords, walls=active_walls)
res_bfs = run_bfs(start_coords, goal_coords, walls=active_walls)
res_dfs = run_dfs(start_coords, goal_coords, walls=active_walls)
res_hc = run_hill_climbing(start_coords, goal_coords, walls=active_walls)
res_sa = run_simulated_annealing(start_coords, goal_coords, walls=active_walls)

st.markdown('<div style="display:flex;align-items:center;gap:14px;"><img src="https://raw.githubusercontent.com/sammangupta477/pathsense-ai-assistant/main/logo.svg" width="48" height="48"/><div class="hero-title">PathSense AI</div></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Interactive State-Space Search Visualizer & Conversational AI Assistant (AIP203 Artificial Intelligence Lab)</div>',
    unsafe_allow_html=True
)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Active Route</div>
        <div class="stat-value" style="font-size:1.05rem;">{st.session_state.start_loc} &rarr; {st.session_state.goal_loc}</div>
        <div class="stat-sub">{start_coords} to {goal_coords}</div>
    </div>""", unsafe_allow_html=True)
with k2:
    cost_disp = f"{res_astar['path_cost']} steps" if res_astar["path_cost"] != -1 else "Unreachable"
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Optimal Cost (A*)</div>
        <div class="stat-value">{cost_disp}</div>
        <div class="stat-sub">Manhattan Heuristic</div>
    </div>""", unsafe_allow_html=True)
with k3:
    savings = res_bfs["explored_count"] - res_astar["explored_count"]
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Nodes Expanded (A* vs BFS)</div>
        <div class="stat-value">{res_astar['explored_count']} vs {res_bfs['explored_count']}</div>
        <div class="stat-sub">{max(savings, 0)} fewer nodes than BFS</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Hill Climbing Status</div>
        <div class="stat-value" style="font-size:1.05rem; color:{'#34d399' if res_hc['path_cost'] != -1 else '#f87171'};">{res_hc['status']}</div>
        <div class="stat-sub">Local search benchmark</div>
    </div>""", unsafe_allow_html=True)

col_map, col_chat = st.columns([1.2, 1], gap="large")

with col_map:
    tab_map, tab_metrics = st.tabs(["Interactive Campus Grid", "Algorithm Benchmark Table"])

    with tab_map:
        fig = go.Figure()

        grid_x, grid_y, hover_texts = [], [], []
        for r in range(12):
            for c in range(12):
                grid_x.append(c)
                grid_y.append(r)
                cell_type = "Obstacle / Building" if (r, c) in active_walls else "Walkable Corridor"
                for lname, lcoord in LANDMARKS.items():
                    if lcoord == (r, c):
                        cell_type = f"Landmark: {lname}"
                hover_texts.append(f"Row {r}, Col {c}<br>{cell_type}")

        fig.add_trace(go.Scatter(
            x=grid_x, y=grid_y,
            mode="markers",
            marker=dict(size=22, color="#1e293b", symbol="square", line=dict(width=1, color="#334155")),
            text=hover_texts,
            hoverinfo="text",
            showlegend=False
        ))

        if active_walls:
            wy, wx = zip(*active_walls)
            fig.add_trace(go.Scatter(
                x=wx, y=wy,
                mode="markers",
                name="Obstacles",
                marker=dict(size=22, color="#475569", symbol="square", line=dict(width=1, color="#64748b")),
                hoverinfo="skip"
            ))

        path_layers = [
            (show_dfs, res_dfs, "DFS Path", "#a855f7", "dot", 2.5),
            (show_sa, res_sa, "Simulated Annealing", "#ec4899", "dashdot", 2.5),
            (show_bfs, res_bfs, "BFS Path", "#f59e0b", "dash", 3),
            (show_hc, res_hc, "Hill Climbing", "#ef4444", "dot", 3),
            (show_astar, res_astar, "A* Optimal Path", "#10b981", "solid", 4.5),
        ]
        for enabled, res_obj, label, color, dash_style, width in path_layers:
            if enabled and res_obj["path"]:
                py, px = zip(*res_obj["path"])
                fig.add_trace(go.Scatter(
                    x=px, y=py,
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color, width=width, dash=dash_style),
                    marker=dict(size=6, color=color),
                    hovertemplate=f"{label}<br>Row: %{{y}}, Col: %{{x}}<extra></extra>"
                ))

        for name, (r, c) in LANDMARKS.items():
            is_start = name == st.session_state.start_loc
            is_goal = name == st.session_state.goal_loc
            color = "#22c55e" if is_start else ("#ef4444" if is_goal else "#38bdf8")
            size = 16 if (is_start or is_goal) else 11
            fig.add_trace(go.Scatter(
                x=[c], y=[r],
                mode="markers+text",
                name=name,
                text=[name],
                textposition="top center",
                textfont=dict(size=10, color="#f8fafc"),
                marker=dict(size=size, color=color, line=dict(width=2, color="#0f172a")),
                showlegend=False,
                hovertemplate=f"<b>{name}</b><br>Coordinates: ({r}, {c})<extra></extra>"
            ))

        fig.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="#0b1120",
            plot_bgcolor="#0f172a",
            xaxis=dict(range=[-0.7, 11.7], showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[11.7, -0.7], showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="left",
                x=0,
                font=dict(size=11, color="#cbd5e1"),
                bgcolor="rgba(15, 23, 42, 0.8)"
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_metrics:
        st.markdown("#### Real-Time Comparative Performance")
        st.dataframe(
            [
                {
                    "Algorithm": r["algorithm"],
                    "Status": r["status"],
                    "Path Cost": r["path_cost"] if r["path_cost"] != -1 else "N/A",
                    "Nodes Explored": r["explored_count"],
                    "Execution Time (ms)": r["time_ms"]
                }
                for r in [res_astar, res_bfs, res_dfs, res_hc, res_sa]
            ],
            use_container_width=True,
            hide_index=True
        )

with col_chat:
    st.markdown("#### Pip — Penguin Virtual Assistant")
    chat_box = st.container(height=355, border=True)
    with chat_box:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="penguin_avatar.svg" if msg["role"] == "assistant" else None):
                st.markdown(msg["content"])

    user_query = st.chat_input("Ask anything: plan a route, compare algorithms, or ask any general question...")
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        if not api_key:
            reply = (
                "Backend API key is not configured yet. Add `GEMINI_API_KEY` in Streamlit Cloud "
                "Settings -> Secrets to enable full conversational responses."
            )
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

        
def ask_gemini_resilient(client, contents, config=None):
    models = ["gemini-3.8-flash", "gemini-3.8-flash-lite", "gemini-flash-latest"]
    last_err = None
    for attempt in range(3):
        for m in models:
            try:
                return client.models.generate_content(model=m, contents=contents, config=config)
            except Exception as e:
                last_err = e
                err_str = str(e)
                if "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str:
                    time.sleep(0.8 * (attempt + 1))
                    continue
                if "404" in err_str or "NOT_FOUND" in err_str:
                    continue
                raise e
    raise last_err

        client = genai.Client(api_key=api_key)

        extraction_prompt = f"""
        Determine if the user's message asks to navigate or find a path between two campus landmarks.
        Valid landmarks: {list(LANDMARKS.keys())}
        Return ONLY valid JSON: {{"is_navigation": true/false, "start": "<landmark or null>", "goal": "<landmark or null>"}}
        User message: {user_query}
        """
        route_updated = False
        try:
            intent_resp = ask_gemini_resilient(
                client,
                contents=extraction_prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            parsed = json.loads(intent_resp.text)
            if parsed.get("is_navigation"):
                new_s = parsed.get("start")
                new_g = parsed.get("goal")
                if new_s in LANDMARKS:
                    st.session_state.start_loc = new_s
                    route_updated = True
                if new_g in LANDMARKS:
                    st.session_state.goal_loc = new_g
                    route_updated = True
                if route_updated:
                    sc = LANDMARKS[st.session_state.start_loc]
                    gc = LANDMARKS[st.session_state.goal_loc]
                    res_astar = run_astar(sc, gc, walls=active_walls)
                    res_bfs = run_bfs(sc, gc, walls=active_walls)
                    res_dfs = run_dfs(sc, gc, walls=active_walls)
                    res_hc = run_hill_climbing(sc, gc, walls=active_walls)
                    res_sa = run_simulated_annealing(sc, gc, walls=active_walls)
        except Exception:
            pass

        telemetry = {
            "active_route": f"{st.session_state.start_loc} -> {st.session_state.goal_loc}",
            "A_star": res_astar,
            "BFS": res_bfs,
            "DFS": res_dfs,
            "Hill_Climbing": res_hc,
            "Simulated_Annealing": res_sa
        }

        system_instruction = (
            "You are Pip, a smart, friendly penguin virtual assistant built into PathSense AI developed for an Artificial Intelligence Lab (AIP203). "
            "You can answer ANY question the user asks—whether about general knowledge, computer science, AI algorithms, mathematics, coding, "
            "or live campus navigation. When the user asks about routes, pathfinding, or search performance, ground your answer in the live "
            "telemetry provided. Do not use emojis in your responses. Keep answers clear, structured, and helpful."
        )

        recent_turns = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_history[-6:]
        )
        full_prompt = (
            f"Live Campus Navigation Telemetry:\n{json.dumps(telemetry, indent=2)}\n\n"
            f"Conversation History:\n{recent_turns}\n\n"
            f"Respond directly to the user's latest message."
        )

        try:
            final_resp = ask_gemini_resilient(
                client,
                contents=full_prompt,
                config=types.GenerateContentConfig(system_instruction=system_instruction)
            )
            assistant_reply = final_resp.text
        except Exception as e:
            assistant_reply = f"Error communicating with Gemini service: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
        st.rerun()
