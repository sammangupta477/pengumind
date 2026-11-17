import json
import streamlit as st
import plotly.graph_objects as go
from google import genai
from google.genai import types
from search_engine import (
    LANDMARKS, DEFAULT_WALLS,
    run_astar, run_bfs, run_dfs, run_hill_climbing, run_simulated_annealing
)

st.set_page_config(page_title="PathSense AI | Campus Navigation & Penguin Assistant", page_icon="logo.svg", layout="wide", initial_sidebar_state="expanded")

st.markdown("<style>.block-container{padding-top:3.6rem!important;padding-bottom:1rem!important;max-width:96%}.hero-title{font-size:1.95rem;font-weight:700;color:#f8fafc}.hero-sub{font-size:0.9rem;color:#94a3b8;margin-bottom:1rem}.stat-card{background:#111827;border:1px solid #1e293b;border-radius:10px;padding:0.7rem 0.95rem;margin-bottom:0.6rem}.stat-label{font-size:0.72rem;text-transform:uppercase;color:#94a3b8}.stat-value{font-size:1.25rem;font-weight:600;color:#38bdf8}.stat-sub{font-size:0.76rem;color:#64748b}</style>", unsafe_allow_html=True)

if "start_loc" not in st.session_state: st.session_state.start_loc = "Main Gate"
if "goal_loc" not in st.session_state: st.session_state.goal_loc = "Central Library"
if "custom_walls" not in st.session_state: st.session_state.custom_walls = set()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "Hello! I am Pip, your PathSense AI Penguin Assistant. Ask me anything—from planning campus routes and comparing A*, BFS, DFS, Hill Climbing, and Simulated Annealing, to general AI, math, or coding questions."}]

api_key = ""
try: api_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception: api_key = ""

def detect_route_change(query_text):
    q = query_text.lower()
    mentioned = sorted([(q.find(name.lower()), name) for name in LANDMARKS if name.lower() in q])
    if len(mentioned) >= 2:
        st.session_state.start_loc, st.session_state.goal_loc = mentioned[0][1], mentioned[1][1]
        return True
    if len(mentioned) == 1:
        lm = mentioned[0][1]
        if f"from {lm.lower()}" in q: st.session_state.start_loc = lm
        else: st.session_state.goal_loc = lm
        return True
    return False

def ask_pip(api_key_val, prompt_text, sys_instruction):
    client = genai.Client(api_key=api_key_val, http_options=types.HttpOptions(timeout=12000))
    last_err = None
    for m in ["gemini-3.8-flash-lite", "gemini-3.8-flash", "gemini-flash-latest"]:
        try:
            resp = client.models.generate_content(model=m, contents=prompt_text, config=types.GenerateContentConfig(system_instruction=sys_instruction))
            if resp and resp.text: return resp.text
        except Exception as e:
            last_err = e
    raise RuntimeError(str(last_err))

with st.sidebar:
    st.markdown("### PathSense Control")
    landmark_names = list(LANDMARKS.keys())
    s_ch = st.selectbox("Origin Landmark", landmark_names, index=landmark_names.index(st.session_state.start_loc))
    g_ch = st.selectbox("Destination Landmark", landmark_names, index=landmark_names.index(st.session_state.goal_loc))
    if s_ch != st.session_state.start_loc or g_ch != st.session_state.goal_loc:
        st.session_state.start_loc, st.session_state.goal_loc = s_ch, g_ch
    st.markdown("---")
    show_astar = st.checkbox("A* Optimal Path (Manhattan)", value=True)
    show_bfs = st.checkbox("Breadth-First Search (BFS)", value=True)
    show_dfs = st.checkbox("Depth-First Search (DFS)", value=False)
    show_hc = st.checkbox("Hill Climbing", value=True)
    show_sa = st.checkbox("Simulated Annealing", value=False)
    st.markdown("---")
    extra_block = st.toggle("Library Corridor Blockade", value=False)
    custom_coord = st.selectbox("Add Custom Barrier (Row, Col)", ["None"] + [f"({r}, {c})" for r in range(12) for c in range(12) if (r, c) not in LANDMARKS.values()])
    c1, c2 = st.columns(2)
    if c1.button("Add Barrier", use_container_width=True) and custom_coord != "None":
        st.session_state.custom_walls.add(eval(custom_coord)); st.rerun()
    if c2.button("Reset Grid", use_container_width=True):
        st.session_state.custom_walls = set(); st.rerun()
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.chat_history = [st.session_state.chat_history[0]]; st.rerun()

active_walls = set(DEFAULT_WALLS) | st.session_state.custom_walls | ({(3, 8), (2, 9)} if extra_block else set())
sc, gc = LANDMARKS[st.session_state.start_loc], LANDMARKS[st.session_state.goal_loc]
res_astar, res_bfs, res_dfs, res_hc, res_sa = run_astar(sc, gc, walls=active_walls), run_bfs(sc, gc, walls=active_walls), run_dfs(sc, gc, walls=active_walls), run_hill_climbing(sc, gc, walls=active_walls), run_simulated_annealing(sc, gc, walls=active_walls)

st.markdown('<div style="display:flex;align-items:center;gap:14px;"><img src="https://raw.githubusercontent.com/sammangupta477/pathsense-ai-assistant/main/logo.svg" width="48" height="48"/><div class="hero-title">PathSense AI</div></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Interactive State-Space Search Visualizer & Penguin Virtual Assistant (AIP203 Artificial Intelligence Lab)</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.markdown(f"<div class='stat-card'><div class='stat-label'>Active Route</div><div class='stat-value' style='font-size:1rem'>{st.session_state.start_loc} &rarr; {st.session_state.goal_loc}</div><div class='stat-sub'>{sc} to {gc}</div></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='stat-card'><div class='stat-label'>Optimal Cost (A*)</div><div class='stat-value'>{res_astar['path_cost']} steps</div><div class='stat-sub'>Manhattan Heuristic</div></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='stat-card'><div class='stat-label'>Nodes Expanded</div><div class='stat-value'>{res_astar['explored_count']} vs {res_bfs['explored_count']}</div><div class='stat-sub'>A* vs BFS</div></div>", unsafe_allow_html=True)
k4.markdown(f"<div class='stat-card'><div class='stat-label'>Hill Climbing</div><div class='stat-value' style='font-size:1rem'>{res_hc['status']}</div><div class='stat-sub'>Local search</div></div>", unsafe_allow_html=True)

col_map, col_chat = st.columns([1.18, 1], gap="large")
with col_map:
    t_map, t_tab = st.tabs(["Interactive Campus Grid", "Algorithm Benchmark Table"])
    with t_map:
        fig = go.Figure()
        gx, gy, ht = [], [], []
        for r in range(12):
            for c in range(12):
                gx.append(c); gy.append(r)
                ct = "Obstacle" if (r, c) in active_walls else "Walkable"
                for ln, lc in LANDMARKS.items():
                    if lc == (r, c): ct = f"Landmark: {ln}"
                ht.append(f"Row {r}, Col {c}<br>{ct}")
        fig.add_trace(go.Scatter(x=gx, y=gy, mode="markers", marker=dict(size=21, color="#1e293b", symbol="square", line=dict(width=1, color="#334155")), text=ht, hoverinfo="text", showlegend=False))
        if active_walls:
            wy, wx = zip(*active_walls)
            fig.add_trace(go.Scatter(x=wx, y=wy, mode="markers", name="Obstacles", marker=dict(size=21, color="#475569", symbol="square"), hoverinfo="skip"))
        for en, ro, lb, cl, ds, wd in [(show_dfs, res_dfs, "DFS", "#a855f7", "dot", 2.5), (show_sa, res_sa, "Simulated Annealing", "#ec4899", "dashdot", 2.5), (show_bfs, res_bfs, "BFS", "#f59e0b", "dash", 3), (show_hc, res_hc, "Hill Climbing", "#ef4444", "dot", 3), (show_astar, res_astar, "A* Optimal", "#10b981", "solid", 4.5)]:
            if en and ro["path"]:
                py, px = zip(*ro["path"])
                fig.add_trace(go.Scatter(x=px, y=py, mode="lines+markers", name=lb, line=dict(color=cl, width=wd, dash=ds), marker=dict(size=6, color=cl)))
        for nm, (r, c) in LANDMARKS.items():
            cl = "#22c55e" if nm == st.session_state.start_loc else ("#ef4444" if nm == st.session_state.goal_loc else "#38bdf8")
            fig.add_trace(go.Scatter(x=[c], y=[r], mode="markers+text", text=[nm], textposition="top center", textfont=dict(size=10, color="#f8fafc"), marker=dict(size=14 if nm in (st.session_state.start_loc, st.session_state.goal_loc) else 10, color=cl), showlegend=False))
        fig.update_layout(height=440, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="#0b1120", plot_bgcolor="#0f172a", xaxis=dict(range=[-0.7, 11.7], showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(range=[11.7, -0.7], showgrid=False, zeroline=False, showticklabels=False), legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, font=dict(size=11, color="#cbd5e1")))
        st.plotly_chart(fig, use_container_width=True)
    with t_tab:
        st.dataframe([{"Algorithm": r["algorithm"], "Status": r["status"], "Cost": r["path_cost"] if r["path_cost"] != -1 else "N/A", "Nodes": r["explored_count"], "Time (ms)": r["time_ms"]} for r in [res_astar, res_bfs, res_dfs, res_hc, res_sa]], use_container_width=True, hide_index=True)

with col_chat:
    st.markdown("#### Pip — Penguin Virtual Assistant")
    chat_box = st.container(height=350, border=True)
    with chat_box:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="penguin_avatar.svg" if msg["role"] == "assistant" else None):
                st.markdown(msg["content"])
    user_query = st.chat_input("Ask Pip anything: plan a route, compare algorithms, or ask any question...")
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        if detect_route_change(user_query):
            sc, gc = LANDMARKS[st.session_state.start_loc], LANDMARKS[st.session_state.goal_loc]
            res_astar, res_bfs, res_dfs, res_hc, res_sa = run_astar(sc, gc, walls=active_walls), run_bfs(sc, gc, walls=active_walls), run_dfs(sc, gc, walls=active_walls), run_hill_climbing(sc, gc, walls=active_walls), run_simulated_annealing(sc, gc, walls=active_walls)
        if not api_key:
            reply = "Backend API key not set in Streamlit Secrets."
        else:
            tel = {"route": f"{st.session_state.start_loc} -> {st.session_state.goal_loc}", "A_star": res_astar, "BFS": res_bfs, "DFS": res_dfs, "Hill_Climbing": res_hc, "Simulated_Annealing": res_sa}
            sys_i = "You are Pip, a smart penguin virtual assistant in PathSense AI (AIP203 AI Lab). Answer ANY question clearly without emojis. Use live telemetry when discussing routes or search algorithms."
            hist = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_history[-6:])
            try:
                reply = ask_pip(api_key, f"Telemetry: {json.dumps(tel)}\n\n{hist}", sys_i)
            except Exception as e:
                reply = f"Route {st.session_state.start_loc} -> {st.session_state.goal_loc}: A* optimal cost is {res_astar['path_cost']} steps ({res_astar['explored_count']} nodes vs {res_bfs['explored_count']} for BFS). (Note: {e})"
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()
