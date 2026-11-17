import base64
import json
import streamlit as st
import plotly.graph_objects as go
from search_engine import (
    LANDMARKS, DEFAULT_WALLS,
    run_astar, run_bfs, run_dfs, run_hill_climbing, run_simulated_annealing
)

st.set_page_config(page_title="PenguMind | Campus Navigation & Penguin Assistant", page_icon="logo.svg", layout="wide", initial_sidebar_state="expanded")

st.markdown("<style>.block-container{padding-top:3.6rem!important;padding-bottom:1rem!important;max-width:96%}.hero-title{font-size:1.95rem;font-weight:700;color:#f8fafc}.hero-sub{font-size:0.9rem;color:#94a3b8;margin-bottom:1rem}.stat-card{background:#111827;border:1px solid #1e293b;border-radius:10px;padding:0.7rem 0.95rem;margin-bottom:0.6rem}.stat-label{font-size:0.72rem;text-transform:uppercase;color:#94a3b8}.stat-value{font-size:1.25rem;font-weight:600;color:#38bdf8}.stat-sub{font-size:0.76rem;color:#64748b}.stChatInput{margin-bottom:38px!important;padding-right:115px!important}</style>", unsafe_allow_html=True)

def get_svg_uri(path):
    with open(path, "rb") as f:
        return "data:image/svg+xml;base64," + base64.b64encode(f.read()).decode("utf-8")

logo_uri = get_svg_uri("logo.svg")
penguin_uri = get_svg_uri("penguin_avatar.svg")

if "start_loc" not in st.session_state: st.session_state.start_loc = "Main Gate"
if "goal_loc" not in st.session_state: st.session_state.goal_loc = "Central Library"
if "custom_walls" not in st.session_state: st.session_state.custom_walls = set()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "Hi! I am Pip. Ask me anything about campus paths, search algorithms, or any topic on your mind."}]

api_key = ""
try: api_key = st.secrets.get("GROQ_API_KEY", "")
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

@st.cache_data(ttl=3600)
def get_valid_models(_api_key):
    client = genai.Client(api_key=_api_key)
    found = [m.name.replace("models/", "") for m in client.models.list() if "gemini" in m.name and "flash" in m.name and "embedding" not in m.name and "image" not in m.name and "audio" not in m.name]
    preferred = [m for m in ["gemini-3.8-flash", "gemini-3.8-flash-lite", "gemini-2.5-flash-lite"] if m in found]
    return preferred + [m for m in found if m not in preferred]

def fetch_live_web_context(query):
    import urllib.request, urllib.parse, xml.etree.ElementTree as ET, re, html
    results = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PenguMind/1.0)"}

    # 1. Live Google News RSS (never blocked on cloud servers, returns latest 2026 news/scores)
    try:
        rss_url = "https://news.google.com/rss/search?q=" + urllib.parse.quote(query) + "&hl=en-IN&gl=IN&ceid=IN:en"
        req = urllib.request.Request(rss_url, headers=headers)
        with urllib.request.urlopen(req, timeout=6) as resp:
            root = ET.fromstring(resp.read())
            for item in root.findall(".//item")[:5]:
                title = (item.findtext("title") or "").strip()
                pub = (item.findtext("pubDate") or "").strip()
                desc = re.sub(r"<[^>]+>", "", html.unescape(item.findtext("description") or "")).strip()
                if title:
                    results.append(f"[{pub}] {title} — {desc[:180]}")
    except Exception:
        pass

    # 2. Wikipedia Live Search API fallback for general facts
    if len(results) < 2:
        try:
            wiki_url = "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=" + urllib.parse.quote(query) + "&format=json&srlimit=3"
            req = urllib.request.Request(wiki_url, headers=headers)
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for hit in data.get("query", {}).get("search", []):
                    snip = re.sub(r"<[^>]+>", "", html.unescape(hit.get("snippet", ""))).strip()
                    results.append(f"{hit.get('title', '')}: {snip}")
        except Exception:
            pass

    return "\n- ".join(results)

def ask_pip(api_key_val, prompt_text, sys_instruction):
    import urllib.request
    from datetime import date

    live_notes = fetch_live_web_context(prompt_text)
    enriched_sys = f"{sys_instruction}\nCurrent Date: {date.today().strftime('%B %d, %Y')}. You have live internet search enabled. NEVER mention a 2024 knowledge cutoff or say you lack real-time information."
    if live_notes:
        enriched_sys += f"\nLive web search context for this query:\n- {live_notes}\nUse these live facts accurately when answering."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key_val}",
        "User-Agent": "PenguMind/1.0"
    }
    models = ["llama-3.1-8b-instant", "openai/gpt-oss-120b", "qwen/qwen3-32b"]
    try:
        m_req = urllib.request.Request("https://api.groq.com/openai/v1/models", headers=headers, method="GET")
        with urllib.request.urlopen(m_req, timeout=8) as m_resp:
            available = [
                m["id"] for m in json.loads(m_resp.read().decode("utf-8")).get("data", [])
                if not any(x in m["id"].lower() for x in ["whisper", "guard", "tts", "embed", "vision"])
            ]
            if available:
                preferred = [m for m in models if m in available]
                models = preferred + [m for m in available if m not in preferred]
    except Exception:
        pass

    last_err = None
    for model_id in models[:4]:
        try:
            payload = json.dumps({
                "model": model_id,
                "messages": [
                    {"role": "system", "content": enriched_sys},
                    {"role": "user", "content": prompt_text}
                ],
                "temperature": 0.3
            }).encode("utf-8")
            req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=12) as resp:
                return json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            continue
    raise last_err

with st.sidebar:
    st.markdown(f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px;'><img src='{logo_uri}' width='36' height='36'/><span style='font-size:1.15rem;font-weight:700;color:#f8fafc;'>PenguMind Control</span></div>", unsafe_allow_html=True)
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

st.markdown('<div style="display:flex;align-items:center;gap:14px;"><img src="https://raw.githubusercontent.com/sammangupta477/pengumind/main/logo.svg" width="48" height="48"/><div class="hero-title">PenguMind</div></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Interactive State-Space Search Visualizer & Penguin Virtual Assistant (AIP203 Artificial Intelligence Lab)</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.markdown(f"<div class='stat-card'><div class='stat-label'>Active Route</div><div class='stat-value' style='font-size:1rem'>{st.session_state.start_loc} &rarr; {st.session_state.goal_loc}</div><div class='stat-sub'>{sc} to {gc}</div></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='stat-card'><div class='stat-label'>Optimal Cost (A*)</div><div class='stat-value'>{res_astar['path_cost']} steps</div><div class='stat-sub'>Manhattan Heuristic</div></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='stat-card'><div class='stat-label'>Nodes Expanded</div><div class='stat-value'>{res_astar['explored_count']} vs {res_bfs['explored_count']}</div><div class='stat-sub'>A* vs BFS</div></div>", unsafe_allow_html=True)
k4.markdown(f"<div class='stat-card'><div class='stat-label'>Hill Climbing</div><div class='stat-value' style='font-size:1rem'>{res_hc['status']}</div><div class='stat-sub'>Local search</div></div>", unsafe_allow_html=True)

col_map, col_chat = st.columns([1.02, 1.28], gap="large")
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
    st.markdown(f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:6px;'><img src='{penguin_uri}' width='34' height='34'/><span style='font-size:1.15rem;font-weight:600;color:#f8fafc;'>Pip — Penguin Virtual Assistant</span></div>", unsafe_allow_html=True)
    chat_box = st.container(height=365, border=True)
    with chat_box:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar=penguin_uri if msg["role"] == "assistant" else None):
                st.markdown(msg["content"])
    if "pip_pending_prompt" not in st.session_state:
        st.session_state.pip_pending_prompt = ""

    def _submit_pip():
        val = st.session_state.get("pip_msg_box", "").strip()
        if val:
            st.session_state.pip_pending_prompt = val
            st.session_state["pip_msg_box"] = ""

    in_col, btn_col = st.columns([7, 1], vertical_alignment="center")
    in_col.text_input(
        "Message Pip",
        key="pip_msg_box",
        placeholder="Ask Pip anything...",
        label_visibility="collapsed",
        on_change=_submit_pip
    )
    if btn_col.button("", icon=":material/send:", type="primary", use_container_width=True):
        _submit_pip()

    user_query = st.session_state.pip_pending_prompt
    if user_query:
        st.session_state.pip_pending_prompt = ""
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        if detect_route_change(user_query):
            sc, gc = LANDMARKS[st.session_state.start_loc], LANDMARKS[st.session_state.goal_loc]
            res_astar, res_bfs, res_dfs, res_hc, res_sa = run_astar(sc, gc, walls=active_walls), run_bfs(sc, gc, walls=active_walls), run_dfs(sc, gc, walls=active_walls), run_hill_climbing(sc, gc, walls=active_walls), run_simulated_annealing(sc, gc, walls=active_walls)
        if not api_key:
            reply = "Backend API key not set in Streamlit Secrets."
        else:
            tel = {"route": f"{st.session_state.start_loc} -> {st.session_state.goal_loc}", "A_star": res_astar, "BFS": res_bfs, "DFS": res_dfs, "Hill_Climbing": res_hc, "Simulated_Annealing": res_sa}
            sys_i = "You are Pip, a live-connected AI assistant in PenguMind (Current Date: November 17, 2026). Always answer using the live web search context provided. NEVER mention a 2024 knowledge cutoff. Answer ANY question clearly without emojis. Use live telemetry when discussing routes or search algorithms."
            hist = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_history[-6:])
            try:
                reply = ask_pip(api_key, f"Telemetry: {json.dumps(tel)}\n\n{hist}", sys_i)
            except Exception as e:
                import urllib.error
                detail = e.read().decode() if isinstance(e, urllib.error.HTTPError) else str(e)
                reply = f"Groq API error: {detail}"
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()
