import json
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from google import genai
from google.genai import types
from search_engine import (
    LANDMARKS, DEFAULT_WALLS,
    run_astar, run_bfs, run_dfs, run_hill_climbing
)

st.set_page_config(page_title="PathSense AI | Virtual Navigation Assistant", page_icon="🧭", layout="wide")

st.title("🧭 PathSense AI: Intelligent Campus Navigation Assistant")
st.caption("AIP203 Artificial Intelligence Lab • Informed/Uninformed Search Engine + LLM Virtual Assistant")

# Session state initialization
if "start_loc" not in st.session_state:
    st.session_state.start_loc = "Main Gate"
if "goal_loc" not in st.session_state:
    st.session_state.goal_loc = "Central Library"
if "blocked_cells" not in st.session_state:
    st.session_state.blocked_cells = set(DEFAULT_WALLS)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar controls
with st.sidebar:
    st.header("🔑 Assistant Setup")
    api_key = st.text_input("Gemini API Key", type="password", help="Required for Natural Language Assistant mode")
    
    st.divider()
    st.header("📍 Manual Route Controls")
    landmark_names = list(LANDMARKS.keys())
    start_choice = st.selectbox("Start Location", landmark_names, index=landmark_names.index(st.session_state.start_loc))
    goal_choice = st.selectbox("Destination", landmark_names, index=landmark_names.index(st.session_state.goal_loc))
    
    if start_choice != st.session_state.start_loc or goal_choice != st.session_state.goal_loc:
        st.session_state.start_loc = start_choice
        st.session_state.goal_loc = goal_choice

    extra_block = st.checkbox("Simulate Construction Block near Library (3,8)", value=False)
    active_walls = set(DEFAULT_WALLS)
    if extra_block:
        active_walls.add((3, 8))
        active_walls.add((2, 9))

# Execute all 4 classical AI algorithms on current state
start_coords = LANDMARKS[st.session_state.start_loc]
goal_coords = LANDMARKS[st.session_state.goal_loc]

res_astar = run_astar(start_coords, goal_coords, walls=active_walls)
res_bfs = run_bfs(start_coords, goal_coords, walls=active_walls)
res_dfs = run_dfs(start_coords, goal_coords, walls=active_walls)
res_hc = run_hill_climbing(start_coords, goal_coords, walls=active_walls)

# Layout: Left = Live Map & Algorithm Benchmark | Right = Conversational Virtual Assistant
col_map, col_chat = st.columns([1.15, 1])

with col_map:
    st.subheader(f"🗺️ Route: {st.session_state.start_loc} → {st.session_state.goal_loc}")
    
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    grid = np.zeros((12, 12))
    for (r, c) in active_walls:
        grid[r, c] = 1  # Obstacle

    ax.imshow(grid, cmap="Greys", vmin=0, vmax=1.3)
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 12, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", linestyle="-", linewidth=0.7)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    # Plot BFS path (dashed orange) & A* path (solid emerald)
    if res_bfs["path"]:
        by, bx = zip(*res_bfs["path"])
        ax.plot(bx, by, linestyle="--", color="#f59e0b", linewidth=2, label="BFS Path")
    if res_astar["path"]:
        ay, ax_p = zip(*res_astar["path"])
        ax.plot(ax_p, ay, color="#10b981", linewidth=3.2, label="A* Optimal Path")
    if res_hc["path"] and res_hc["path_cost"] == -1:
        hy, hx = zip(*res_hc["path"])
        ax.plot(hx, hy, linestyle=":", color="#ef4444", linewidth=2, label="Hill Climbing (Stuck)")

    # Plot landmarks
    for name, (r, c) in LANDMARKS.items():
        marker_color = "#2563eb"
        size = 65
        if name == st.session_state.start_loc:
            marker_color = "#16a34a"
            size = 130
        elif name == st.session_state.goal_loc:
            marker_color = "#dc2626"
            size = 130
        ax.scatter(c, r, c=marker_color, s=size, zorder=5, edgecolors="white", linewidth=1.2)
        ax.text(c, r - 0.32, name, fontsize=7.5, ha="center", fontweight="bold", color="#111827")

    ax.legend(loc="upper right", fontsize=8)
    st.pyplot(fig)

    st.markdown("### 📊 Live Search Algorithm Comparison")
    st.table([
        {"Algorithm": r["algorithm"], "Status": r["status"], "Path Cost": r["path_cost"], "Nodes Explored": r["explored_count"], "Time (ms)": r["time_ms"]}
        for r in [res_astar, res_bfs, res_dfs, res_hc]
    ])

with col_chat:
    st.subheader("🤖 AI Navigation Virtual Assistant")
    st.markdown(
        "Ask in plain English — e.g., *'Take me from Boys Hostel to Lecture Hall Complex and explain why A* explored fewer nodes than BFS'* "
        "or *'Why did Hill Climbing fail on this route?'*"
    )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask PathSense AI to plan a route or analyze search behavior...")
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        if not api_key:
            reply = "⚠️ Please enter your **Gemini API Key** in the left sidebar to enable natural language assistant responses."
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

        client = genai.Client(api_key=api_key)

        # Step 1: Intent & Entity Extraction (maps natural language to Campus Landmarks)
        extraction_prompt = f"""
        Extract start and destination campus landmarks from the user's message if mentioned.
        Valid landmarks: {list(LANDMARKS.keys())}
        Current start: {st.session_state.start_loc}, Current goal: {st.session_state.goal_loc}
        Return ONLY valid JSON: {{"start": "<landmark>", "goal": "<landmark>"}}
        User message: {user_query}
        """
        try:
            intent_resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=extraction_prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            parsed = json.loads(intent_resp.text)
            if parsed.get("start") in LANDMARKS and parsed.get("goal") in LANDMARKS:
                st.session_state.start_loc = parsed["start"]
                st.session_state.goal_loc = parsed["goal"]
                # Re-run search on newly extracted route
                sc = LANDMARKS[st.session_state.start_loc]
                gc = LANDMARKS[st.session_state.goal_loc]
                res_astar = run_astar(sc, gc, walls=active_walls)
                res_bfs = run_bfs(sc, gc, walls=active_walls)
                res_dfs = run_dfs(sc, gc, walls=active_walls)
                res_hc = run_hill_climbing(sc, gc, walls=active_walls)
        except Exception:
            pass

        # Step 2: Grounded AI Explanation using real search telemetry
        telemetry = {
            "route": f"{st.session_state.start_loc} {LANDMARKS[st.session_state.start_loc]} -> {st.session_state.goal_loc} {LANDMARKS[st.session_state.goal_loc]}",
            "A_star": res_astar,
            "BFS": res_bfs,
            "DFS": res_dfs,
            "Hill_Climbing": res_hc
        }
        advisor_prompt = f"""
        You are PathSense AI, a campus navigation virtual assistant built for an Artificial Intelligence Lab (AIP203).
        Real-time search telemetry computed by our Python search engine:
        {json.dumps(telemetry, indent=2)}

        Answer the user's request clearly, concisely, and accurately using the telemetry above.
        Highlight heuristic efficiency (Manhattan distance), optimality, or local optima behavior where relevant.
        User request: {user_query}
        """
        final_resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=advisor_prompt
        )
        st.session_state.chat_history.append({"role": "assistant", "content": final_resp.text})
        st.rerun()
