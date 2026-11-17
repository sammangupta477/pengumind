import heapq
import math
import random
import time
from collections import deque

LANDMARKS = {
    "Main Gate": (0, 0),
    "Admin Block": (1, 5),
    "Central Library": (3, 9),
    "Computer Center": (5, 3),
    "Lecture Hall Complex": (6, 7),
    "Cafeteria": (8, 2),
    "Sports Complex": (9, 10),
    "Boys Hostel": (11, 1),
    "Girls Hostel": (11, 8),
}

DEFAULT_WALLS = {
    (2, 2), (2, 3), (2, 4), (4, 6), (4, 7), (4, 8),
    (5, 6), (7, 4), (7, 5), (8, 8), (8, 9), (10, 4), (10, 5)
}

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(pos, grid_size, walls):
    r, c = pos
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return [
        (r + dr, c + dc)
        for dr, dc in moves
        if 0 <= r + dr < grid_size and 0 <= c + dc < grid_size and (r + dr, c + dc) not in walls
    ]

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

def run_astar(start, goal, grid_size=12, walls=DEFAULT_WALLS):
    t0 = time.perf_counter()
    open_set = [(manhattan(start, goal), 0, start)]
    came_from = {}
    g_score = {start: 0}
    explored = []

    while open_set:
        _, current_g, current = heapq.heappop(open_set)
        if current in explored:
            continue
        explored.append(current)
        if current == goal:
            return {
                "algorithm": "A* Search (Manhattan)",
                "path": reconstruct_path(came_from, current),
                "explored_count": len(explored),
                "path_cost": current_g,
                "time_ms": round((time.perf_counter() - t0) * 1000, 3),
                "status": "Optimal Path Found"
            }
        for neighbor in get_neighbors(current, grid_size, walls):
            tentative_g = current_g + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g + manhattan(neighbor, goal), tentative_g, neighbor))

    return {"algorithm": "A* Search (Manhattan)", "path": [], "explored_count": len(explored), "path_cost": -1, "time_ms": 0, "status": "No Path"}

def run_bfs(start, goal, grid_size=12, walls=DEFAULT_WALLS):
    t0 = time.perf_counter()
    queue = deque([start])
    came_from = {}
    visited = {start}
    explored = []

    while queue:
        current = queue.popleft()
        explored.append(current)
        if current == goal:
            path = reconstruct_path(came_from, current)
            return {
                "algorithm": "Breadth-First Search (BFS)",
                "path": path,
                "explored_count": len(explored),
                "path_cost": len(path) - 1,
                "time_ms": round((time.perf_counter() - t0) * 1000, 3),
                "status": "Optimal Path Found"
            }
        for neighbor in get_neighbors(current, grid_size, walls):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

    return {"algorithm": "BFS", "path": [], "explored_count": len(explored), "path_cost": -1, "time_ms": 0, "status": "No Path"}

def run_dfs(start, goal, grid_size=12, walls=DEFAULT_WALLS):
    t0 = time.perf_counter()
    stack = [start]
    came_from = {}
    visited = {start}
    explored = []

    while stack:
        current = stack.pop()
        explored.append(current)
        if current == goal:
            path = reconstruct_path(came_from, current)
            return {
                "algorithm": "Depth-First Search (DFS)",
                "path": path,
                "explored_count": len(explored),
                "path_cost": len(path) - 1,
                "time_ms": round((time.perf_counter() - t0) * 1000, 3),
                "status": "Suboptimal Path"
            }
        for neighbor in get_neighbors(current, grid_size, walls):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append(neighbor)

    return {"algorithm": "DFS", "path": [], "explored_count": len(explored), "path_cost": -1, "time_ms": 0, "status": "No Path"}

def run_hill_climbing(start, goal, grid_size=12, walls=DEFAULT_WALLS):
    t0 = time.perf_counter()
    current = start
    path = [current]
    visited = {current}

    while current != goal:
        neighbors = [n for n in get_neighbors(current, grid_size, walls) if n not in visited]
        if not neighbors:
            return {
                "algorithm": "Hill Climbing",
                "path": path,
                "explored_count": len(visited),
                "path_cost": -1,
                "time_ms": round((time.perf_counter() - t0) * 1000, 3),
                "status": "Stuck (Dead End)"
            }
        best_next = min(neighbors, key=lambda n: manhattan(n, goal))
        if manhattan(best_next, goal) >= manhattan(current, goal):
            return {
                "algorithm": "Hill Climbing",
                "path": path,
                "explored_count": len(visited),
                "path_cost": -1,
                "time_ms": round((time.perf_counter() - t0) * 1000, 3),
                "status": "Stuck (Local Optimum)"
            }
        current = best_next
        visited.add(current)
        path.append(current)

    return {
        "algorithm": "Hill Climbing",
        "path": path,
        "explored_count": len(visited),
        "path_cost": len(path) - 1,
        "time_ms": round((time.perf_counter() - t0) * 1000, 3),
        "status": "Goal Reached"
    }

def run_simulated_annealing(start, goal, grid_size=12, walls=DEFAULT_WALLS):
    t0 = time.perf_counter()
    rng = random.Random(42)
    current = start
    path = [current]
    visited = {current}
    temp = 10.0

    for _ in range(250):
        if current == goal:
            return {
                "algorithm": "Simulated Annealing",
                "path": path,
                "explored_count": len(visited),
                "path_cost": len(path) - 1,
                "time_ms": round((time.perf_counter() - t0) * 1000, 3),
                "status": "Goal Reached"
            }
        neighbors = get_neighbors(current, grid_size, walls)
        if not neighbors:
            break
        nxt = rng.choice(neighbors)
        delta_e = manhattan(nxt, goal) - manhattan(current, goal)
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(temp, 0.05)):
            current = nxt
            visited.add(current)
            path.append(current)
        temp *= 0.96

    return {
        "algorithm": "Simulated Annealing",
        "path": path,
        "explored_count": len(visited),
        "path_cost": -1,
        "time_ms": round((time.perf_counter() - t0) * 1000, 3),
        "status": "Cooling Limit Reached"
    }
