import heapq
import time
from collections import deque

# Campus Landmark Coordinates on a 12x12 Grid (row, col)
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

# Default Static Obstacles (Buildings/Walls)
DEFAULT_WALLS = {
    (2, 2), (2, 3), (2, 4), (4, 6), (4, 7), (4, 8),
    (5, 6), (7, 4), (7, 5), (8, 8), (8, 9), (10, 4), (10, 5)
}

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(pos, grid_size, walls):
    r, c = pos
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    valid = []
    for dr, dc in moves:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size and (nr, nc) not in walls:
            valid.append((nr, nc))
    return valid

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

def run_astar(start, goal, grid_size=12, walls=DEFAULT_WALLS):
    t0 = time.perf_counter()
    open_set = []
    heapq.heappush(open_set, (manhattan(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    explored = []

    while open_set:
        _, current_g, current = heapq.heappop(open_set)
        if current in explored:
            continue
        explored.append(current)

        if current == goal:
            elapsed = (time.perf_counter() - t0) * 1000
            return {
                "algorithm": "A* Search (Manhattan Heuristic)",
                "path": reconstruct_path(came_from, current),
                "explored_count": len(explored),
                "path_cost": current_g,
                "time_ms": round(elapsed, 3),
                "status": "Optimal Path Found"
            }

        for neighbor in get_neighbors(current, grid_size, walls):
            tentative_g = current_g + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + manhattan(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))

    return {"algorithm": "A* Search", "path": [], "explored_count": len(explored), "path_cost": -1, "time_ms": 0, "status": "No Path Found"}

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
            elapsed = (time.perf_counter() - t0) * 1000
            path = reconstruct_path(came_from, current)
            return {
                "algorithm": "Breadth-First Search (BFS)",
                "path": path,
                "explored_count": len(explored),
                "path_cost": len(path) - 1,
                "time_ms": round(elapsed, 3),
                "status": "Optimal Path Found"
            }
        for neighbor in get_neighbors(current, grid_size, walls):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

    return {"algorithm": "BFS", "path": [], "explored_count": len(explored), "path_cost": -1, "time_ms": 0, "status": "No Path Found"}

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
            elapsed = (time.perf_counter() - t0) * 1000
            path = reconstruct_path(came_from, current)
            return {
                "algorithm": "Depth-First Search (DFS)",
                "path": path,
                "explored_count": len(explored),
                "path_cost": len(path) - 1,
                "time_ms": round(elapsed, 3),
                "status": "Path Found (Suboptimal)"
            }
        for neighbor in get_neighbors(current, grid_size, walls):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append(neighbor)

    return {"algorithm": "DFS", "path": [], "explored_count": len(explored), "path_cost": -1, "time_ms": 0, "status": "No Path Found"}

def run_hill_climbing(start, goal, grid_size=12, walls=DEFAULT_WALLS):
    t0 = time.perf_counter()
    current = start
    path = [current]
    visited = {current}

    while current != goal:
        neighbors = [n for n in get_neighbors(current, grid_size, walls) if n not in visited]
        if not neighbors:
            elapsed = (time.perf_counter() - t0) * 1000
            return {
                "algorithm": "Steepest-Ascent Hill Climbing",
                "path": path,
                "explored_count": len(visited),
                "path_cost": -1,
                "time_ms": round(elapsed, 3),
                "status": "Stuck in Local Optimum / Dead End"
            }
        best_next = min(neighbors, key=lambda n: manhattan(n, goal))
        if manhattan(best_next, goal) >= manhattan(current, goal):
            elapsed = (time.perf_counter() - t0) * 1000
            return {
                "algorithm": "Steepest-Ascent Hill Climbing",
                "path": path,
                "explored_count": len(visited),
                "path_cost": -1,
                "time_ms": round(elapsed, 3),
                "status": "Stuck in Local Optimum (Plateau/Obstacle)"
            }
        current = best_next
        visited.add(current)
        path.append(current)

    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "algorithm": "Steepest-Ascent Hill Climbing",
        "path": path,
        "explored_count": len(visited),
        "path_cost": len(path) - 1,
        "time_ms": round(elapsed, 3),
        "status": "Goal Reached"
    }
