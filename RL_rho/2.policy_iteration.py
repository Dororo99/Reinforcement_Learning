import numpy as np

# settings
grid_size = 4
gamma = 1.0
reward = -1
actions = [(-1,0), (1,0), (0,1), (0,-1)]

# grid
grid = np.zeros((grid_size, grid_size))

# goal
goal = (3,3)

# terminate
def is_terminal(state):
    return state == goal

# value iteration with greedy policy
iteration = 10
for k in range(iteration):
    new_grid = np.copy(grid)
    for i in range(grid_size):
        for j in range(grid_size):
            state = (i, j)
            if is_terminal(state):
                continue

            max_value = -float('inf')
            for dx, dy in actions:
                nx, ny = i + dx, j + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    value = reward + gamma * grid[nx][ny]
                    max_value = max(max_value, value)
            new_grid[i][j] = max_value
    grid = new_grid

    if k in {0, 1, 2, 3, 9}:
        print(f"\nk = {k + 1}")
        print(np.round(grid, 2))