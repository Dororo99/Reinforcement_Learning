'''
value evaluation iteration
    - uniform percentage of policy selection
'''
import numpy as np

# settings
grid_size = 4 # change num
gamma = 1.0
reward = -1
action = [(-1,0),(1,0), (0,1),(0,-1)]
pi = 0.25 # uniform policy

# grid
grid = np.zeros((grid_size,grid_size))

# goal
goal = (3,3)

# terminate
def is_terminate(state):
    return state == goal

# iteration policy evaluation
iteration = 3
while iteration:
    matrix = np.copy(grid)
    for i in range(grid_size):
        for j in range(grid_size):
            state = (i,j)
            if is_terminate(state):
                continue
            # initialize v every iteration
            v = 0
            for dx,dy in action:
                nx,ny = i+dx, j+dy
                if 0<=nx<grid_size and 0<=ny<grid_size:
                    next_state_value = grid[nx][ny]
                    v += pi * (reward + gamma * next_state_value)
                else:
                    next_state_value = grid[i][j]
                    v += pi * (reward + gamma * next_state_value)
            matrix[i][j] = v
    grid = matrix
    iteration -= 1
    if iteration in {0, 1, 2, 3, 50}:  # 원하는 k
        print(f"k = {iteration}")
        print(matrix)