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
iteration = 10
while iteration:
    matrix = np.copy(grid)
    for i in range(grid_size):
        for j in range(grid_size):
            state = (i,j)
            if is_terminate(state):
                continue
            # initialize v every iteration
            v = 0
            max_val = -float('inf')
            for dx,dy in action:
                nx,ny = i+dx, j+dy
                if 0<=nx<grid_size and 0<=ny<grid_size:
                    next_state_value = grid[nx][ny]
                    v = (reward + gamma * next_state_value)
                else:
                    next_state_value = grid[i][j]
                    v = (reward + gamma * next_state_value)
                max_val = max(max_val, v)

            matrix[i][j] = max_val
    grid = matrix
    iteration -= 1
    if iteration in {0, 1, 2, 3, 4,5,6,7,8,9}:  # 원하는 k
        print(f"k = {iteration}")
        print(matrix)