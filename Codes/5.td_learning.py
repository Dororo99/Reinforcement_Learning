import numpy as np
import random

np.random.seed(42)

class GridWorld():
    def __init__(self):
        self.x=0
        self.y=0
    def step(self,a):
        if a==0:
            self.move_left()
        elif a==1:
            self.move_up()
        elif a==2:
            self.move_right()
        elif a==3:
            self.move_down()

        # setting
        reward = -1
        done = self.is_done()
        return (self.x,self.y), reward, done
    def move_left(self):
        self.x -= 1
        if self.x < 0:
            self.x = 0
    def move_up(self):
        self.y += 1
        if self.y > 3:
            self.y = 3
    def move_right(self):
        self.x += 1
        if self.x > 3:
            self.x = 3
    def move_down(self):
        self.y -= 1
        if self.y < 0:
            self.y = 0
    def is_done(self):
        if self.x == 3 and self.y == 3:
            return True
        else:
            return False
    def get_state(self):
        return (self.x, self.y)
    def reset(self):
        self.x=0
        self.y=0
        return (self.x,self.y)

class Agent():
    def __init__(self):
        pass
    def select_action(self):
        coin = random.random()
        if coin<0.25:
            action = 0
        elif coin <0.5:
            action = 1
        elif coin < 0.75:
            action = 2
        else:
            action = 3
        return action

def main():
    env = GridWorld()
    agent = Agent()
    grid_size = 4
    map = np.zeros((grid_size,grid_size))
    gamma = 1.0
    reward = -1
    alpha = 0.01

    iterations = 10
    for k in range(iterations):
        done = False
        while not done:
            x, y = env.get_state()
            action = agent.select_action()
            (x_prime,y_prime), reward, done = env.step(action)
            map[x][y] += alpha * (reward + gamma * map[x_prime][y_prime] - map[x][y])
        env.reset()

    for row in map:
        print(map)

if __name__ == '__main__':
    main()

