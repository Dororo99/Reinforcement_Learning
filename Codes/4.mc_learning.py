'''
monte carlo learning(=mc learning)
    - v_pi(state_t) = expectation of G_t
    - v_pi(state_t) <- v_pi(state)-alpha*(G_t - v_pi(state_t))
'''
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
    data = np.zeros((grid_size,grid_size))
    gamma = 1.0
    reward = -1
    # alpha = 0.001
    iteration = 10000

    for k in range(iteration):
        done = False
        history = []

        while not done:
            action = agent.select_action()
            (x,y), reward, done = env.step(action)
            history.append((x,y,reward))
        env.reset()

        cum_reward = 0
        for transition in history[::-1]:
            x,y,reward = transition
            data[x][y] = data[x][y] +