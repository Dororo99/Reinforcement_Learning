import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim

import numpy as np

# Set HyperParams
LR = 0.0002
GAMMA = 0.98
LAMBDA = 0.95
T_horizon = 20
K_epoch = 10000
eps_clip = 0.1

# Set Model
class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(),lr=LR)
    def pi(self,x,softmax_dim=0):
        x = F.relu(self.fc1(x))
        prob = F.softmax(self.fc_pi(x),dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst,prob_a_lst, done_lst = [],[],[],[],[],[]
        for transition in self.data:
            s,a,r,s_prime,prob_a,done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r/100.0])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s,a,r,s_prime,prob_a,done_mask = torch.tensor(s_lst,dtype= torch.float),torch.tensor(a_lst),torch.tensor(r_lst),\
        torch.tensor(s_prime_lst,dtype=torch.float), torch.tensor(prob_a_lst),torch.tensor(done_lst, dtype = torch.float)

        self.data = []
        return s,a,r,s_prime,prob_a,done_mask

"""
Train
    - Args
        - s
        - a
        - r
        - s_prime
        - prob_a
        - done_mask
"""
def train_net(self):
    s,a,r,s_prime,prob_a,done_mask = self.make_batch()

    for _ in range(K_epoch):
        td_target = r + GAMMA * self.v(s_prime) * done_mask
        delta = td_target - self.v(s)
        delta = delta.detach().numpy() # make delta numpy

        advantage_lst = []
        advantage = 0.0

    for delta_t in delta[::-1]:
        advantage = GAMMA * LAMBDA * advantage + delta_t[0]
        advantage_lst.append([advantage])
    advantage_lst.reverse()
    advantage = torch.tensor(advantage_lst, dtype=torch.float)

    pi = self.pi(s,softmax_dim=1)
    pi_a = pi.gather(1,a)
    ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a)) # r(t)_theta

    # Get Actor Loss
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip)*advantage
    actor_loss = -torch.min(surr1,surr2)
    critic_loss = F.smooth_l1_loss(self.v(s),td_target.detach())
    loss = actor_loss + critic_loss

    self.optimizer.zero_grad()
    loss.mean().backward()
    self.optimizer.step()

"""
main
"""
def main():
    env = gym.make("CartPole-v1")
    model = PPO()

    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s = env.reset()
        done = False

    while not done:
        for t in range(T_horizon):
            prob = model.pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample()
            s_prime, r, done, truncated, info = env.step(a)

            model.put_data((s,a,r,s_prime,prob[a].item(),done))
            # s_lst, a_lst, r_lst, s_prime_lst,prob_a_lst, done_lst = [],[],[],[],[],[]
            s = s_prime

            score += r
            if done:
                break

        model.train_net()

        if n_epi%print_interval == 0 and n_epi!=0:
            print(f"[Episode]: {n_epi}, [Avg_Score]: {(score/print_interval):.2f}")
            score = 0.0
    env.close()

if __name__ == "__main__":
    main()