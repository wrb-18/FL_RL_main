from ast import While
import copy
from pickle import FALSE
from xmlrpc.client import Server
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from options import args_parser
from FL.FL import Federate_learing as FL
from tqdm import tqdm
import random

# 类定义
    
class Simple_test2():
    def __init__(self):
        self.action_space = 1
        self.state_space = 11
        self.state = [0] * self.state_space
        self.observation_space =self.state_space
        self.correct = 0
        self.total = 0
        self.test = 0
        self.action_bound = [0, 9.4]
    def reset(self):
        for i in range(self.state_space - 1):
            self.state[i] = random.random() * 10
        self.state[-1] = 0

        return copy.copy(self.state)
    def step(self, action):
        self.total += 1
        self.state[-1] = action[0]
        
        max_val = self.state[0]
        max_i = 0
        
        # 为了调试
        temp = self.state[1 : -1]
        for i, val in enumerate(temp):
           if val > max_val:
               max_val = val
               max_i = i + 1
               
        rewards = []
        reward = -(self.state[-1] - max_i) ** 2
        if abs(self.state[round(self.state[-1])] - round(max_val, 3)) < 1:
            self.correct += 1
        if abs(self.state[round(random.random() * 9.4)] - round(max_val, 3)) < 1:
            self.test += 1
        #reward = -(self.state[-1] - 3.4) ** 2
        rewards.append(reward)
        return copy.copy(self.state), rewards
            
        
    def render(self):
        print("diff", self.test / self.total, self.correct / self.total)
        # max_val = self.state[0]
        # max_i = 0
        
        # # 为了调试
        # temp = self.state[1 : -1]
        # for i, val in enumerate(temp):
        #    if val > max_val:
        #        max_val = val
        #        max_i = i + 1
        # print("diff")
        # for i in range(self.state_space - 1):
        #     print(i, self.state[i])
        # print("")
        # print("output ans", self.state[round(self.state[-1])], round(max_val, 3))
        #print("output ans", self.state[-1], max_i)
    
                

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1):
        super(ActorNetwork, self).__init__()

        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_dim)
        self.max_action = max_action
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        x = torch.tanh(x)
        x = ((x + 1) / 2)
        # x = x * (self.max_action + 1) - 1
        # x = x * self.max_action
        return x

    def choose_action(self, s):
        s = torch.FloatTensor(s)
        return self.forward(s).detach().cpu()  # single action


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x, u):
        temp_ = torch.cat((x, u), 1)
        temp = self.l1(temp_)
        x = F.relu(temp)
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class Memory(object):
    def __init__(self, capacity, state_dim, action_dim, num_agents):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.dims = 2 * state_dim + action_dim + self.num_agents
        self.data = np.zeros((capacity, self.dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= n, 'Memory has not been fulfilled'
        indices = np.random.choice(min(self.capacity, self.pointer), size=n)
        b_M = self.data[indices, :]
        b_s = b_M[:, :self.state_dim]
        b_a = b_M[:, self.state_dim: self.state_dim + self.action_dim]
        b_r = b_M[:, -self.state_dim - self.num_agents: -self.state_dim]
        b_s_ = b_M[:, -self.state_dim:]
    
        states = torch.FloatTensor(b_s) # 转换成tensor类型
        actions = torch.FloatTensor(b_a)
        rewards = torch.FloatTensor(b_r)
        states_ = torch.FloatTensor(b_s_)
        return states, actions, rewards, states_


class Agent:
    def __init__(self, state_dim, action_dim, id, args, max_action):
        self.id = id
        self.target_actor = ActorNetwork(state_dim, action_dim, max_action=max_action)
        # 一直忘加了
        self.actor = ActorNetwork(state_dim, action_dim, max_action=max_action)
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        
        self.args = args
    #用于test
    # def observation(self, state):
    #     if type(state) == list:
    #         return state
    #     else:
    #         return state[:,]

    #用于FL
    def observation(self, state):
        
        if type(state) == list:
            state = copy.copy(state)
            # obs = []
            # obs.append(state[self.id])
            # for i in range(self.args.num_edges):
            #     # if state[self.args.num_clients + i] < 0:
            #     #     state[self.args.num_clients + i] = state[self.id]
            #     # else:
            #     obs.append(state[self.args.num_clients + i])               
            # obs.append(state[self.args.num_clients + self.args.num_edges])
            # obs.append(state[self.args.num_clients + self.args.num_edges + self.id])

        else:
            state = state.clone()
            # obs = torch.FloatTensor([])
            
            # # state[state[:, self.args.num_clients + i] < 0, self.args.num_clients + i] = state[self.id]
            # obs = torch.cat((obs, state[:, self.id:self.id + 1]), 1)
            # obs = torch.cat((obs, state[:, self.args.num_clients: self.args.num_clients + self.args.num_edges]), 1)
            # obs = torch.cat((obs, state[:, self.args.num_clients + self.args.num_edges: self.args.num_clients + self.args.num_edges + 1]), 1)
            # obs = torch.cat((obs, state[:, self.args.num_clients + self.args.num_edges + self.id: self.args.num_clients + self.args.num_edges + self.id + 1]), 1)
        return state



class Server:
    def __init__(self, state_dim, action_dim, num_of_agents):
        self.critic_list = [CriticNetwork(state_dim, action_dim) for i in range(num_of_agents)]
        self.target_critic_list = [CriticNetwork(state_dim, action_dim) for i in range(num_of_agents)]
        self.num_of_agents = num_of_agents
        self.state_dim = state_dim
        self.action_dim = action_dim


def train(args):
    # 参数定义
    RENDER_INTERVAL = args.render_interval
    
    # 是否打印actor_loss 和td_error
    PRINT_LOSS = True
    
    static_weights = [random.random() for i in range(10)]
    var = 0.3 # control exploration
    for ep in tqdm(range(args.max_episodes)):
        s = env.reset()
        env_GREEN.reset()
        env_BLUE.reset()
        
        # weights = [random.random() for i in range(env.num_clients)]
        
        weights = [0.3, 0.3, 0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7]
        ups = env_GREEN.clients + env_GREEN.edges
        for i, up in enumerate(ups):
            up.weight = weights[i]

        downs = env_BLUE.clients + env_BLUE.edges
        for i, down in enumerate(downs):
            down.weight = static_weights[i]

        ep_reward = 0
        for t in range(args.max_ep_step):

            var = max([var * .995, 0])  # decay the action randomness
            actions = [] 
            for i, agent in enumerate(agents_list):
                a = agent.actor.choose_action(agent.observation(s))
                
                a = np.clip(np.random.normal(a, var), *env.action_bound)  # add randomness to action selection for exploration
                #a = np.clip(np.random.normal(a, var), -0.5, 0.5)
                actions += a.tolist()
                
            s_, r = env.step(actions)
            #actions_ = [0] * env.action_space
            env_GREEN.step([0.3, 0.3, 0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7])
            
            env_BLUE.step(static_weights)
            
            env.render()
            env_GREEN.render()
            env_BLUE.render()
            
            M.store_transition(s, actions, r, s_)
            s = s_
            ep_reward += sum(r)
            # if t % RENDER_INTERVAL == 0:
                
                # env_GREEN.render()
                # env_BLUE.render()
            if M.pointer <= args.rl_batch_size or ep % 2:
                continue
            for i, agent in enumerate(agents_list):
                states, actions, rewards, states_ = M.sample(args.rl_batch_size)
                server.critic_list[i].optimizer.zero_grad()
                actions_ = torch.FloatTensor([]) 
                for i_, agent_ in enumerate(agents_list):
                    a_ = agent_.target_actor.forward(agent_.observation(states_))
                    actions_ = torch.cat((actions_, a_), 1)
                y_ = rewards[:, i:i+1] + args.gamma * server.target_critic_list[i].forward(states_, actions_)
                y = server.critic_list[i].forward(states, actions)
                td_error = F.mse_loss(y_.detach(), y)
                if ep % RENDER_INTERVAL == 0 and PRINT_LOSS:
                    print("td_error", td_error)
                td_error.backward()
                torch.nn.utils.clip_grad_norm_(server.critic_list[i].parameters(), 0.5)
                server.critic_list[i].optimizer.step()

                agent.actor.optimizer.zero_grad()
                _actions = torch.FloatTensor([]) 
                temp = 0
                for i_, agent_ in enumerate(agents_list):
                    if i_ == i:
                        temp = agent_.actor.forward(agent_.observation(states))
                        a = temp
                    else: 
                        a = actions[:, i_ * agent.action_dim : (i_ + 1) * agent.action_dim]
                    _actions = torch.cat((_actions, a), 1)
                loss = server.critic_list[i].forward(states, _actions)
                actor_loss = -torch.mean(loss)
                actor_loss += (temp ** 2).mean() * 1e-3
                actor_loss = actor_loss 
                # if ep % 5 and t % RENDER_INTERVAL == 0 and PRINT_LOSS:
                #     print("actor_loss", actor_loss)
                
                
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
                agent.actor.optimizer.step()
                
            for i, agent in enumerate(agents_list):
                for target_param, param in zip(agent.target_actor.parameters(), agent.actor.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - args.TAU) + param.data * args.TAU)
                for target_param, param in zip(server.target_critic_list[i].parameters(), server.critic_list[i].parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - args.TAU) + param.data * args.TAU)




if __name__ == '__main__':
    np.random.seed(1)
    args = args_parser()
    
    env = FL(args)
    
    env_GREEN = FL(args)
    env_GREEN.sync_dataloader(env)
    
    env_BLUE = FL(args)
    env_BLUE.sync_dataloader(env)
    
    RED = '#FF0000'
    GREEN = '#00FF00'
    BLUE = '#0000FF'
    env.set_render_color(RED)
    env_GREEN.set_render_color(GREEN)
    env_BLUE.set_render_color(BLUE)

    
    
    #env = Simple_test2() 
    agents_list = [Agent(env.observation_space, env.action_space, i, args, env.action_bound[1]) for i in range(1)]
    M = Memory(capacity=args.memory_capacity, state_dim=env.state_space, action_dim=env.action_space, num_agents=1)
    server = Server(env.state_space, env.action_space, 1)
    train(args)
    
    # while True:
    #     x = 1
    #
    
        
    