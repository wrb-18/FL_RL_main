
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from FL import edge
from options import args_parser
from FL.FL import Federate_learning as FL
from tqdm import tqdm
import random       
import json
from FL.datasets.get_data import *
from FL.client import Client

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1, PATH = './actor_model', LOAD = False):
        super(ActorNetwork, self).__init__()

        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_dim)
        self.max_action = max_action
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        #self.layer_norm1 = nn.LayerNorm(normalized_shape=state_dim // 2, eps=0, elementwise_affine=False)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=action_dim, eps=0, elementwise_affine=False)
        if LOAD:
            self.load_model(PATH)
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        x = self.layer_norm2(x)
        x = torch.tanh(x)
        x = ((x + 1) / 2)
        return x

    def choose_action(self, s):
        s = torch.FloatTensor(s)
        s = s.unsqueeze(0)
        s = self.forward(s)
        s = s.squeeze()
        s = s.detach().cpu()
        return s # single action

    
    def load_model(self, PATH):
        self.load_state_dict(torch.load(PATH))
    
    

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
    def __init__(self, state_dim, action_dim, id, args, max_action, LOAD = False):
        self.id = id
        self.target_actor = ActorNetwork(state_dim, action_dim, max_action=max_action)
        self.actor = ActorNetwork(state_dim, action_dim, max_action=max_action, LOAD = LOAD)
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        
        self.args = args
    #用于FL
    def observation(self, state):
        if type(state) == list:
            state = copy.copy(state)
        else:
            state = state.clone()
        return state

class Server:
    def __init__(self, state_dim, action_dim, num_of_agents):
        self.critic_list = [CriticNetwork(state_dim, action_dim) for i in range(num_of_agents)]
        self.target_critic_list = [CriticNetwork(state_dim, action_dim) for i in range(num_of_agents)]
        self.num_of_agents = num_of_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
                    
def compare(dataloaders, locations_list):
    dir = './compare/'
    list_path = os.listdir(dir)  # 根目录下的文件路径组成列表
            
    for file in list_path:

        plot_x = np.zeros(0)
        plot_reward = np.zeros(0)
        plot_acc = np.zeros(0)
        plot_cost = np.zeros(0)

        setting_file = os.path.join(dir, file)

        with open(setting_file) as fp:
            load_dict = json.load(fp)
        
        args = args_parser(load_dict)
        tmp = os.path.splitext(file)
        result_file = tmp[0] + ".npz"
        result_path = os.path.join('./result/', result_file)

        print(file)
        env = FL(args, dataloaders, locations_list)
        for i in tqdm(range(args.max_episodes * args.max_ep_step)):
            if i % args.max_ep_step == 0:
                env.reset()
            actions = [0.1] * (env.num_clients + env.num_edges * env.tao_max)
        
            _, reward, acc, cost = env.step(actions)
            plot_x = np.append(plot_x, i)
            plot_acc = np.append(plot_acc, acc)
            plot_reward = np.append(plot_reward, reward)
            plot_cost = np.append(plot_cost, cost)
            np.savez(result_path, plot_x, plot_acc, plot_cost, plot_reward)


# def all_in_one(args, data_distribution):
#     args_ = copy.deepcopy(args)
#     args_.num_clients = 1
#
#     data_distribution_ = np.array(data_distribution)
#     data_distribution_ = np.sum(data_distribution_, axis = 0, keepdims=True)
#     data_distribution_ = data_distribution_.tolist()
#     dataloaders = get_dataloaders(args_, data_distribution_)
#     train_loaders, test_loaders, v_train_loader, v_test_loader, data_distribution_ = dataloaders
#     plot_x = np.zeros(0)
#     plot_reward = np.zeros(0)
#     plot_acc = np.zeros(0)
#     plot_cost = np.zeros(0)
#
#     result_file = "all_in_one.npz"
#     result_path = os.path.join('./result/', result_file)
#     print(result_file)
#     # 所有数据训练完
#     #args_.num_iteration = len(train_loaders[0]) // 3
#     for i in tqdm(range(args_.max_episodes * args_.max_ep_step)):
#         if i % args_.max_ep_step == 0:
#             client = Client(id=0,
#                             train_loader=train_loaders[0],
#                             test_loader=v_test_loader,
#                             args=args_,
#                             device=args_.device,
#                             data_distribution = data_distribution_[0]
#             )
#         edge.local_update()
#         client.local_update()
#         acc = client.test_model()
#         reward, cost = acc, 0
#         plot_x = np.append(plot_x, i)
#         plot_acc = np.append(plot_acc, acc)
#         plot_reward = np.append(plot_reward, reward)
#         plot_cost = np.append(plot_cost, cost)
#         np.savez(result_path, plot_x, plot_acc, plot_cost, plot_reward)

    

if __name__ == '__main__':
    np.random.seed(1)
    
    args = args_parser()
    
    if args.data_distribution == 0.8:
        data_distribution =  [
            [2000, 0, 0, 0, 124, 0, 0, 62, 62, 248],  #0
            [2000, 0, 186, 0, 124, 62, 62, 62, 0, 0],  #1
            [2000, 0, 124, 186, 0, 62, 0, 62, 62, 0],  #2
            [2000, 0, 186, 0, 62, 124, 0, 0, 124, 0],  #3
            [2000, 0, 62, 0, 62, 62, 124, 0, 0, 186],  #4
            [2000, 0, 0, 186, 0, 0, 124, 0, 62, 124],  #5
            [0, 2000, 0, 62, 124, 0, 0, 310, 0, 0],  #6
            [0, 2000, 0, 124, 62, 62, 0, 0, 124, 124],  #7
            [0, 2000, 186, 62, 0, 62, 62, 62, 62, 0],  #8
            [0, 2000, 0, 124, 62, 62, 62, 62, 62, 62],  #9
            [0, 2000, 62, 124, 0, 0, 124, 62, 62, 62],  #10
            [0, 2000, 0, 62, 124, 62, 0, 124, 124, 0],  #11
            [0, 0, 2124, 0, 62, 124, 124, 0, 0, 62],  #12
            [0, 0, 62, 2062, 124, 62, 0, 62, 0, 124],  #13
            [0, 0, 62, 124, 2000, 0, 0, 0, 186, 124],  #14
            [0, 0, 0, 0, 124, 2000, 62, 186, 124, 0],  #15
            [0, 0, 62, 124, 62, 62, 2000, 62, 62, 62],  #16
            [0, 0, 62, 62, 0, 186, 0, 2062, 124, 0],  #17
            [0, 0, 0, 0, 0, 62, 62, 186, 2062, 124],  #18
            [0, 0, 248, 0, 62, 62, 0, 62, 62, 2000],  #19
            [2000, 0, 186, 0, 0, 0, 0, 62, 124, 124],  #20
            [0, 2000, 0, 186, 62, 0, 124, 62, 0, 62],  #21
            [0, 0, 2186, 0, 62, 62, 0, 62, 124, 0],  #22
            [0, 0, 0, 2062, 0, 62, 62, 62, 62, 186],  #23
                ]
    elif args.data_distribution == 0.5:
        data_distribution =  [
            [1250, 0, 468, 0, 156, 312, 0, 312, 0, 0],  #0
            [1250, 0, 312, 156, 0, 468, 156, 0, 156, 0],  #1
            [1250, 0, 0, 156, 0, 156, 156, 156, 624, 0],  #2
            [1250, 0, 156, 0, 0, 312, 156, 312, 156, 156],  #3
            [1250, 0, 468, 156, 0, 156, 156, 312, 0, 0],  #4
            [1250, 0, 0, 156, 0, 312, 156, 312, 156, 156],  #5
            [0, 1250, 156, 0, 468, 0, 156, 468, 0, 0],  #6f
            [0, 1250, 156, 156, 312, 156, 156, 312, 0, 0],  #7
            [0, 1250, 156, 156, 156, 0, 468, 0, 312, 0],  #8
            [0, 1250, 0, 312, 624, 156, 0, 0, 156, 0],  #9
            [0, 1250, 156, 468, 0, 0, 156, 0, 0, 468],  #10
            [0, 1250, 156, 0, 468, 156, 0, 156, 156, 156],  #11
            [0, 0, 1562, 0, 468, 156, 156, 0, 0, 156],  #12
            [0, 0, 156, 1874, 156, 156, 0, 0, 156, 0],  #13
            [0, 0, 312, 156, 1250, 312, 156, 156, 0, 156],  #14
            [0, 0, 156, 156, 312, 1250, 0, 0, 156, 468],  #15
            [0, 0, 0, 0, 0, 468, 1562, 156, 156, 156],  #16
            [0, 0, 156, 156, 312, 0, 312, 1250, 156, 156],  #17
            [0, 0, 0, 312, 0, 0, 468, 156, 1406, 156],  #18
            [0, 0, 0, 468, 0, 0, 0, 312, 468, 1250],  #19
            [1250, 0, 156, 0, 312, 312, 0, 468, 0, 0],  #20
            [0, 1250, 312, 156, 0, 468, 0, 156, 156, 0],  #21
            [0, 0, 1406, 156, 156, 156, 156, 312, 156, 0],  #22
            [0, 0, 156, 1406, 0, 312, 156, 0, 312, 156],  #23
            ]
    elif args.data_distribution == 0.2:
        data_distribution = [
            [500, 0, 250, 0, 0, 0, 250, 250, 750, 500],  #0
            [500, 0, 250, 750, 0, 0, 250, 250, 500, 0],  #1  
            [500, 0, 250, 250, 250, 500, 500, 0, 0, 250],  #2
            [500, 0, 250, 0, 750, 250, 500, 250, 0, 0],  #3  
            [500, 0, 250, 0, 250, 750, 500, 0, 0, 250],  #4  
            [500, 0, 500, 750, 250, 0, 250, 250, 0, 0],  #5  
            [0, 500, 500, 250, 0, 0, 500, 500, 0, 250],  #6  
            [0, 500, 0, 500, 500, 250, 250, 0, 500, 0],  #7  
            [0, 500, 250, 500, 250, 250, 500, 0, 0, 250],  #8
            [0, 500, 0, 0, 0, 500, 0, 250, 750, 500],  #9    
            [0, 500, 250, 500, 500, 0, 250, 0, 500, 0],  #10 
            [0, 500, 500, 0, 500, 250, 250, 0, 0, 500],  #11 
            [0, 0, 750, 0, 0, 0, 250, 500, 1000, 0],  #12    
            [0, 0, 0, 1000, 0, 500, 0, 250, 0, 750],  #13    
            [0, 0, 0, 250, 750, 500, 0, 250, 250, 500],  #14 
            [0, 0, 250, 500, 0, 750, 250, 0, 0, 750],  #15   
            [0, 0, 500, 250, 250, 0, 500, 250, 750, 0],  #16 
            [0, 0, 250, 750, 250, 500, 0, 750, 0, 0],  #17   
            [0, 0, 0, 250, 500, 500, 500, 0, 500, 250],  #18
            [0, 0, 750, 250, 250, 0, 250, 0, 250, 750],  #19
            [500, 0, 500, 0, 0, 500, 250, 0, 500, 250],  #20
            [0, 500, 250, 250, 0, 250, 0, 500, 750, 0],  #21
            [0, 0, 750, 0, 500, 0, 250, 250, 0, 750],  #22
            [0, 0, 250, 1000, 250, 0, 500, 500, 0, 0],  #23
                ]  
    elif args.data_distribution == 0:
        data_distribution = [
        [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  #0
        [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  #1
        [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  #2
        [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  #3
        [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  #4
        [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  #5
        [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  #6
        [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  #7
        [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  #8
        [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  #9
        [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  #10
        [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  #11
        [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  #12
        [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  #13
        [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  #14
        [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  #15
        [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  #16
        [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  #17
        [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  #18
        [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  #19
        [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  #20
        [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  #21
        [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  #22
        [250, 250, 250, 250, 250, 250, 250, 250, 250, 250]   #23
        ]

    data_distribution = np.array(data_distribution)
    data_distribution = data_distribution // 10
    data_distribution = data_distribution.tolist()

    dataloaders = get_dataloaders(args, data_distribution)
    cloud_location = [(0, 0)]
    edge_location = [(10, 17), (-10, 17) , (0, -20), (-52, 13), (-39, -19), (-72, 11)]
    client_location = []
    client_location += [(4, 15), (4, 19), (5, 13), (5, 21), (6, 12), (6, 22), (8, 11)]
    client_location += [(-16, 15), (-16, 19), (-15, 13), (-15, 21), (-14, 12), (-14, 22), (-12, 11)]
    client_location += [(-6, -22), (-6, -18), (-5, -24), (-5, -16), (-4, -25), (-4, -15), (-2, -26)]
    client_location += [(-69, 12), (-53, -19), (-48, 10)]
    
    
    locations_list = []
    locations_list.append(cloud_location)
    locations_list.append(edge_location)
    locations_list.append(client_location)

    ##########################################################################################################
    #                                         LOCAL_ML                                                       #
    ##########################################################################################################
    #all_in_one(args, data_distribution)    
    ##########################################################################################################
    #                                         FL                                                             #
    ##########################################################################################################
    compare(dataloaders, locations_list)

    
    ##########################################################################################################
    #                                         RL                                                             #
    ##########################################################################################################
    env = FL(args, dataloaders, locations_list)
    var = 0.3
    agents_list = [Agent(env.observation_space, env.action_space, i, args, env.action_bound[1], LOAD = args.load) for i in range(1)]
    M = Memory(capacity=args.memory_capacity, state_dim=env.state_space, action_dim=env.action_space, num_agents=1)
    server = Server(env.state_space, env.action_space, 1)
    
    plot_x = np.zeros(0)
    plot_reward = np.zeros(0)
    plot_acc = np.zeros(0)
    plot_cost = np.zeros(0)
    for t in range(10000):
        state = env.reset()
        print("RL", t)
        for ep in tqdm(range(args.max_ep_step)):
            actions = [] 
            for i, agent in enumerate(agents_list):
                a = agent.actor.choose_action(agent.observation(state))
                a = np.clip(np.random.normal(a, var), *env.action_bound)  # add randomness to action selection for exploration
                actions += a.tolist()
            state_, reward, acc, cost = env.step(actions)
            plot_x = np.append(plot_x, args.max_ep_step * t + ep)
            plot_acc = np.append(plot_acc, acc)
            plot_reward = np.append(plot_reward, reward)
            plot_cost = np.append(plot_cost, cost)
            np.savez('./result/RL.npz', plot_x, plot_acc, plot_cost, plot_reward)

            M.store_transition(state, actions, reward, state_)
            state = copy.deepcopy(state_)
            if not (args.load or M.pointer <= args.rl_batch_size or ep % 2):
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
                    
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
                    agent.actor.optimizer.step()
                    
                for i, agent in enumerate(agents_list):
                    for target_param, param in zip(agent.target_actor.parameters(), agent.actor.parameters()):
                        target_param.data.copy_(target_param.data * (1.0 - args.TAU) + param.data * args.TAU)
                    for target_param, param in zip(server.target_critic_list[i].parameters(), server.critic_list[i].parameters()):
                        target_param.data.copy_(target_param.data * (1.0 - args.TAU) + param.data * args.TAU)

                #保存
                torch.save(agents_list[0].target_actor.state_dict(), './actor_model')
