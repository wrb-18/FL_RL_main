#from init import *



# from init import EDGES
# from FL.models.initialize_model import mnist_lenet
import math
import torch
from options import args_parser
import numpy as np
import torchvision
from FL.datasets.get_data import *
from FL.client import Client
from FL.edge import Edge
from FL.cloud import Cloud
import copy
import time
from tqdm import tqdm
from math import *
import matplotlib.pyplot as plt
import random

import torch.nn.functional as F



class Federate_learing():
    def __init__(self, args1):
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        args = copy.deepcopy(args1)

        self.train_loaders, self.test_loaders, self.v_train_loader, self.v_test_loader = get_dataloaders(args)

        # 初始化时修改了客户端个数为总客户端数减去选为edge的客户端数
        args.num_clients = args.num_clients - args.num_edges
        self.args = args

        self.clients = [Client(id=cid,
                            train_loader=self.train_loaders[cid],#train_loaders[cid],
                            test_loader=self.v_test_loader,
                            args=self.args,
                            # 把客户端个数改了
                            device=self.device) for cid in range(self.args.num_clients)]
        # 初始化edge时加入了为edge分配数据过程
        self.edges = [Edge(id = eid,
                           train_loader=self.train_loaders[eid+self.args.num_clients],  # train_loaders[cid],
                           test_loader=self.v_test_loader,
                           args=self.args) for eid in range(self.args.num_edges)]
        self.cloud = Cloud(args=self.args, edges=self.edges, test_loader=self.v_test_loader)

        self.num_clients = args.num_clients
        self.num_edges = args.num_edges
        self.step_count = 0

        self.reward = 0
        self.cost = 0
        self.cost_real = 0
        self.plot_x = np.zeros(0)
        self.plot_y = np.zeros(0)
        self.rendered = 0
        
        self.state = [-1] * (2 * self.num_clients + 1)
        self.state_space = len(self.state)
        self.observation_space = self.state_space
        self.action_bound = [0, 1]
        self.action_space = self.num_clients

        self.weight_threshold = 0.03

        self.sum_weight_float = 0
        # self.set_location(locations)
        
        edges_choices =  [min(i/(self.num_clients/self.num_edges), self.num_edges) for i in range(self.num_clients) ]

        for i, client in enumerate(self.clients):
            client.set_edge(edges_choices[i])
            self.edges[client.eid].add_client(client)
        

    def reset(self):
        self.sum_weight_float = 0
        self.cost = 0
        self.cost_real = 0
        # agents初始化
        # self.reward = 0
        torch.manual_seed(random.randint(0, 20))
        np.random.seed(random.randint(0, 20))
        
        
        # 清空模型参数
        self.cloud.reset()
        
        shared_state_dict = self.cloud.shared_state_dict
        
        for edge in self.edges:
            edge.reset(shared_state_dict)
        # update客户端也重置模型
        for client in self.clients:
            client.reset(shared_state_dict)
            
        edges_choices = [random.randint(0, self.num_edges - 1) for i in range(self.num_clients)]
        for client, edges_choice in zip(self.clients, edges_choices):
            client.reset(shared_state_dict)
        # for client, edge in zip(self.clients, self.edges):
        #     client.set_edge(edge.id)
            self.edges[client.eid].add_client(client)

        for i in range(self.num_clients):
            self.state[i] = self.clients[i].local_update()
        for i in range(self.num_clients):
            self.state[i + self.num_clients] = self.clients[i].weight
         
        return self.state
    
    def step(self, actions):
        # 问题
        for i in range(self.num_clients):
            #self.clients[i].weight += actions[i] 
            #self.clients[i].weight = max(0, self.clients[i].weight)
            self.clients[i].weight = actions[i] 
            
        
        client_loss, cloud_loss = self.train()
        
        for i in range(self.num_clients):
            self.state[i] = client_loss[i]
        for i in range(self.num_clients):
            self.state[i + self.num_clients] = self.clients[i].weight
        reward = [cloud_loss[0]]
        self.reward = reward
        return copy.copy(self.state), self.reward
        
    def set_render_color(self, color):
        self.render_color = color

    # 根据需要自己编写，现在是横坐标ep  纵坐标global_loss
    def render(self):
        if self.render_color == "#FF0000" and self.step_count % 5 == 0:
            print("weights:\n", self.state[self.num_clients:])
        self.step_count += 1
        self.plot_x = np.append(self.plot_x, self.step_count)
        test_val = self.reward[0]
        self.plot_y = np.append(self.plot_y, test_val)
        if self.step_count >= 20:
            self.plot_x = self.plot_x[1:]
            self.plot_y = self.plot_y[1:]
        if not self.rendered:
            plt.ion()
            plt.figure(1)
            self.rendered = 1
        else:
            plt.plot(self.plot_x, self.plot_y, '-r', color=self.render_color)
            plt.draw()
            plt.pause(0.1)

    def train(self):
        EDGES = self.num_edges
        clients = self.clients
        edges = self.edges
        cloud = self.cloud
        args = self.args
        
        client_loss = [0] * self.num_clients
        cloud_loss = [0]
        
        for num_comm in range(args.num_communication):
            for num_edgeagg in range(args.num_edge_aggregation):
                for eid in range(EDGES):
                    for client in edges[eid].clients:
                        
                        loss = client.local_update()
                        client_loss[client.id] = loss
                        client.send_to_edge(edges[eid])

                    edges[eid].local_update()

                    edges[eid].aggregate()
                    
                    for client in edges[eid].clients:
                        edges[eid].send_to_client(client)
                    edges[eid].send_to_self()



            for eid in range(EDGES):
                edges[eid].send_to_cloud(cloud)
                
            cloud.aggregate()
            
            cloud_loss[0] = cloud.test_model()
            
            for client in clients:
                cloud.send_to_client(client)
            #每次全局迭代cloud要发送全局模型到参与训练的客户端和edge
            for edge in edges:
                cloud.send_global_to_edge(edge)
                
        return client_loss, cloud_loss
    
    def sync_dataloader(self, env):
        self.train_loaders = env.train_loaders
        self.test_loaders = env.test_loaders
        self.v_train_loader = env.v_train_loader
        self.v_test_loader = env.v_test_loader