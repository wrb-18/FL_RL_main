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
    def __init__(self, args1, dataloaders, locations):
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        args = copy.deepcopy(args1)

        self.train_loaders, self.test_loaders, self.v_train_loader, self.v_test_loader, self.data_distribution = dataloaders

        # 初始化时修改了客户端个数为总客户端数减去选为edge的客户端数
        args.num_clients = args.num_clients - args.num_edges
        self.args = args

        self.clients = [Client(id=cid,
                            train_loader=self.train_loaders[cid],#train_loaders[cid],
                            test_loader=self.v_test_loader,
                            args=self.args,
                            # 把客户端个数改了
                            device=self.device,
                            data_distribution = self.data_distribution[cid]) for cid in range(self.args.num_clients)]
        # 初始化edge时加入了为edge分配数据过程
        self.edges = [Edge(id = eid,
                           train_loader=self.train_loaders[eid+self.args.num_clients],  # train_loaders[cid],
                           test_loader=self.v_test_loader,
                           args=self.args,
                           data_distribution = self.data_distribution[eid]) for eid in range(self.args.num_edges)]
        self.cloud = Cloud(args=self.args, edges=self.edges, test_loader=self.v_test_loader)

        self.num_clients = args.num_clients
        self.num_edges = args.num_edges
        # self.step_count = 0

        self.reward = 0
        self.cost = 0
        self.cost_real = 0
        self.state = [-1] * (2 * self.num_clients + 1)
        self.state_space = len(self.state)
        self.observation_space = self.state_space
        self.action_bound = [0, 1]
        self.action_space = self.num_clients
        self.weight_threshold = 0.03
        self.sum_weight_float = 0
        self.set_location(locations)
        

    def reset(self):
        self.sum_weight_float = 0
        self.cost = 0
        self.cost_real = 0
        # agents初始化
        # self.reward = 0
        torch.manual_seed(random.randint(0, 20))
        np.random.seed(random.randint(0, 20))
        
        
        # 清空模型参数
        self.cloud.reset_model()
        
        # shared_state_dict = self.cloud.shared_state_dict

        for edge in self.edges:
            edge.reset_model()
        # update客户端也重置模型
        for client in self.clients:
            client.reset_model()

        # 置cloud的testing_acc=0
        self.cloud.testing_acc = 0

        # 如果只有一层，那么就全部连接到cloud
        if self.args.is_layered == 0:
            for client in self.clients:
                self.cloud.clients.append(client)
        # 如果分层了
        else:
            # 固定的下层连接方式
            if self.args.edge_choice == "fixed":
                edges_choices = [min(i // (self.num_clients // self.num_edges), self.num_edges) for i in
                             range(self.num_clients)]
            for client, edges_choice in zip(self.clients, edges_choices):  # 根据choices选择edges
                    client.set_edge(edges_choice)
                    self.edges[client.eid].add_client(client)

        if self.args.is_random_weight:
            weights = [random.random() for i in range(self.num_clients)]
            self.sum_weight_float = sum(weights)
            for i in range(self.num_clients):
                self.clients[i].weight_float = weights[i]
                if weights[i] / self.sum_weight_float < self.weight_threshold:
                    self.clients[i].weight = 0
                else:
                    self.clients[i].weight = weights[i]
        # 更新state
        for i in range(self.num_clients):
            self.state[i] = self.clients[i].local_update()
        for i in range(self.num_clients, 2 * self.num_clients):
            self.state[i] = self.clients[i - self.num_clients].weight_float
        self.state[-1] = 0
        return self.state

    def step(self, actions):
        if self.args.edge_choice == "random":
            edges_choices = [random.randint(0, self.num_edges - 1) for i in range(self.num_clients)]
            # for i, edges_choice in enumerate(edges_choices):
                # if random.random() < 0.5:
                #     edges_choices[i] = -1
            # 根据edges_choices连接edges
            # for edge in self.edges:  # 断开edges与所有clients的连接
            #     edge.clients = []
            for client, edges_choice in zip(self.clients, edges_choices):  # 根据choices选择edges
                client.set_edge(edges_choice)
                self.edges[client.eid].add_client(client)

        if self.args.is_random_weight == 0:
            self.sum_weight_float = sum(actions)
            for i in range(self.num_clients):
                self.clients[i].weight_float = actions[i]
                if actions[i] / self.sum_weight_float < self.weight_threshold:
                    self.clients[i].weight = 0
                else:
                    self.clients[i].weight = actions[i]

        client_loss, cloud_loss, cost_communication, cost_communication_real = self.train()

        for i in range(self.num_clients):
            self.state[i] = client_loss[i]
        for i in range(self.num_clients, 2 * self.num_clients):
            self.state[i] = self.clients[i - self.num_clients].weight_float
        self.state[-1] += 1
        self.cost_real += cost_communication_real
        self.cost += cost_communication
        # state[-1] 为 epoch
        self.reward = [cloud_loss[0] * 5 - self.cost / self.state[-1] / 20]
        # print("cost_communication", cost_communication)
        # self.reward = [cloud_loss[0]]
        return copy.copy(self.state), self.reward, cloud_loss[0], self.cost_real

    def cost_comm(self, client_or_edge, edge_or_cloud):
        g0 = 1.42 * 0.0001
        G0 = 2.2846
        sigma = 1
        p = 100
        B = 1000000
        R_square = (client_or_edge.location[0] - edge_or_cloud.location[0]) ** 2 + (
                    client_or_edge.location[1] - edge_or_cloud.location[1]) ** 2
        delta = g0 * G0 / sigma ** 2
        r = B * math.log2(1 + delta * p / R_square)
        M = 100
        return M / r

    def train(self):
        EDGES = self.num_edges
        clients = self.clients
        edges = self.edges
        cloud = self.cloud
        args = self.args
        client_loss = [0] * self.num_clients
        cloud_loss = [0]
        total_cost_comm = 0
        total_cost_comm_real = 0

        # print("+++++++++++++++++++++++++++++")
        # for i in range(self.args.num_clients):
        #     print(self.clients[i].weight_float, end = ' ')
        # print("+++++++++++++++++++++++++++++")
        for num_comm in range(args.num_communication):
            if self.args.is_layered:
                for num_edgeagg in range(args.num_edge_aggregation):
                    for eid in range(EDGES):
                        for client in edges[eid].clients:
                            client_loss[client.id] = client.local_update()
                            client.send_to_edge(edges[eid])
                            if self.args.algorithm == "W_avg":
                                total_cost_comm += client.weight_float / self.sum_weight_float * self.cost_comm(client,
                                                                                                                edges[
                                                                                                                    eid])
                                total_cost_comm_real += (client.weight != 0) * self.cost_comm(client, edges[eid])
                            else:
                                total_cost_comm += self.cost_comm(edges[eid], client)
                                total_cost_comm_real = total_cost_comm

                        edges[eid].aggregate()

                        for client in edges[eid].clients:
                            edges[eid].send_to_client(client)

                for eid in range(EDGES):
                    edges[eid].send_to_cloud(cloud)
                    if self.args.algorithm == "W_avg":
                        total_cost_comm += edges[eid].all_weight_float_num / self.sum_weight_float * self.cost_comm(
                            edges[eid], cloud)
                        total_cost_comm_real += (edges[eid].all_weight_num != 0) * self.cost_comm(edges[eid], cloud)
                    else:
                        total_cost_comm += self.cost_comm(edges[eid], cloud)
                        total_cost_comm_real = total_cost_comm

                cloud.aggregate()

                for client in clients:
                    cloud.send_to_client(client)
            else:
                for num_edgeagg in range(args.num_edge_aggregation):
                    for client in self.clients:
                        client_loss[client.id] = client.local_update()

                    for client in self.clients:
                        client.send_to_cloud(cloud)
                    if self.args.algorithm == "W_avg":
                        total_cost_comm += client.weight_float / self.sum_weight_float * self.cost_comm(client, cloud)
                        total_cost_comm_real += (client.weight != 0) * self.cost_comm(client, cloud)
                    else:
                        total_cost_comm += self.cost_comm(client, cloud)
                        total_cost_comm_real = total_cost_comm

                    cloud.aggregate()

                    for client in clients:
                        cloud.send_to_client(client)

            cloud_loss[0] = cloud.test_model()

        return client_loss, cloud_loss, total_cost_comm, total_cost_comm_real

    def set_location(self, location_list):
        cloud_location = location_list[0][0]
        edge_location_list = location_list[1]
        client_location_list = location_list[2]

        self.cloud.location = cloud_location
        for edge, edge_location in zip(self.edges, edge_location_list):
            edge.location = edge_location
        for client, client_location in zip(self.clients, client_location_list):
            client.location = client_location

    # def D_kl(self, Pa, Pb):
    #     Pa = torch.Tensor(Pa)
    #     Pa = Pa / Pa.sum()
    #     Pb = torch.Tensor(Pb)
    #     Pb = Pb / Pb.sum()
    #     kl_sum = F.kl_div(Pa.softmax(dim=-1).log(), Pb.softmax(dim=-1), reduction='sum')
    #     return kl_sum.item()

    def train(self):
        EDGES = self.num_edges
        clients = self.clients
        edges = self.edges
        cloud = self.cloud
        args = self.args
        client_loss = [0] * self.num_clients
        cloud_loss = [0]
        total_cost_comm = 0
        total_cost_comm_real = 0

        for num_comm in range(args.num_communication):
            if self.args.is_layered:
                for num_edgeagg in range(args.num_edge_aggregation):
                    for eid in range(EDGES):
                        edges[eid].send_to_cloud(cloud)
                        if self.args.algorithm == "W_avg":
                            total_cost_comm += edges[eid].all_weight_float_num / self.sum_weight_float * self.cost_comm(
                                edges[eid], cloud)
                            total_cost_comm_real += (edges[eid].all_weight_num != 0) * self.cost_comm(edges[eid], cloud)
                        else:
                            total_cost_comm += self.cost_comm(edges[eid], cloud)
                            total_cost_comm_real = total_cost_comm

                    cloud.aggregate()

                    for client in clients:
                        cloud.send_to_client(client)
            else:
                for num_edgeagg in range(args.num_edge_aggregation):
                    for client in self.clients:
                        client_loss[client.id] = client.local_update()

                    for client in self.clients:
                        client.send_to_cloud(cloud)
                    if self.args.algorithm == "W_avg":
                        total_cost_comm += client.weight_float / self.sum_weight_float * self.cost_comm(client,
                                                                                                        cloud)
                        total_cost_comm_real += (client.weight != 0) * self.cost_comm(client, cloud)
                    else:
                        total_cost_comm += self.cost_comm(client, cloud)
                        total_cost_comm_real = total_cost_comm

                    cloud.aggregate()

                    for client in clients:
                        cloud.send_to_client(client)

            cloud_loss[0] = cloud.test_model()

        return client_loss, cloud_loss, total_cost_comm, total_cost_comm_real

    def set_location(self, location_list):
        cloud_location = location_list[0][0]
        edge_location_list = location_list[1]
        client_location_list = location_list[2]

        self.cloud.location = cloud_location
        for edge, edge_location in zip(self.edges, edge_location_list):
            edge.location = edge_location
        for client, client_location in zip(self.clients, client_location_list):
            client.location = client_location


# def sync_dataloader(self, env):
#         self.train_loaders = env.train_loaders
#         self.test_loaders = env.test_loaders
#         self.v_train_loader = env.v_train_loader
#         self.v_test_loader = env.v_test_loader