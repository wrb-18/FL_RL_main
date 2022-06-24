import random

import torch

class args_parser():
    def __init__(self, load_dict=None):
        if load_dict != None:
            self.algorithm = load_dict["algorithm"]
            self.is_layered = load_dict["is_layered"]
            # self.edge_choice = load_dict["edge_choice"]
            self.is_random_weight = load_dict["is_random_weight"]
        else:
            self.algorithm = "W_avg"
            self.is_layered = 1
            self.is_random_weight = 0

        self.edge_choice = "fixed"
        self.model = "lenet"
        self.batch_size = 10
        self.max_ep_step = 80
        self.num_iteration = 16
        self.num_edge_aggregation = 3
        self.num_communication = 1
        self.data_distribution = 0.5

        self.num_clients = 16
        self.num_edges = 8

        self.tao_max = 100
        self.selected_num = random.randint(self.num_clients - 6, self.num_clients - self.num_edges // 2)
        self.max_episodes = 1

        # cost
        self.noise = 100
        self.upload_dim = 100
        self.capacitance = 100


        self.lr = 0.01
        self.lr_decay = 0.995
        self.lr_decay_epoch = 1
        self.momentum = 0
        self.weight_decay = 0

        self.gamma = 0.9
        self.rl_batch_size = 32
        self.memory_capacity = 10000
        self.TAU = 0.01

        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        
        self.load = False

        self.total_resource = [10000] * self.num_edges
        
        


# class args_parser():
#     def __init__(self, load_dict=None):
#         if load_dict != None:
#             self.algorithm = load_dict["algorithm"]
#             self.is_layered = load_dict["is_layered"]
#             self.edge_choice = load_dict["edge_choice"]
#             self.is_random_weight = load_dict["is_random_weight"]
            
#         else:
#             self.algorithm = "W_avg"
#             self.is_layered = 1
#             self.edge_choice = "min"
#             self.is_random_weight = 0

#         self.model = "logistic"
#         self.batch_size = 16
#         self.max_ep_step  =  60
#         self.num_iteration = 20
#         self.num_edge_aggregation  =  2
#         self.num_communication = 1
#         self.data_distribution = 0.8
        
#         self.max_episodes = 1
        
#         self.num_clients = 24
#         self.num_edges = 6

#         self.lr = 0.01
#         self.lr_decay = 0.995
#         self.lr_decay_epoch = 1
#         self.momentum = 0
#         self.weight_decay = 0

#         self.gamma = 0.9
#         self.rl_batch_size = 32
#         self.memory_capacity = 10000
#         self.TAU = 0.01

#         self.cuda = torch.cuda.is_available()
#         self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        
#         self.load = True