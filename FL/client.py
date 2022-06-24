import math
import random
import torch
from FL.models.initialize_model import initialize_model
import copy

class Client:

    def __init__(self, id, train_loader, test_loader, args, device, data_distribution):
        self.id = id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = initialize_model(args, device)
        self.receiver_buffer = {}
        self.epoch = 0
        self.args = args

        # 计算computation cost所用的参数
        self.frequency = random.randint(10, 50)
        self.cycles = random.randint(10, 50)
        self.iterations = random.randint(1, 10)
        # communication cost
        self.transmit = random.randint(50, 100)
        self.bandwidth = random.randint(50, 100)
        self.gain = random.randint(50, 100)
        self.pps = random.randint(50, 100)
        self.privacy_b = random.randint(10, 20)
        self.privacy_p = random.randint(10, 20)


        # self.weight = 0.5
        # self.weight_float = 0.5
        self.eid = -1
        self.device = device
        self.location = (0, 0)
        self.testing_acc = 0
        self.train_index = 0
        self.data_distribution = data_distribution
        # 这里可能有兼容错误，因为我们是列表
        self.data_num = len(self.train_loader)
        #self.loss=

    def local_update(self):
        num_iter = self.iterations
        loss = 0.0
        for i in range(num_iter):
            data = self.train_loader[self.train_index]
            inputs, labels = data
            loss += self.model.optimize_model(input_batch=inputs,
                                                label_batch=labels)
            self.train_index += 1
            self.train_index %= self.data_num
        # self.epoch += 1
        # self.model.exp_lr_sheduler(epoch=self.epoch)
        loss /= num_iter
        self.loss = loss
        return loss

    def test_model(self):
        correct = 0.0
        total = 0.0
        for data in self.test_loader:
            inputs, labels = data
            break
        size = labels.size(0)
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                outputs = self.model.test_model(input_batch= inputs)
                _, predict = torch.max(outputs, 1)
                total += size
                correct += (predict == labels).sum()
        self.testing_acc = correct.item() / total 
        return correct.item() / total 

    def send_to_edge(self, edge):
        edge.receiver_buffer[self.id] = copy.deepcopy(self.model.shared_layers.state_dict())
    
    def send_to_cloud(self, cloud):

        cloud.receiver_buffer[self.id] = copy.deepcopy(self.model.shared_layers.state_dict())
    
    def set_edge(self, eid):
        self.eid = eid
        
    def reset_model(self):
        self.receiver_buffer = {}
        self.epoch = 0
        # self.weight = random.random()
        # self.weight_float = self.weight
        self.model = initialize_model(self.args, self.device)
        self.train_index = 0
        # self.loss = 10
        # 因为默认数据集合不变，所以这里没有重新初始化data_num
        # 但是数据同步那里可能会有影响，所以在那里要更新

    # 定义每个客户端一次训练的计算时间（每个客户端迭代次数给定）
    def comp_time(self):
        comp_time = self.data_num * self.iterations * self.cycles / self.frequency
        return comp_time

    def comm_time(self):
        r_i = (self.bandwidth * math.log2(1 + (self.gain * self.transmit) / self.args.noise))
        comm_time = self.args.upload_dim / r_i
        return comm_time

    def train_time(self):
        return self.comm_time() + self.comp_time()

    # 计算send_to_edge的通信成本 z_{i,l_k}
    def comm_cost(self):
        comm_cost = self.comm_time() * self.pps
        return comm_cost

    def comp_cost(self):
        comp_cost = self.args.capacitance * self.data_num * self.cycles * (self.frequency ** 2)
        return comp_cost

    def pri_cost(self):
        pri_cost = self.privacy_b * self.privacy_p
        return pri_cost
