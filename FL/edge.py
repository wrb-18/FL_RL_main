import copy
import random

from FL.average import average_weights
from FL.models.initialize_model import initialize_model
import torch
class Edge():

    def __init__(self, id, train_loader, test_loader, args, data_distribution):
        self.device = args.device

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.id = id
        self.clients = []
        self.receiver_buffer = {}

        self.self_receiver_buffer = {}
        self.epoch = 0
        self.weight = 0.5
        self.weight_float = 0.5
        self.testing_acc = 0
        self.data_num = len(self.train_loader)

        self.all_weight_num = 0
        self.all_weight_float_num = 0
        self.all_data_num = 0
        self.args = args
        self.shared_state_dict = {}
        self.model = initialize_model(args, self.device)
        self.shared_state_dict = copy.deepcopy(self.model.shared_layers.state_dict())
        self.location = (0, 0)

        self.train_index = 0
        self.data_distribution = data_distribution

    # 添加本地更新方法，与client中的相同
    def local_update(self):
        num_iter = self.args.num_iteration
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
        return loss

    # 添加test_model方法
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
        
    def aggregate(self):
        received_dict = []
        sample_num = []
        self.all_weight_num = 0
        self.all_weight_float_num = 0
        self.all_data_num = 0
        if self.args.algorithm == 'W_avg':
            for client in self.clients:
                self.all_weight_float_num += client.weight_float
                if client.weight:
                    self.all_weight_num += client.weight
                    received_dict.append(self.receiver_buffer[client.id])
                    sample_num.append(client.weight)
            # 在聚合方法中将edge自己的权重加进去
            if self.weight:
                self.all_weight_num += self.weight
                received_dict.append(self.self_receiver_buffer)
                sample_num.append(self.weight)

            if self.all_weight_num == 0:
                return
            
            self.shared_state_dict = average_weights(w = received_dict,
                                                    s_num= sample_num)
        elif self.args.algorithm == 'FD_avg':
            for client in self.clients:
                # 为了保证train里面计算cost的时候兼容
                self.all_weight_float_num += client.weight_float
                self.all_weight_num += client.weight
                if client.data_num:
                    self.all_data_num += client.data_num
                    received_dict.append(self.receiver_buffer[client.id])
                    sample_num.append(client.data_num)

            # 在聚合方法中将edge自己的权重加进去
            self.all_weight_num += self.weight
            received_dict.append(self.self_receiver_buffer)
            sample_num.append(self.weight)

            if self.all_data_num == 0:
                return
            
            self.shared_state_dict = average_weights(w = received_dict,
                                                    s_num=sample_num)
        else:
            pass

    # 在edge发送cluster全局模型时发给自己
    def send_to_self(self, edge):
        self.self_receiver_buffer= copy.deepcopy(self.model.shared_layers.state_dict())
        self.model.update_model(self.self_receiver_buffer)

    def send_to_client(self, client):
        client.receiver_buffer = copy.deepcopy(self.shared_state_dict)
        client.model.update_model(client.receiver_buffer)

    def send_to_cloud(self, cloud):
        cloud.receiver_buffer[self.id] = copy.deepcopy(self.shared_state_dict)

    def remove_client(self, client):
        self.receiver_buffer.pop(client.id)
        self.clients.remove(client)
        client.set_edge(-2)
        # 上面聚合的时候清0了，所以不需要了
        # self.all_weight_num -= client.weight
        # self.all_weight_float_num -= client.weight_float
        # self.all_data_num -= client.data_num
        
    def add_client(self, client):
        self.clients.append(client)
        # 上面聚合的时候清0了，所以不需要了
        # self.all_weight_num += client.weight
        # self.all_weight_float_num += client.weight_float
        # self.all_data_num += client.data_num
        
    def reset_model(self):
        self.receiver_buffer = {}
        self.self_receiver_buffer = {}
        self.model = initialize_model(self.args, self.device)
        self.shared_state_dict = copy.deepcopy(self.model.shared_layers.state_dict())
        self.epoch = 0
        self.weight = random.random()
        self.weight_float = self.weight
        self.train_index = 0
        self.model = initialize_model(self.args, self.device)