import random
from torch.autograd import Variable
import torch
from FL.models.initialize_model import initialize_model
import copy
import numpy as np

class Client():

    def __init__(self, id, train_loader, test_loader, args, device, max_size = 1000):
        self.id = id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = initialize_model(args, device)
        self.receiver_buffer = {}
        self.epoch = 0
        self.args = args
        self.weight = 0.5
        self.testing_acc = 0
        self.eid = -1
        self.device = device

    def local_update(self):
        num_iter = self.args.num_iteration
        loss = 0.0
        for i in range(num_iter):
            for data in self.train_loader:
                inputs, labels = data
                loss += self.model.optimize_model(input_batch=inputs,
                                                  label_batch=labels)
            self.epoch += 1
            self.model.exp_lr_sheduler(epoch=self.epoch)
        loss /= num_iter
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
                outputs = self.model.test_model(input_batch=inputs)
                _, predict = torch.max(outputs, 1)
                total += size
                correct += (predict == labels).sum()
        self.testing_acc = correct.item() / total 
        return correct.item() / total 

    def send_to_edge(self, edge):
        edge.receiver_buffer[self.id] = copy.deepcopy(self.model.shared_layers.state_dict())

    def get_edge(self):
        return self.eid
    
    def set_edge(self, eid):
        self.eid = eid
        
    def reset(self, shared_state_dict):
        self.receiver_buffer = {}
        self.epoch = 0
        self.weight = random.random()
        # self.model.update_model(copy.deepcopy(shared_state_dict))
        self.model = initialize_model(self.args, self.device)



