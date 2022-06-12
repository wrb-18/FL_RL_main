# import copy
# from FL.average import average_weights

# class Edge():

#     def __init__(self, id, shared_layers):
#         self.id = id
#         self.clients = []
#         self.receiver_buffer = {}
#         self.all_trainsample_num = 0
        
        
#         self.shared_state_dict = shared_layers.state_dict()

#     def aggregate(self):
#         if len(self.clients) == 0:
#             return

#         received_dict = [dict for dict in self.receiver_buffer.values()]
#         sample_num = [client.total for client in self.clients]
#         self.shared_state_dict = average_weights(w = received_dict,
#                                                  s_num= sample_num)

#     def send_to_client(self, client):
#         client.receiver_buffer = copy.deepcopy(self.shared_state_dict)
#         client.model.update_model(client.receiver_buffer)
#         return None

#     def send_to_cloud(self, cloud):
#         cloud.receiver_buffer[self.id] = copy.deepcopy(self.shared_state_dict)
#         return None

#     def remove_client(self, client):
#         self.receiver_buffer.pop(client.id)
#         self.clients.remove(client)
                
#         client.set_edge(-1)
#         self.all_trainsample_num -= client.total

#     def add_client(self, client):
#         self.clients.append(client)
#         self.all_trainsample_num += client.total
        
#     def clear(self):
#         self.clients = []
#         self.receiver_buffer = {}
#         self.all_trainsample_num = 0
import copy
from FL.average import average_weights
from FL.models.initialize_model import initialize_model
import torch
class Edge():

    def __init__(self, id, train_loader, test_loader, args):
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = initialize_model(args, self.device)
        self.id = id
        self.clients = []
        self.receiver_buffer = {}
        self.self_receiver_buffer = {}
        self.all_weight_num = 0
        self.args = args
        self.shared_state_dict = {}
        self.weight = 0.5

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
        self.self_receiver_buffer = loss
        return loss

    def aggregate(self):
        received_dict = []
        sample_num = []
        self.all_weight_num = 0

        for client in self.clients:
            if client.weight:
                self.all_weight_num += client.weight
                received_dict.append(self.receiver_buffer[client.id])
                sample_num.append(client.weight)

        if self.weight:
            self.all_weight_num += self.weight
            received_dict.append(self.self_receiver_buffer)
            sample_num.append(self.weight)
                
        if self.all_weight_num == 0:
            return
        
        self.shared_state_dict = average_weights(w=received_dict,
                                                 s_num=sample_num)

    def send_to_self(self):
        self.self_receiver_buffer = copy.deepcopy(self.shared_state_dict)
        self.model.update_model(self.self_receiver_buffer)

    def send_to_client(self, client):
    
        client.receiver_buffer = copy.deepcopy(self.shared_state_dict)
        client.model.update_model(client.receiver_buffer)
        return None

    def send_to_cloud(self, cloud):
        cloud.receiver_buffer[self.id] = copy.deepcopy(self.shared_state_dict)
        return None

    def remove_client(self, client):
        self.receiver_buffer.pop(client.id)
        self.clients.remove(client)
        client.set_edge(-2)
        self.all_weight_num -= client.weight
        
    def add_client(self, client):
        self.clients.append(client)
        self.all_weight_num += client.weight
        
    def reset(self, shared_state_dict):
        self.receiver_buffer = {}
        self.clients = []
        self.all_weight_num = 0
        self.shared_state_dict = copy.deepcopy(shared_state_dict)