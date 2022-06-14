import copy
from FL.average import average_weights
from FL.models.initialize_model import initialize_model
import torch

class Cloud():
    def __init__(self, args, edges, test_loader, shared_layers=0):
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.receiver_buffer = {}
        
        self.edges = edges
        self.args = args
        self.num_edges = args.num_edges
        self.test_loader = test_loader
        self.model = initialize_model(args, self.device)
        self.shared_state_dict = copy.deepcopy(self.model.shared_layers.state_dict())
        
        # update添加location初始化和测试精度初始化过程
        self.location = (0, 0)
        self.testing_acc = 0
        # # update添加layered比较
        # if not self.args.is_layered:
        #     self.clients = []
        
    def aggregate(self):
        received_dict = []
        sample_num = []
        edges_aggregate_num = 0
        
        for edge in self.edges:
            if edge.all_weight_num:
                edges_aggregate_num += 1
                received_dict.append(self.receiver_buffer[edge.id])
                sample_num.append(edge.all_weight_num)
                
        if edges_aggregate_num == 0:
            return
        self.shared_state_dict = average_weights(w = received_dict,
                                                 s_num= sample_num)
        self.model.update_model(copy.deepcopy(self.shared_state_dict))
        return None
    
    def send_to_client(self, client):
        client.receiver_buffer = copy.deepcopy(self.shared_state_dict)
        client.model.update_model(client.receiver_buffer)
        return None

    # 添加 将全局模型发送给edge作为下一轮的初始模型 的方法
    def send_global_to_edge(self, edge):
        edge.self_receiver_buffer = copy.deepcopy(self.shared_state_dict)
        edge.model.update_model(edge.self_receiver_buffer)
        return None

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
    
    def reset(self):
        self.receiver_buffer = {}
        self.model = initialize_model(self.args, self.device)
        self.shared_state_dict = copy.deepcopy(self.model.shared_layers.state_dict())

# 0613与FL_RL_update相比，未添加不分层的对比，以后再添加