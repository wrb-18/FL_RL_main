import math

import numpy as np


def action_choice(actions, num_clients=16, num_edges=8, tao_max=2, selected_num=10):
    client_choice_ = actions[:num_clients]
    client_choice_ = [-client_choice_[i] for i in range(len(client_choice_))]
    ordered_client_choice = sorted(range(len(client_choice_)), key=lambda k: client_choice_[k])
    # 客户端选择：selected_clients元素为被选的客户端下标
    selected_clients = []
    for i in range(selected_num):
        selected_clients.append(ordered_client_choice[i])
    # 每次训练应该使用被选择的客户端进行训练
    taos_each_edge = []
    for i in range(num_edges):
        taos = actions[num_clients + i * tao_max: num_clients + (i+1) * tao_max]
        # a = np.where(taos == np.max(taos))
        a = taos.index(max(taos))
        taos_each_edge.append(a+1)

    return selected_clients, taos_each_edge


def comm_cost():
    comm_cost = 8 * 9 / \
                    (10 * math.log2(1 + (1 * 3) / 3))
    return comm_cost



# tao_max = 2, num_clients = 16, num_edges = 8
testAction = [0.2, 0.3, 0.1, 0.1, 0.2, 0.3, 0.5, 0.6,
              0.7, 0.8, 0.9, 0.1, 0.4, 0.9, 0.6, 0.4,
              0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2,
              0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2]

test1 = [1,2,4,5,3,5]
print(action_choice(testAction))

# print(comm_cost())
# print(max(test1))

def cluster_size(nums1, nums2):
    m = {}
    if len(nums1) < len(nums2):
        nums1, nums2 = nums2, nums1
    for i in nums1:
        if i not in m:
            m[i] = 1
        else:
            m[i] += 1
    result = []
    for i in nums2:
        if i in m and m[i]:
            m[i] -= 1
            result.append(i)
    return result

nums1 = [1,2,2,3]
nums2 = [2,3,4,5]
print(cluster_size(nums1, nums2))