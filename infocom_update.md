# 0612
修改edge.py文件为自身参与训练：
    添加local_update方法；
    在aggregate方法中将自己的权重也加进去；
    添加send_to_self方法，将edge聚合后的cluster全局模型发送给自己；

修改FL.py文件，使edge自身参与训练：
    初始化client_num为选出edge后的剩余客户端数量；
    将为edges分配数据写入初始化过程；
    
    