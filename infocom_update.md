# 0612
## 修改edge为自身参与训练：
    
    在edge.py文件中：
        添加local_update方法；
        在aggregate方法中将自己的权重也加进去；
        添加send_to_self方法，将edge聚合后的cluster全局模型发送给自己；
        在reset方法中添加edge参与后的reset规则；

    修改FL.py文件，使edge自身参与训练：
        初始化client_num为选出edge后的剩余客户端数量；
        将为edges分配数据写入初始化过程；
    修改cloud.py文件，在每次全局聚合之后将全局模型也发回给edges
    
    现在的options.py里面的client个数是edge加上client的总个数，为方便测试修改了edge默认值为3，因为client总个数10没变，所以client个数即为7。
# 0614
## 与最新的原始代码同步了一下。。。
## 修改对edge分配clients的方式由随机改为分配相同个数客户端，对每个edges选择的客户端设置相同的计算能力（FL.py中的edges_choices）
## (to do:)修改RL中的action、reward、state
    
    |参数名|模型中变量名|
    |num_iteration|tao1|
    |num_edge_aggregation|tao2|
    |num_communication|T|

    修改action为模型中的三个决策变量，即客户端选择、本轮客户端本地迭代次数、本轮cluster迭代次数；
        
    修改state为上轮完成后的资源剩余，上轮完成后的损失差；
        资源剩余的计算：cost包括训练cost1和隐私cost2，
    修改reward为一轮完成时间的相反数，并且在reward中添加约束作为punishment
    
## 所有文件都已更新与FL_RL_update同步，RL相关的设置和对比算法没丢，edges_choices客户端的集群方式使用fixed方法，固定之后不再变化。data_distribution方式没有变