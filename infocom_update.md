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
- 与最新的原始代码同步了一下。。。
- 修改对edge分配clients的方式由随机改为分配相同个数客户端，对每个edges选择的客户端设置相同的计算能力（FL.py中的edges_choices）
- (to do:)修改RL中的action、reward、state

    | 参数名 |模型中变量名|
    | ---|---|
    |num_iteration        |$\delta_i$|
    |num_edge_aggregation|$\tao$|
    |num_communication    |T|
    
    修改action为模型中的三个决策变量，即客户端选择、本轮客户端本地迭代次数、本轮cluster迭代次数；
        
    修改state为上轮完成后的资源剩余，上轮完成后的损失差；
        资源剩余的计算：cost包括训练cost1和隐私cost2，
    修改reward为一轮完成时间的相反数，并且在reward中添加约束作为punishment

- 所有文件都已更新与FL_RL_update同步，RL相关的设置和对比算法没丢，edges_choices客户端的集群方式使用fixed方法，固定之后不再变化。data_distribution方式没有变

# 0615

修改了一些RL中state 和action的设置，主要按照：

state：当前每个cluster的剩余可消耗cost $C_{r}^{k}$, 每个cluster的loss，以及当前的剩余的privacy cost $C_{r}^{p}$，dim=n+k+1

action： 每个client的选择情况（01变量）+每个cluster的tau(设置为0，1变量，n+1~n+1+$\tau_{max}$个输出分别表示整数1~$\tau_{max}$，选其中概率最大的表示为tau)；设$\tau_{max}$为最大值 ;

dim=n+k$\tau_{max}$

reward ：每一轮所用完成时间的数，Reward=$-T_{c}^{t}$; $C_{r}^{k}$为负表示超出资源限制，punishment1=$u_{1}C_{r}^{k}$, 同样punishment2=$u_{2}C_{r}^{p}$,

注意：每个client的weight在action的赋值已经注释掉了，所以后面所有的weight使用都需要修改。


# 0624
目前做的修改和问题：
- 客户端分层平均聚合，edge参与训练， 每次全局聚合后选择selected_num个客户端（目前设selected_num为选择总数，这里需要按每个cluster需要的个数分别设吗）
- 去掉了之前在FL中对num_edge和num_client的重新赋值，edge和client指的就是确定好cluster head之后的实际edge和client
- 隐私成本需要在edge训练之后也加上吗？文章中只算客户端，目前加了每个edge的隐私成本
- RL设置：action前num_client位为客户端选择（最大的selected_num个），后num_edge * tao_max位为集群迭代次数选择（每个edge选择概率最大的迭代次数）
- RL设置：state前num_client位为客户端当前轮损失，中间num_edge位为每个集群当前轮总资源剩余（计算+通信+隐私），最后一位为当前轮数
- RL设置：reward为当前轮完成时间，punishment1为超出的资源限制，punishment2为集群规模超出程度
下一步：
- 当前代码在edge.aggregate()那里有错，可能是因为加了选择步骤，导致edge.py那里的聚合方法出现了问题
- 添加预聚类的接口
