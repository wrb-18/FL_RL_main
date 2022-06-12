# 0612
修改edge为自身参与训练：
    
    在edge.py文件中：
        添加local_update方法；
        在aggregate方法中将自己的权重也加进去；
        添加send_to_self方法，将edge聚合后的cluster全局模型发送给自己；
        在reset方法中添加edge参与后的reset规则；

    修改FL.py文件，使edge自身参与训练：
        初始化client_num为选出edge后的剩余客户端数量；
        将为edges分配数据写入初始化过程；
    
    现在的options.py里面的client个数是edge加上client的总个数，为方便测试修改了edge默认值为3，因为client总个数10没变，所以client个数即为7。
    