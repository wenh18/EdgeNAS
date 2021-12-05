import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import copy
from torch.nn.parameter import Parameter
from change_layers import *
from env import *
# TODO：目前的问题：探索不够，没有得到正的奖励，解决方法：增大EPOCH，增大随机探索轮数
# 不知道目前在神经网络的哪个位置。解决方法：位置编码或者使用RNN，发现RNN没有起到效果，估计需要使用位置编码了，可不可以理解为LSTM仅仅让agent知道前几个层的决策，并没有知道位置呢？
# 知道的前几层的决策信息不够多。解决方法：输入3个决策(33维)或者使用RNN
# 把agent也放到GPU上吧
# 目前只有1/10的概率跳层，这样有点小，应该增加几率？ preoritized replay 两个trick dualing network
# 包装成gym的environment
# python缓存操作，让搬运更高效
# 问题：depth不能随意跳呀，只能跳后面的，对此有三种解决办法，一种是一旦智能体决定跳过某一层，则该组的所有层全部跳过。一种是增加智能体的动作，让它在每一组的开始决定该组的深度，一种是让它正常学习，学完后一旦有跳中间层的行为就把该组后面的也全部删掉，最后再测评准确率
# 当然，这里也同样是一个可以优化的点，因为关于是否跳层，我们只有3^5 = 243种选择，是否可以打一个表，根据输入的时间先大致估计出如何跳层，再在里面选ks和channel
cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")

Gamma = 0.95  # reward discount
LR = 0.0005
# Epsilon = 0.9  # 应该是递减的！
Memory_capacity = 50000  # memory_capacity≈EPISODE*STEP*0.6，此时需要占用约30M内存，也许不够吧？我们能不能优先替换掉一些reward非常小的记忆呢？
Batch_size = 32
Target_replace_iter = 100  # target update frequency
EPISODE = 4000
STEP = 20

# alpha = 0.5  # 环境中时延对于reward的占比

def get_modulelist(model):
    net_structure = list(model.children())
    new_module = nn.ModuleList()
    for i in range(len(net_structure)):
        if isinstance(net_structure[i], nn.ModuleList):
            for j in net_structure[i]:
                new_module.append(j)
            continue
        new_module.append(net_structure[i])
    return new_module

def warm_up_hardware(model_list, x, times):
    shape = x.shape
    for _ in range(times):
        t = torch.rand(shape).to(device)
        for j in range(len(new_module) - 1):
            t = model_list[j](t)
        t = model_list[-1](t.view(-1, t.size()[1]))
    t_start = time.time()
    t = torch.rand(shape).to(device)
    for j in range(len(new_module) - 1):
        t = model_list[j](t)
    t = model_list[-1](t.view(-1, t.size()[1]))
    if cuda_available:
        torch.cuda.synchronize()
    full_model_time = time.time() - t_start
    print("full model time:", full_model_time)
    return full_model_time

def step2state(step, STEP):
    step_state = torch.zeros(STEP)
    step_state[step] = 1
    return step_state

class Net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=31):  # state_dim:9, action_dim:8
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(128, 128)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(128, action_dim)
        self.fc3.weight.data.normal_(0, 0.1)
        self.embedding = nn.Linear(state_dim - 1, state_dim - 1)# nn.Embedding(state_dim - 1, state_dim - 1)  # 注意embedding是layer normalization，可以之后试试batch normalization
        self.embedding.weight.data.normal_(0, 0.1)
        self.hidden = None
        self.rnn = nn.LSTM(state_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.batch_size = 1
        # self.relu = nn.ReLU()

    # def scalar2vector(action):
    def init_hidden(self):
        return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
        Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))
        

    def forward(self, x):
        self.hidden = self.init_hidden()
        # print(self.hidden.shape)
        batch_size = x.size(0)
        # print("::::::::::", batch_size)
        # print(x[:, 1:].shape)
        x0 = self.embedding(x[:, :-1])  # 上一个state和上一个action
        # print(x0.shape)
        # x0 = self.relu(x0)
        x1 = torch.unsqueeze(x[:, -1], dim=1)
        x = torch.cat((x1, x0), dim=1)
        out, self.hidden = self.rnn(x.view(1, batch_size, -1), self.hidden)
        # print(out.shape)
        # print(":::::::::", out.shape)
        out = torch.squeeze(out, 0)
        # print(";;;;;;;;;", out.shape)
        out = F.relu(self.fc1(out))
        # print(out)
        out = F.relu(self.fc2(out))
        # print(out)
        out = self.fc3(out)
        # print(out)
        return out

        # # print(x[:, 1:].shape)
        # x0 = self.embedding(x[:, 1:])  # 上一个action
        # # print(x0.shape)
        # # x0 = self.relu(x0)
        # x1 = torch.unsqueeze(x[:, 0], dim=1)
        # x = torch.cat((x1, x0), dim=1)
        # print(x.shape)
        # out = F.relu(self.fc1(x))
        # out = F.relu(self.fc2(out))
        # out = self.fc3(out)
        # return out

class DQN(object):
    def __init__(self, state_dim, action_dim):  # , scalar_state_dim=2
        self.state_dim = state_dim
        # self.scalar_state_dim = scalar_state_dim
        self.action_dim = action_dim
        self.eval_net, self.target_net = Net(state_dim, action_dim), Net(state_dim, action_dim)
        self.learn_step_counter = 0  #for target updating
        self.memory_counter = 0  #for storing memory
        self.memory = np.zeros((Memory_capacity, state_dim * 2 + 3)) #innitialize memory
        self.optimizer = optim.Adam(self.eval_net.parameters(),lr=LR)
        self.loss_func = nn.MSELoss()
        self.Epsilon = 1.0

    def choose_action(self, x, test=False):
        if test:
            self.Epsilon = 0.0
        self.eval_net.batch_size = 1
        self.target_net.batch_size = 1
        x = Variable(torch.unsqueeze(x,0))
        # print("********", self.Epsilon)
        if np.random.uniform() > self.Epsilon:
            action_value = self.eval_net.forward(x)
            # print(action_value)
            action = torch.max(action_value, 1)[1]
        else:
            # print("yes")
            action = torch.tensor([np.random.randint(0, self.action_dim)])
        return action


    def store_transition(self,s,a,r,s_, end):
        transition = np.hstack((s,[a,r],s_, end))
        # print(transition.shape)
        index = self.memory_counter % Memory_capacity
        self.memory[index,:] = transition
        self.memory_counter += 1

    def learn(self):
        #target net update
        if self.learn_step_counter % Target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(Memory_capacity,Batch_size)
        b_memory = self.memory[sample_index,:]
        b_s = Variable(torch.FloatTensor(b_memory[:,:self.state_dim]))
        b_a = Variable(torch.LongTensor(b_memory[:,self.state_dim:self.state_dim+1]))
        b_r = Variable(torch.FloatTensor(b_memory[:,self.state_dim+1:self.state_dim+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:,self.state_dim+2:-1]))
        b_end = Variable(torch.FloatTensor(b_memory[:,-1]))  # 1*32
        
        self.eval_net.batch_size = Batch_size
        self.target_net.batch_size = Batch_size
        # print(self.eval_net(b_s).shape)
        q_eval = self.eval_net(b_s).gather(1,b_a)
        q_next = self.target_net(b_s_).detach()
        # print(b_a.shape, q_next.shape, q_eval.shape)
        # q_target = b_r +Gamma * q_next.max(1)[0].view(Batch_size, 1)
        q_target = Variable(torch.FloatTensor(self.eval_net.batch_size, 1))
        # show = True
        for b in range(self.eval_net.batch_size):
            if b_end[b] == 0:
                # if show:
                #     # show = False
                #     print(q_target[b], b_r[b], q_next[b])
                q_target[b] = b_r[b] + Gamma * q_next[b][b_a[b]]
            else:
                q_target[b] = b_r[b]
        loss = self.loss_func(q_eval,q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def legal_arch(step, skip, ks_list):
    if step % 4 == 0 and skip == 1:
        return False
    # if step % 4 > 0 and skip == 0:  # 在非通道变换层，如果前面的层选择了跳过，那么这一层也必须跳过
    #     if ks_list[-1] == 0:
    #         return False
    return True

def train_agent(new_module, agent, time_budget, env):
    # 在训练过程中不需要有mask，在测试时需要在跑到某些阶段（例如通道数变换时mask掉一些skip之类的选项）
    print("object time:", time_budget)
    # new_module = new_module.to(device)
    x = torch.rand(4, 3, 224, 224).to(device)
    warm_up_hardware(new_module, x, 10)
    ep_rs = []
    ACCs = []
    LATENs = []
    start_training = False
    for episode in range(EPISODE):
        print("----------------", episode, "----------------")
        x = torch.rand(4, 3, 224, 224).to(device)  # 待改进：将imagenet数据加载进来
        t = time.time()
        for i in range(2):
            x = new_module[i](x)
        if cuda_available:
            torch.cuda.synchronize()
        time_used = (time.time() - t) / time_budget
        #  get initial state
        # state = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, time_used])  # time_used, skip(1->skip, 0->keep), kernel_size[3,5,7], channel[3,4,6] 
        state = torch.zeros(31)
        state[1] = 1
        state[-1] = time_used

        input_channels = [24, 32, 48, 96, 136, 192]
        channel_idx = 0
        ks_list = []  # 用于网络精度预测
        ex_list = []
        ks_lists = []
        ex_lists = []
        
        ep_r = 0
        trans_time = []
        for step in range(STEP):
            state[11:31] = step2state(step, STEP)

            if (step - 1) % 4 == 0:
                channel_idx += 1
            action = agent.choose_action(state).numpy()
            skip, ks, channels_extension = scalar2block(action[0].item())
            # print(step+2, input_channels[channel_idx])
            if not legal_arch(step, skip, ks_list):
                reward = -100
                agent.store_transition(state, action, reward, state, 1)  # 1->end
                ep_r += reward
                print("not legal")
                break
            if skip == 0:
                tmp_module = copy.deepcopy(new_module[step + 2]).to(device)
                t1 = time.time()
                change_kernel(model_layer=tmp_module, sub_kernel_size=ks, expanded_channel=input_channels[channel_idx] * channels_extension, 
                              kernel_id=step, kernel_mapping_weights=kernel_mapping_weights)
                trans_time.append(time.time() - t1)
                t1 = time.time()
                x = tmp_module(x)
                if cuda_available:
                    torch.cuda.synchronize()
                this_layer_time = (time.time() - t1) / time_budget
                time_used = this_layer_time + time_used
                
                del tmp_module
                ks_list.append(ks)
                ex_list.append(channels_extension)
            else:
                this_layer_time = 0.0
                ks_list.append(0)
                ex_list.append(0)
                # print("skip")

            reward = env.get_reward(step, this_layer_time, time_used, ks_list, ex_list)
            ep_r += reward
            new_state = [0 for _ in range(30)] + [time_used]
            # print("--------------", action[0].item())
            new_state[action[0].item()] = 1
            # new_state = torch.tensor([action[0].item(), time_used])
            new_state = torch.tensor(new_state)
            if step < 19:
                agent.store_transition(state, action, reward, new_state, 0)
            else:
                agent.store_transition(state, action, reward, new_state, 1)
            
            if agent.memory_counter > Memory_capacity:
                agent.learn()
                if not start_training:
                    print("training is starting......")
                    start_training = True
            
            state = new_state
            if step == 19:
                print(reward)
            # print(ep_r)
        ep_rs.append(ep_r)
        if len(ks_list) == 20:
            ACCs.append(env.get_accuracy(ks_list, ex_list))
            LATENs.append(time_used)
        else:
            ACCs.append(0)
            LATENs.append(10)
        if episode > int(EPISODE * 0.6):
            agent.Epsilon = agent.Epsilon * 0.9
        # print(ep_r)
        if episode % 20 == 0:
            print(ks_list)
            print(ex_list)
            ks_lists.append(ks_list)
            ex_lists.append(ex_list)
    print(ep_rs)
    print(ACCs)
    print(LATENs)
    # print(ks_lists)
    # print(ex_lists)

def get_new_model(agent, model, time_budget, env):
    # model = model.to(device)
    x = torch.rand(4, 3, 224, 224).to(device)  # 待改进：将imagenet数据加载进来
    warm_up_hardware(model, x, 10)
    t = time.time()
    for i in range(2):
        x = new_module[i](x)
    if cuda_available:
        torch.cuda.synchronize()
    time_used = (time.time() - t) / time_budget
    #  get initial state
    # state = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, time_used])  # time_used, skip(1->skip, 0->keep), kernel_size[3,5,7], channel[3,4,6] 
    state = torch.zeros(31)
    state[1] = 1
    state[-1] = time_used
    input_channels = [24, 32, 48, 96, 136, 192]
    channel_idx = 0
    ks_list = []  # 用于网络精度预测
    ex_list = []
    for step in range(STEP):

        state[11:31] = step2state(step, STEP)

        if (step - 1) % 4 == 0:
            channel_idx += 1
        action = agent.choose_action(state).numpy()
        skip, ks, channels_extension = scalar2block(action[0].item())
        if skip == 0:
            change_kernel(model_layer=model[step + 2], sub_kernel_size=ks, expanded_channel=input_channels[channel_idx] * channels_extension, 
                        kernel_id=step, kernel_mapping_weights=kernel_mapping_weights)
            t1 = time.time()
            x = model[step + 2](x)
            if cuda_available:
                torch.cuda.synchronize()
            time_used = (time.time() - t1) / time_budget + time_used
            ks_list.append(ks)
            ex_list.append(channels_extension)
        else:
            if step % 4 == 0:
                print("warning, got an illegal arch")
                change_kernel(model_layer=model[step + 2], sub_kernel_size=3, expanded_channel=input_channels[channel_idx] * 3, 
                        kernel_id=step, kernel_mapping_weights=kernel_mapping_weights)
                t1 = time.time()
                x = model[step + 2](x)
                if cuda_available:
                    torch.cuda.synchronize()
                time_used = (time.time() - t1) / time_budget + time_used
                ks_list.append(ks)
                ex_list.append(channels_extension)
            else:
                ks_list.append(0)
                ex_list.append(0)

        new_state = [0 for _ in range(30)] + [time_used]
        new_state[action[0].item()] = 1
        new_state = torch.tensor(new_state)
        state = new_state
    x = torch.rand(4, 3, 224, 224).to(device)
    t = time.time()
    # print("ks:", ks_list)
    for i in range(len(model) - 1):
        if i >= 2 and i <= 21:
            if ks_list[i - 2] == 0:
                if (i - 2) % 4 != 0:
                    # print("passed", i)
                    continue
                else:
                    print("warning, got an illegal arch")
        # print(i)
        x = model[i](x)
    x = model[-1](x.view(-1, x.size()[1]))
    time_used = time.time() - t
    print(ks_list)
    print(ex_list)
    print(time_used)
    print(env.get_accuracy(ks_list, ex_list))

model = torch.load("full_subnet.pkl")
new_module = get_modulelist(model).to(device)
x = torch.rand(4, 3, 224, 224).to(device)
time_budget = warm_up_hardware(new_module, x, 15)

# model_test = get_modulelist(torch.load("ofa_mobilev3.pkl"))
# for i in range(2, 22):
#     print(i)
# #     if (len(new_module[i].conv.depth_conv) == 4):  # 存在SE模块
# #         model_layer.conv.depth_conv[3].channel = expanded_channel  # 这里SE的reduction也是可以作为RL的调节对象的
# #         print(type(model_layer.conv.depth_conv[3]))   
#     change_kernel(new_module[i], 5, 120, i - 2, kernel_mapping_weights)
# #     if (len(new_module[i].conv.depth_conv) == 4):  # 存在SE模块
# #         print(new_module[i].conv.depth_conv[3])
kernel_mapping_weights = get_kernel_mapping_weights()
agent = DQN(state_dim=31, action_dim=10)
acc_predictor = AccuracyPredictor()
env = enviroment(acc_predictor)
train_agent(new_module, agent, time_budget * 0.55, env)
get_new_model(agent, new_module, time_budget * 0.55,env)
# print(new_module[5].conv.depth_conv[0].weight.shape)
# x = torch.ones(4,3,224,224)
# x = x.to(device)
# # print(type(x))
# new_module = new_module.to(device)
# t_start = time.time()
# for j in range(len(new_module) - 1):
#     print(j)
#     x = new_module[j](x)
# x = new_module[-1](x.view(-1, x.size()[1]))
# print(time.time() - t_start)

# train the policy network
# class 