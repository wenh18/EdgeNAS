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
# from pth2onnx import pth2onnx
# from TestTensorRT import get_inference
# TODO：目前的问题：探索不够，没有得到正的奖励，解决方法：增大EPOCH，增大随机探索轮数
# 不知道目前在神经网络的哪个位置。解决方法：位置编码或者使用RNN，发现RNN没有起到效果，估计需要使用位置编码了，可不可以理解为LSTM仅仅让agent知道前几个层的决策，并没有知道位置呢？
# 知道的前几层的决策信息不够多。解决方法：输入3个决策(33维)或者使用RNN
# 把agent也放到GPU上吧
# 目前只有1/10的概率跳层，这样有点小，应该增加几率？ preoritized replay 两个trick dualing network
# 包装成gym的environment
# python缓存操作，让搬运更高效
# 问题：depth不能随意跳呀，只能跳后面的，对此有三种解决办法，一种是一旦智能体决定跳过某一层，则该组的所有层全部跳过。一种是增加智能体的动作，让它在每一组的开始决定该组的深度，一种是让它正常学习，学完后一旦有跳中间层的行为就把该组后面的也全部删掉，最后再测评准确率
# 当然，这里也同样是一个可以优化的点，因为关于是否跳层，我们只有3^5 = 243种选择，是否可以打一个表，根据输入的时间先大致估计出如何跳层，再在里面选ks和channel

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

def step2state(step, STEP):
    step_state = torch.zeros(STEP)
    step_state[int(step)] = 1
    return step_state

class Net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=30):  # state_dim:9, action_dim:8
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(128, 128)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(128, action_dim)
        self.fc3.weight.data.normal_(0, 0.1)
        self.embedding = nn.Linear(state_dim, state_dim)# nn.Embedding(state_dim - 1, state_dim - 1)  # 注意embedding是layer normalization，可以之后试试batch normalization
        self.embedding.weight.data.normal_(0, 0.1)
        self.hidden = None
        self.batch_size = 1
        

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class DQN(object):
    def __init__(self, state_dim, action_dim, Epsilon):  # , scalar_state_dim=2
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.eval_net, self.target_net = Net(state_dim, action_dim), Net(state_dim, action_dim)
        self.learn_step_counter = 0  #for target updating
        self.memory_counter = 0  #for storing memory
        self.memory = np.zeros((Memory_capacity, state_dim * 2 + 3)) #innitialize memory
        self.optimizer = optim.Adam(self.eval_net.parameters(),lr=LR)
        self.loss_func = nn.MSELoss()
        self.Epsilon = Epsilon

    def choose_action(self, x, test=False):
        if test:
            self.Epsilon = 0.0
        self.eval_net.batch_size = 1
        self.target_net.batch_size = 1
        x = Variable(torch.unsqueeze(x,0))
        if np.random.uniform() > self.Epsilon:
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1]
        else:
            action = torch.Tensor([np.random.randint(0, self.action_dim)])
        return action


    def store_transition(self,s,a,r,s_, end):
        transition = np.hstack((s,[a,r],s_, end))
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
        q_eval = self.eval_net(b_s).gather(1,b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = Variable(torch.FloatTensor(self.eval_net.batch_size, 1))
        for b in range(self.eval_net.batch_size):
            if b_end[b] == 0:
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
    return True
        

def train_agent(agent, time_budget, env):
    print("object time:", time_budget)
    x = torch.rand(4, 3, 224, 224).to(device)
    ep_rs = []
    ACCs = []
    LATENs = []
    start_training = False
    for episode in range(EPISODE):
        if cuda_available:
            torch.cuda.empty_cache()
        # new_module = new_module.to(device)
        print("----------------", episode, "----------------")
        # x = torch.rand(4, 3, 224, 224).to(device)  # 待改进：将imagenet数据加载进来
        # for i in range(2):
        #     x = new_module[i](x)
        state = torch.zeros(30)
        state[0] = 1
        
        input_channels = [24, 32, 48, 96, 136, 192]
        channel_idx = 0
        ks_list = []  # 用于网络精度预测
        ex_list = []
        ks_lists = []
        ex_lists = []
        
        ep_r = 0
        skip_flag = 0
        for step in range(STEP):
            
            if (step - 1) % 4 == 0:
                channel_idx += 1
            if step % 4 == 0:
                skip_flag = 0
            if skip_flag == 1:
                continue
            action = agent.choose_action(state).numpy()
            skip, ks, channels_extension = scalar2block(action[0].item())
            if not legal_arch(step, skip, ks_list):
                reward = -100
                agent.store_transition(state, action, reward, state, 1)  # 1->end
                ep_r += reward
                print("not legal")
                break
            if skip == 0:
                # tmp_module = copy.deepcopy(new_module[step + 2]).to(device)
                # change_kernel(model_layer=tmp_module, sub_kernel_size=ks, expanded_channel=input_channels[channel_idx] * channels_extension, 
                #               kernel_id=step, kernel_mapping_weights=kernel_mapping_weights)
                # x = tmp_module(x)
                ks_list.append(ks)
                ex_list.append(channels_extension)
            else:
                skip_flag = 1
                for skip_layer_idx in range(4):
                    if (step + skip_layer_idx) % 4 == 0:
                        break
                    ks_list.append(0)
                    ex_list.append(0)
            new_state = [0 for _ in range(30)]
            new_state[int(action[0].item())] = 1
            new_state = torch.Tensor(new_state)
            new_state[10:30] = step2state(len(ks_list) - 1, STEP)
            if len(ks_list) == STEP:
                # for i in range(22, 26):
                #     tmp_module_list.append(new_module[i])
                try:
                    reward, time_used, acc = env.get_reward(time_budget, ks_list=ks_list, ex_list=ex_list)
                except:
                    print("oops, something unpredictable happened")
                    break
                # del tmp_module_list
                ACCs.append(acc)
                LATENs.append(time_used)
                print(reward)
                ep_r += reward
                agent.store_transition(state, action, reward, new_state, 0)
            else:
                agent.store_transition(state, action, 0, new_state, 1)
            
            if agent.memory_counter > Memory_capacity:
                agent.Epsilon = agent.Epsilon * EPSILON_DECLINING_RATE
                agent.learn()
                if not start_training:
                    print("training is starting................................................................................................")
                    start_training = True
            
            state = new_state
        ep_rs.append(ep_r)
        if len(ks_list) != 20:
            ACCs.append(0)
            LATENs.append(10)
        if episode % 20 == 0:
            print(ks_list)
            print(ex_list)
            ks_lists.append(ks_list)
            ex_lists.append(ex_list)
        if episode % (int(EPISODE / 5)) == 0:
            PATH = 'EPOCH' + str(episode) + 'reward' + str(ep_r) + 'agent.pth'
            torch.save(agent.eval_net, PATH)
    return ep_rs, ACCs, LATENs

def get_new_model(agent, time_budget, env):
    if cuda_available:
        torch.cuda.empty_cache()
    # model = model.to(device)
    # x = torch.rand(4, 3, 224, 224).to(device)  # 待改进：将imagenet数据加载进来
    # new_module_list = nn.ModuleList()
    # for i in range(2):
    #     x = model[i](x)
    #     new_module_list.append(model[i])
    state = torch.zeros(30)
    state[0] = 1
    input_channels = [24, 32, 48, 96, 136, 192]
    channel_idx = 0
    ks_list = []  # 用于网络精度预测
    ex_list = []
    skip_flag = 0

    for step in range(STEP):
        if (step - 1) % 4 == 0:
            channel_idx += 1
        if step % 4 == 0:
            skip_flag = 0
        if skip_flag == 1:
            continue
        action = agent.choose_action(state).numpy()
        skip, ks, channels_extension = scalar2block(action[0].item())
        if skip == 0:
            # change_kernel(model_layer=model[step + 2], sub_kernel_size=ks, expanded_channel=input_channels[channel_idx] * channels_extension, 
            #             kernel_id=step, kernel_mapping_weights=kernel_mapping_weights)
            # x = model[step + 2](x)
            # new_module_list.append(model[step + 2])
            ks_list.append(ks)
            ex_list.append(channels_extension)
        else:
            # if step % 4 == 0:
            #     print("warning, got an illegal arch")
            #     change_kernel(model_layer=model[step + 2], sub_kernel_size=3, expanded_channel=input_channels[channel_idx] * 3, 
            #             kernel_id=step, kernel_mapping_weights=kernel_mapping_weights)
            #     x = model[step + 2](x)
            #     new_module_list.append(model[step + 2])
            #     ks_list.append(ks)
            #     ex_list.append(channels_extension)
            # else:
            skip_flag = 1
            for skip_layer_idx in range(4):
                if (step + skip_layer_idx) % 4 == 0:
                    break
                ks_list.append(0)
                ex_list.append(0)

        new_state = [0 for _ in range(30)]
        new_state[int(action[0].item())] = 1
        new_state = torch.Tensor(new_state)
        new_state[10:30] = step2state(len(ks_list) - 1, STEP)
        state = new_state
    # for i in range(22, 26):
    #     new_module_list.append(model[i])
    # x = torch.rand(4, 3, 224, 224).to(device)
    _, time_used, acc = env.get_reward(time_budget, ks_list=ks_list, ex_list=ex_list, delete_original_file=False)
    print(ks_list)
    print(ex_list)
    print(time_used)
    print(acc)

if __name__ == '__main__':
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_available else "cpu")
    # PATH = 
    Gamma = 0.95  # reward discount
    LR = 0.0005
    # SEARCH_RATE = 0.3
    EPSILON_DECLINING_RATE = 0.9
    # Epsilon = 0.9  # 应该是递减的！
    Memory_capacity = 8000  # memory_capacity≈EPISODE*STEP*0.6，此时需要占用约30M内存，也许不够吧？我们能不能优先替换掉一些reward非常小的记忆呢？
    Batch_size = 32
    Target_replace_iter = 100  # target update frequency
    EPISODE = 3000
    STEP = 20
    # model = torch.load("full_subnet.pkl")
    # new_module = get_modulelist(model).to(device)
    x = torch.rand(4, 3, 224, 224).to(device)
    agent = DQN(state_dim=30, action_dim=10, Epsilon=0.5)
    acc_predictor = AccuracyPredictor()
    env = enviroment(acc_predictor)
    _, time_budget, acc = env.get_reward(1.0, [7 for i in range(20)], [6 for i in range(20)])
    ep_rs, ACCs, LATENs = train_agent(agent, time_budget * 0.55, env)
    print(ep_rs)
    print(ACCs)
    print(LATENs)
    get_new_model(agent, time_budget * 0.55, env)