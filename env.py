import torch.nn.functional as F
import torch.nn as nn
import torch
import copy

STEP = 20
alpha = 0.03  # 环境中时延对于reward的占比
gamma = 10
def construct_maps(keys):
    d = dict()
    keys = list(set(keys))
    for k in keys:
        if k not in d:
            d[k] = len(list(d.keys()))
    return d


ks_map = construct_maps(keys=(3, 5, 7))
ex_map = construct_maps(keys=(3, 4, 6))
dp_map = construct_maps(keys=(2, 3, 4))


def scalar2block(scalar):  # 0~9,9:skip, 8:all max, 0:all min
    scalar = int(scalar)
    if scalar == 9:
        return 1, 0, 0
    # scalar -= 1
    ks_list = [3, 5, 7]
    channel_list = [3, 4, 6]
    channels = channel_list[scalar % 3]
    scalar = scalar // 3
    ks = ks_list[scalar % 3]
    scalar = scalar // 3
    return 0, ks, channels

class AccuracyPredictor:
    def __init__(self, pretrained=True, device='cuda:0'):
        self.device = device

        self.model = nn.Sequential(
            nn.Linear(128, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 1),
        )
        if pretrained:
#             self.model = torch.load()
            # load pretrained model
#             fname = download_url("https://hanlab.mit.edu/files/OnceForAll/tutorial/acc_predictor.pth", model_dir="D:\scientific_work\project1\once-for-all-master\.torch")
            self.model.load_state_dict(
                torch.load("acc_predictor.pth", map_location=torch.device('cpu'))
            )
        self.model = self.model.to(self.device)

    # TODO: merge it with serialization utils.
    @torch.no_grad()
    def predict_accuracy(self,population):  # population[i]->第i个神经网络，ks[j]即为第j个mobile block的ks，取值为[3, 5, 7]
        all_feats = []
        for sample in population:
            ks_list = copy.deepcopy(sample['ks'])
            ex_list = copy.deepcopy(sample['e'])
#             d_list = copy.deepcopy(sample['d'])
            r = copy.deepcopy(sample['r'])[0]
            feats = AccuracyPredictor.spec2feats(ks_list, ex_list, r).reshape(1, -1).to(self.device)
            all_feats.append(feats)
#         feats = AccuracyPredictor.spec2feats(ks_list, ex_list, r).reshape(1, -1).to(self.device)
        all_feats = torch.cat(all_feats, 0)
        pred = self.model(all_feats).cpu()
        return pred

    @staticmethod
    def spec2feats(ks_list, ex_list, r=224):  # r∈[112, 224]
        # This function converts a network config to a feature vector (128-D).
#         start = 0
#         end = 4
#         for d in d_list:
#             for j in range(start+d, end):
#                 ks_list[j] = 0
#                 ex_list[j] = 0
#             start += 4
#             end += 4

        # convert to onehot
        ks_onehot = [0 for _ in range(60)]
        ex_onehot = [0 for _ in range(60)]
        r_onehot = [0 for _ in range(8)]

        for i in range(20):
            start = i * 3
            if ks_list[i] != 0:
                ks_onehot[start + ks_map[ks_list[i]]] = 1
            if ex_list[i] != 0:
                ex_onehot[start + ex_map[ex_list[i]]] = 1

        r_onehot[(r - 112) // 16] = 1
        return torch.Tensor(ks_onehot + ex_onehot + r_onehot)

# def get_reward(time_budget, time_used, acc=None):
#     if legal_arch()

class enviroment:
    def __init__(self, acc_predictor):
        self.acc_predictor = acc_predictor
#     def 
    # TODO：更改reward，目前的reward越往后越叠加之前的时延，这是不对的，这样就更有可能丢掉后面的layer，因此尽量尝试每次reward更改执行该层的时延
    # 此外，还要研究在GPU上执行的时候应该怎样计算每层的时延
    def get_reward(self, step, this_layer_time, time_used, ks_list=None, ex_list=None):
        # steps_change_stride = [2, 6, 10, 18]
        # if step in steps_change_stride and ks_list[-1] == 0 :
        #     return -100
        if step < STEP - 1:
            reward = -alpha * this_layer_time
            # return 0
            # if time_used < 1:
            #     reward = -alpha * time_used
            # else:
            #     reward = -time_used
        else:
            accuracy = self.get_accuracy(ks_list, ex_list)
#             population = [{'ks':ks_list, 'e':ex_list, 'r':[224]}]
#             accuracy = self.acc_predictor.predict_accuracy(population)[0][0].item()
            if time_used <= 1.0:
                reward = accuracy * gamma
            else:
                reward = (accuracy - time_used) * gamma
        return reward
    
    def get_accuracy(self, ks_list, ex_list):
        population = [{'ks':ks_list, 'e':ex_list, 'r':[224]}]
        accuracy = self.acc_predictor.predict_accuracy(population)[0][0].item()
#         reward = accuracy - time_used * alpha
        return accuracy
#     def legal_arch(last_block, this_block):