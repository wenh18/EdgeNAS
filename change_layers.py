import torch.nn.functional as F
import torch.nn as nn
import torchvision
import time
import torch
from torch.nn.parameter import Parameter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_kernel_mapping_weights():
    model = torch.load("ofa_mobilev3.pkl")
    kernel_mapping_weights = {}
    block_id = 0
    for p in model.parameters():
        if p.shape == torch.Size([9, 9]):
            kernel_mapping_weights[block_id] = [p]
        elif p.shape == torch.Size([25, 25]):
            kernel_mapping_weights[block_id].append(p)
            block_id += 1
    return kernel_mapping_weights
# test:
# print(block_id)
# for i in range(len(kernel_mapping_weights[3])):
#     print(kernel_mapping_weights[3][i].shape)


def get_smaller_depth_kernel(kernel, sub_kernel_size, out_channel, in_channel, kernel_id, kernel_mapping_weights):
    # kernel_mapping_weights = kernel_mapping_weights.to(device)
#     out_channel = in_channel
    max_kernel_size = 7
    # 找到大核中小核的起止位置
    center = max_kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    
    filters = kernel.weight[:out_channel, :in_channel, start:end, start:end]  # conv.weight是filter的权重
    src_ks = 7
    if sub_kernel_size < max_kernel_size:
        start_filter = kernel.weight[:out_channel, :in_channel, :, :]  # start with max kernel
        for i in range(2):
            if src_ks <= sub_kernel_size:
                break
            target_ks = src_ks - 2
            
            # 找到当前小Kernel在大kernel中的位置
            center = src_ks // 2
            dev = target_ks // 2
            start, end = center - dev, center + dev + 1

            _input_filter = start_filter[:, :, start:end, start:end]
            _input_filter = _input_filter.contiguous()
            _input_filter = _input_filter.view(_input_filter.size(0), _input_filter.size(1), -1)
            _input_filter = _input_filter.view(-1, _input_filter.size(2))
            kernel_mapping_weights[kernel_id][1-i] = kernel_mapping_weights[kernel_id][1-i].to(device)
            _input_filter = F.linear(
                _input_filter, kernel_mapping_weights[kernel_id][1-i],
            )
            _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks ** 2)
            _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks, target_ks)
            start_filter = _input_filter
            src_ks = target_ks
        filters = start_filter
    return filters

def get_smaller_kernel(kernel, out_channel, in_channel):
    filters = kernel.weight[:out_channel, :in_channel, :, :]
    return filters

# def adjust_SEModule()

def change_kernel(model_layer, sub_kernel_size, expanded_channel, kernel_id, kernel_mapping_weights):
    # w=1.0版本
    # 改变扩张层
    expansion_layer_in_channel = model_layer.conv.inverted_bottleneck[0].in_channels
    model_layer.conv.inverted_bottleneck[0].weight = Parameter(get_smaller_kernel(model_layer.conv.inverted_bottleneck[0],
                                                                                out_channel=expanded_channel, in_channel=expansion_layer_in_channel))
    model_layer.conv.inverted_bottleneck[0].out_channels = expanded_channel
    model_layer.conv.inverted_bottleneck[1].weight = Parameter(model_layer.conv.inverted_bottleneck[1].weight[:expanded_channel])
    model_layer.conv.inverted_bottleneck[1].bias = Parameter(model_layer.conv.inverted_bottleneck[1].bias[:expanded_channel])
    model_layer.conv.inverted_bottleneck[1].num_features = expanded_channel
    model_layer.conv.inverted_bottleneck[1].running_mean = model_layer.conv.inverted_bottleneck[1].running_mean[:expanded_channel]
    model_layer.conv.inverted_bottleneck[1].running_var = model_layer.conv.inverted_bottleneck[1].running_var[:expanded_channel]
    
#     print(model_layer.conv.inverted_bottleneck[1].weight.shape)
    # 改变深度卷积
    model_layer.conv.depth_conv[0].weight = Parameter(get_smaller_depth_kernel(model_layer.conv.depth_conv[0], sub_kernel_size,
                                                                         expanded_channel, expanded_channel, kernel_id, kernel_mapping_weights))
    model_layer.conv.depth_conv[0].kernel_size = (sub_kernel_size, sub_kernel_size)
    model_layer.conv.depth_conv[0].groups = expanded_channel
    model_layer.conv.depth_conv[0].in_channels = expanded_channel
    model_layer.conv.depth_conv[0].out_channels = expanded_channel
    sub_padding_size = sub_kernel_size // 2
    model_layer.conv.depth_conv[0].padding = (sub_padding_size, sub_padding_size)
    model_layer.conv.depth_conv[1].weight = Parameter(model_layer.conv.depth_conv[1].weight[:expanded_channel])
    model_layer.conv.depth_conv[1].bias = Parameter(model_layer.conv.depth_conv[1].bias[:expanded_channel])
    model_layer.conv.depth_conv[1].num_features = expanded_channel
    model_layer.conv.depth_conv[1].running_mean = model_layer.conv.depth_conv[1].running_mean[:expanded_channel]
    model_layer.conv.depth_conv[1].running_var = model_layer.conv.depth_conv[1].running_var[:expanded_channel]
#     print(model_layer.conv.depth_conv[1].weight.shape)


# Sequential(
#   (reduce): Conv2d(192, 48, kernel_size=(1, 1), stride=(1, 1))
#   (relu): ReLU(inplace=True)
#   (expand): Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))
#   (h_sigmoid): Hsigmoid()
# )
# Given groups=1, weight of size [48, 192, 1, 1], expected input[8, 180, 1, 1] to have 192 channels, but got 180 channels instead

    if (len(model_layer.conv.depth_conv) == 4):  # 存在SE模块
        model_layer.conv.depth_conv[3].channel = expanded_channel  # 这里SE的reduction也是可以作为RL的调节对象的
        model_layer.conv.depth_conv[3].fc[0].weight = Parameter(get_smaller_kernel(model_layer.conv.depth_conv[3].fc[0],
                                                                                out_channel=expanded_channel // 4, in_channel=expanded_channel))
        model_layer.conv.depth_conv[3].fc[0].bias = Parameter(model_layer.conv.depth_conv[3].fc[0].bias[:(expanded_channel//4)])
        model_layer.conv.depth_conv[3].fc[2].weight = Parameter(get_smaller_kernel(model_layer.conv.depth_conv[3].fc[2],
                                                                                out_channel=expanded_channel, in_channel=expanded_channel // 4))
        model_layer.conv.depth_conv[3].fc[2].bias = Parameter(model_layer.conv.depth_conv[3].fc[2].bias[:expanded_channel])
#         print(type(model_layer.conv.depth_conv[3]))
#         model_layer.conv.depth_conv[3].reduction = 17
#         print((new_module[i].conv.depth_conv[3]).reduction)
    # 改变收缩层
    changed_channel = [32, 48, 96, 136, 192]
    if (kernel_id) % 4 == 0:
#         print((kernel_id) % 4)
        model_layer.conv.point_linear[0].weight = Parameter(get_smaller_kernel(model_layer.conv.point_linear[0],
                                                                                    out_channel=changed_channel[(kernel_id) // 4], in_channel=expanded_channel))
        model_layer.conv.point_linear[0].in_channels = expanded_channel
    else:
        model_layer.conv.point_linear[0].weight = Parameter(get_smaller_kernel(model_layer.conv.point_linear[0],
                                                                                    out_channel=expansion_layer_in_channel, in_channel=expanded_channel))
        model_layer.conv.point_linear[0].in_channels = expanded_channel
    # 不必再改变batch norm层
    # 改变batch normal层
# for i in range(3, 5):
#     change_kernel(new_module[i], 5, 180, i, kernel_mapping_weights)