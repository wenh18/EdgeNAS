from TestTensorRT import get_inference
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import copy
from torch.nn.parameter import Parameter
from change_layers import *
import cv2
# from TestTensorRT import get_inference


def get_img_np_nchw(filename):
    image = cv2.imread(filename)
    image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_cv = cv2.resize(image_cv, (224, 224))
    miu = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = np.array(image_cv, dtype=float) / 255.
    r = (img_np[:, :, 0] - miu[0]) / std[0]
    g = (img_np[:, :, 1] - miu[1]) / std[1]
    b = (img_np[:, :, 2] - miu[2]) / std[2]
    img_np_t = np.array([r, g, b])
    img_np_nchw = np.expand_dims(img_np_t, axis=0)
    return img_np_nchw

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

def Block2onnx(Block, input_size, block_idx, ks, depth):
    input_name = ['input']
    output_name = ['output']
    input = Variable(torch.randn(input_size)).cuda()
    Block = Block.cuda()
    # NewModel = new_model(self.ModuleList)
    print("completed building new model")
    # print(new_module[0])
    torch.onnx.export(Block, input, './TensorRTBlocks/OnnxBlocks/Block_{}_ks_{}_depth{}.onnx'.format(block_idx, ks, depth), input_names=input_name, output_names=output_name, verbose=True)

full_model = get_inference('full_subnet.onnx', './TensorRTBlocks/TrtBlocks/full_net.trt', fp16_mode=True, int8_mode=False, remove_original_file=False)
latency1, feat2 = full_model.run()


# def 
model = torch.load("full_subnet.pkl")
new_module = get_modulelist(model).cuda()
# print(new_module)
# x = Variable(torch.randn(1, 3, 224, 224)).cuda()
x = get_img_np_nchw('test.JPEG')
x = x.astype(dtype=np.float32)
x = Variable(torch.from_numpy(x)).cuda()
# input_sizes = [x.shape]
latencys = []
latency_sum = 0
for j in range(len(new_module) - 1):
    print("===================", j)
    # Block2onnx(new_module[j], x.shape, j, 7, 6)
    temp = copy.deepcopy(x)
    tmp = get_inference(onnx_model_path='./TensorRTBlocks/OnnxBlocks/Block_{}_ks_{}_depth{}.onnx'.format(j, 7, 6), 
    trt_model_path = './TensorRTBlocks/TrtBlocks/Block_{}_ks_{}_depth{}.trt'.format(j, 7, 6), fp16_mode=True, int8_mode=False, remove_original_file=False)
    x = new_module[j](x)
    # import pdb;pdb.set_trace()
    if j == 0:
        latency, out = tmp.run_block(temp, x.shape, h2d_time=False, d2h_time=False)
    else:
        latency, out = tmp.run_block(temp, x.shape, h2d_time=False, d2h_time=False)
    latency_sum += latency
    x = torch.from_numpy(out).cuda()
    # import pdb;pdb.set_trace()
    latencys.append(latency)
    # input_sizes.append(x.shape)
# Block2onnx(new_module[len(new_module) - 1], (x.view(-1, x.size()[1])).shape, len(new_module) - 1, 7, 6)
temp = copy.deepcopy(x)
x = new_module[-1](x.view(-1, x.size()[1]))
j = len(new_module) - 1
tmp = get_inference(onnx_model_path='./TensorRTBlocks/OnnxBlocks/Block_{}_ks_{}_depth{}.onnx'.format(j, 7, 6), 
trt_model_path = './TensorRTBlocks/TrtBlocks/Block_{}_ks_{}_depth{}.trt'.format(j, 7, 6), fp16_mode=True, int8_mode=False, remove_original_file=False)
# x = new_module[j](x)
latency, feat1 = tmp.run_block(temp.view(-1, temp.size()[1]), x.shape, h2d_time=False, d2h_time=False)
latencys.append(latency)
latency_sum += latency
print(latencys)
print("latency of original net:", latency1)
print("sum latency of blocks:", latency_sum)


# feat1 = feat1.cpu().data.numpy()
# feat2 = feat2.cpu().data.numpy()
mse = np.mean((feat1 - feat2)**2)
print(mse)
# for i in range(len(new_module)):
#     # tmp = get_inference(onnx_model_path='./TensorRTBlocks/OnnxBlocks/Block_{}_ks_{}_depth{}.onnx'.format(i, 7, 6), 
#     # trt_model_path = './TensorRTBlocks/TrtBlocks/Block_{}_ks_{}_depth{}.trt', remove_original_file=False)
#     tmp.get_engine(max_batch_size=1, onnx_file_path='./TensorRTBlocks/OnnxBlocks/Block_{}_ks_{}_depth{}.onnx'.format(i, 7, 6), 
#     engine_file_path='./TensorRTBlocks/TrtBlocks/Block_{}_ks_{}_depth{}.trt'.format(i, 7, 6), fp16_mode=True, int8_mode=False, save_engine=True)
# # print(input_sizes)


