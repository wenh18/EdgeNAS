import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch
import os
import time
from PIL import Image
import cv2
import torchvision
import copy
filename = 'test.JPEG'
max_batch_size = 1
# onnx_model_path = 'full_moduleList.onnx'
# onnx_model_path = 'full_subnet.onnx'

TRT_LOGGER = trt.Logger()  # This logger is required to build an engine

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class get_inference(object):

    def __init__(self, onnx_model_path, trt_model_path, fp16_mode=True, int8_mode=False, remove_original_file=True):
        self.onnx_model_path = onnx_model_path
        self.remove_original_file = remove_original_file
        self.trt_model_path = trt_model_path
        self.fp16_mode = fp16_mode
        self.int8_mode = int8_mode

    def get_img_np_nchw(self, filename):
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


    def allocate_buffers(self, engine):
        # print("engine:", engine)
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            if size < 0:
                size *= -1
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream


    def get_engine(self, max_batch_size=1, onnx_file_path="", engine_file_path="", \
                fp16_mode=False, int8_mode=False, save_engine=False,
                ):
        """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

        def build_engine(max_batch_size, save_engine):
            """Takes an ONNX file and creates a TensorRT engine to run inference with"""
            explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            with trt.Builder(TRT_LOGGER) as builder, \
                    builder.create_network(explicit_batch) as network, \
                    trt.OnnxParser(network, TRT_LOGGER) as parser:

                builder.max_workspace_size = 1 << 31  # 30  # Your workspace size
                builder.max_batch_size = max_batch_size
                # pdb.set_trace()
                builder.fp16_mode = fp16_mode  # Default: False
                builder.int8_mode = int8_mode  # Default: False
                if int8_mode:
                    # To be updated
                    raise NotImplementedError

                # Parse model file
                if not os.path.exists(onnx_file_path):
                    quit('ONNX file {} not found'.format(onnx_file_path))

                print('Loading ONNX file from path {}...'.format(onnx_file_path))
                with open(onnx_file_path, 'rb') as model:
                    print('Beginning ONNX file parsing')
                    parser.parse(model.read())
                # #########################
                last_layer = network.get_layer(network.num_layers - 1)
                network.mark_output(last_layer.get_output(0))
                # explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
                # #########################
                print('Completed parsing of ONNX file')
                print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))

                engine = builder.build_cuda_engine(network)
                print("Completed creating Engine")

                if save_engine:
                    with open(engine_file_path, "wb") as f:
                        f.write(engine.serialize())
                return engine
        # print("path:", engine_file_path)
        if os.path.exists(engine_file_path):
            # If a serialized engine exists, load it instead of building a new one.
            print("Reading engine from file {}".format(engine_file_path))
            with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            return build_engine(max_batch_size, save_engine)

    # 能不能把context整合起来，去掉内存搬移的过程，即传进来的是一个context的list
    def do_inference(self, context, bindings, inputs, outputs, stream, include_h2d_time=False, include_d2h_time=False, batch_size=1):
        # print("the type of context:", type(context))
        # print(context)
        # Transfer data from CPU to the GPU. htod->host to device
        t1 = time.time()
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        t2 = time.time()
        context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        # print("output of inference, ", outputs)
        t3 = time.time()
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        # t4 = time.time()
        stream.synchronize()
        t5 = time.time()
        if include_h2d_time:
            latency = t3 - t1
        elif include_d2h_time:
            latency = t5 - t2
        else:
            latency = t3 - t2
        # Return only the host outputs.
        return [out.host for out in outputs], latency, t2 - t1


    def postprocess_the_outputs(self, h_outputs, shape_of_output):
        h_outputs = h_outputs.reshape(*shape_of_output)
        return h_outputs


    def run_block(self, x, output_size, h2d_time=False, d2h_time=False):
        max_batch_size = 1
        with self.get_engine(max_batch_size, self.onnx_model_path, self.trt_model_path, self.fp16_mode, self.int8_mode, True) as engine:
            with engine.create_execution_context() as context:
                inputs, outputs, bindings, stream = self.allocate_buffers(engine) # input, output: host # bindings
                
                new_x = x.reshape(-1)
                # import pdb;pdb.set_trace()
                inputs[0].host = new_x.cpu().detach().numpy()
                # import pdb;pdb.set_trace()
                average_time = 0
                for _ in range(100000):
                    t1 = time.time()
                    trt_outputs, latency, _ = self.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, include_h2d_time=h2d_time, include_d2h_time=d2h_time) # numpy data
                    t2 = time.time()
                    average_time = average_time + latency
                average_time /= 100000
                # import pdb;pdb.set_trace()
                feat = self.postprocess_the_outputs(trt_outputs[0], output_size)
                return average_time, feat

    def run(self):
        max_batch_size = 1
        # print(torch.cuda.is_available())
        img_np_nchw = self.get_img_np_nchw(filename)
        img_np_nchw = img_np_nchw.astype(dtype=np.float32)

        # These two modes are dependent on hardwares
        # fp16_mode = False
        # int8_mode = False
        # trt_engine_path = './model_fp16_{}_int8_{}.trt'.format(fp16_mode, int8_mode)
        # Build an engine
        with self.get_engine(max_batch_size, self.onnx_model_path, self.trt_model_path, self.fp16_mode, self.int8_mode, True) as engine:
            
            # engine = self.get_engine(max_batch_size, self.onnx_model_path, trt_engine_path, fp16_mode, int8_mode)
            # Create the context for this engine
            # import pdb;pdb.set_trace()
            with engine.create_execution_context() as context:
                # Allocate buffers for input and output
                inputs, outputs, bindings, stream = self.allocate_buffers(engine) # input, output: host # bindings

                # Do inference
                shape_of_output = (max_batch_size, 1000)
                # Load data to the buffer
                inputs[0].host = img_np_nchw.reshape(-1)
                # print("inputs shape:", inputs[0].shape)
                # import pdb;pdb.set_trace()
                # inputs[1].host = ... for multiple input
                average_time = 0
                averate_copy_time = 0
                for _ in range(100000):
                    t1 = time.time()
                    trt_outputs, latency, cpy = self.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream) # numpy data
                    t2 = time.time()
                    average_time = average_time + t2 - t1
                    averate_copy_time += cpy
                average_time /= 100000
                averate_copy_time /= 100000
                print('--------------', averate_copy_time)
                feat = self.postprocess_the_outputs(trt_outputs[0], shape_of_output)
                print('TensorRT ok') ###########################################################################
                # model = torchvision.models.resnet50(pretrained=True).cuda()
                model = torch.load("full_subnet.pkl").cuda()
                full_model = model.eval()

                input_for_torch = torch.from_numpy(img_np_nchw).cuda()
                feat_2= full_model(input_for_torch)
                feat_2 = feat_2.cpu().data.numpy()
                
                mse = np.mean((feat - feat_2)**2)
                print("tensorrt and original model mse:", mse)
                ##################################################################
                print("Inference time with the TensorRT engine: {}".format(average_time))
                if self.remove_original_file:
                    os.remove(self.onnx_model_path)
                return average_time, feat

# model_inference = get_inference('full_moduleList.onnx')
# print(model_inference.run())
if __name__ == '__main__':
    model_inference = get_inference('test.onnx', remove_original_file=False)
    print(model_inference.run())