'''
Custom handler for wall segmentation
main part is about post process
make prediction img to vector msg
'''

import logging
import time
import torch
import io
import base64
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
from torchvision import transforms as T
import cv2
import numpy as np
import torch.nn.functional as F
import json
from wall_vector import *
import os

logger = logging.getLogger(name='HD-Wall')
n_classes = 2


class WallSegmentation(BaseHandler):
    def __init__(self):
        self.device = None
        self.model = None
        self._context = None
        self.initialized = False

    def initialize(self, context):
        # https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py#L77
        properties = context.system_properties
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        self.initialized = True
        self._context = context
        self.manifest = context.manifest
        # model path
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest['model']['serializedFile']
        # weight file path
        model_pt_path = os.path.join(model_dir, serialized_file)
        # model def file
        model_file = self.manifest["model"].get("modelFile", "")
        # initialize model
        self.model = self._load_pickled_model(
            model_dir, model_file, model_pt_path)
        # self.model = torch.jit.load(model_pt_path)
        self.model.to(device=self.device)
        self.model.eval()

    def preprocess(self, data):
        '''
        将输入数据转换成opencv常用的img格式 - h,w,c
        '''
        pre_data = []
        for row in data:
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)
            if isinstance(image, (bytearray, bytes)):
                # If the image is sent as bytesarray
                image = Image.open(io.BytesIO(image))
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            pre_data.append(image)
        return pre_data

    def inference(self, pre_data):
        '''
        使用滑窗分块预测img后，再进行拼接
        '''
        inf_data = []
        for img in pre_data:
            img_grids = make_grid(img.shape[:2])
            mask_pred = torch.zeros(
                1, n_classes, img.shape[0], img.shape[1]).to(device=self.device)
            weight_mask = torch.zeros(
                1, 1, img.shape[0], img.shape[1]).to(device=self.device)

            # CPU
            for points in img_grids:
                x1, x2, y1, y2 = points
                weight_mask[:, :, y1:y2, x1:x2] += 1
                img_i = img[y1:y2, x1:x2, :]/255
                img_i = img_i.transpose((2, 0, 1))
                img_i = torch.from_numpy(img_i).float().to(device=self.device)
                img_i = img_i.unsqueeze(0)
                with torch.no_grad():
                    mask_pred[:, :, y1:y2,
                              x1:x2] += self.model.forward(img_i).to(device=self.device)

            # # GPU
            # batch_size = 4
            # tensor_list = []
            # tensor_input = None
            # # 拼接待推理tensor
            # i = 0
            # for points in img_grids:
            #     i += 1
            #     x1, x2, y1, y2 = points
            #     weight_mask[:, :, y1:y2, x1:x2] += 1  # 更新权重
            #     img_i = img[y1:y2, x1:x2, :]/255
            #     img_i = img_i.transpose((2, 0, 1))
            #     img_i = torch.from_numpy(img_i).float().to(device=self.device)
            #     img_i = img_i.unsqueeze(0)
            #     # 对于起始点的初始化
            #     if i % batch_size == 1:
            #         tensor_input = img_i
            #         if i == len(img_grids):
            #             tensor_list.append(tensor_input)
            #             break
            #         continue
            #     # 拼接当前tensor
            #     tensor_input = torch.cat([tensor_input, img_i], 0)
            #     # 对于末尾数据的提取操作
            #     if i % batch_size == 0 or i == len(img_grids):
            #         tensor_list.append(tensor_input)
            #         continue
            # # 推理tensor并解构
            # i = 0  # 对应index
            # for tensor_input in tensor_list:
            #     with torch.no_grad():
            #         img_i = self.model.forward(
            #             tensor_input).to(device=self.device)
            #         for img_i_j in img_i:
            #             x1, x2, y1, y2 = img_grids[i]  # 对应index边界位置
            #             # 拼接成全图
            #             mask_pred[:, :, y1:y2,
            #                       x1:x2] += img_i_j.unsqueeze(0).to(device=self.device)
            #             i += 1

            # tensor to np - (h,w)
            mask_pred = mask_pred/weight_mask

            # mask_pred = F.softmax(mask_pred, dim=1)[0]
            # tf = T.Compose([T.ToPILImage(),
            #                 T.Resize((img.shape[0], img.shape[1])),
            #                 T.ToTensor()])
            # mask_pred = tf(mask_pred).squeeze()
            # mask_pred = F.one_hot(mask_pred.argmax(
            #     dim=0), n_classes).permute(2, 0, 1).numpy()
            # mask_pred = np.uint8(mask_pred[1]*255)

            inf_data.append(mask_pred)
        return inf_data

    def postprocess(self, pre_data, inf_data):
        wall_vectors = []
        for image, probs in zip(pre_data, inf_data):

            # process pridiction data - tensor -> np(h,w)
            probs = F.softmax(probs, dim=1)[0]
            tf = T.Compose([T.ToPILImage(),
                            T.Resize((image.shape[0], image.shape[1])),
                            T.ToTensor()])
            probs = tf(probs.cpu()).squeeze()
            probs = F.one_hot(probs.argmax(
                dim=0), n_classes).permute(2, 0, 1).numpy()
            probs = np.uint8(probs[1]*255)

            # (h,w)->(c,h,w) -> (h,w,c)
            img_pre = np.transpose(np.array([probs, probs, probs]), (1, 2, 0))
            # verctorizaton
            wall_vector_i = wall_vector(image, img_pre)
            wall_vectors.append(wall_vector_i)
        return wall_vectors

    def handle(self, data, context):
        start_time = time.perf_counter()
        pre_data = self.preprocess(data)
        preProcess_time = time.perf_counter()
        inf_data = self.inference(pre_data)
        inference_time = time.perf_counter()
        wall_vector = self.postprocess(pre_data, inf_data)
        postProcess_time = time.perf_counter()

        # log time cost
        time_total = postProcess_time-start_time
        time_preProcess = preProcess_time-start_time
        time_inferrence = inference_time-preProcess_time
        time_postProcess = postProcess_time-inference_time
        logger.info("total time : {}".format(time_total))
        logger.info("pre process time : {} , percent : {}%".format(
            time_preProcess, time_preProcess/time_total*100))
        logger.info("model inferrence time : {} , percent : {}%".format(
            time_inferrence, time_inferrence/time_total*100))
        logger.info("post process time : {} , percent : {}%".format(
            time_postProcess, time_postProcess/time_total*100))

        return wall_vector
