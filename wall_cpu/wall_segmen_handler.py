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

logger = logging.getLogger(name='HD-Wall')
n_classes=2

class WallSegmentation(BaseHandler):
    def __init__(self):
        '''
        声明需要自定义的变量
        '''
        self.device=None

    def initialize(self, context):
        '''
        初始化
        '''
        properties = context.system_properties
        device_name="cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        logger.info("using device : {}".format(device_name))

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
        inf_data=[]
        for img in pre_data:
            img_grids=make_grid(img.shape[:2])
            mask_pred=torch.zeros(1,n_classes,img.shape[0],img.shape[1])
            weight_mask=torch.zeros(1,1,img.shape[0],img.shape[1])
            for points in img_grids:
                x1,x2,y1,y2=points
                weight_mask[:,:,y1:y2,x1:x2]+=1
                img_i=img[y1:y2,x1:x2,:]/255
                img_i=img_i.transpose((2, 0, 1))
                img_i = torch.from_numpy(img_i).float()
                img_i = img_i.unsqueeze(0)
                with torch.no_grad():
                    mask_pred[:,:,y1:y2,x1:x2] +=self.model.forward(img_i)
            mask_pred=mask_pred/weight_mask
            inf_data.append(mask_pred)
        return inf_data


    def postprocess(self, pre_data, inf_data):
        wall_vectors=[]
        for image,probs in zip(pre_data,inf_data):
            # process pridiction data - (h,w,c)
            probs = F.softmax(probs, dim=1)[0]
            tf = T.Compose([T.ToPILImage(),
                            T.Resize((image.shape[0], image.shape[1])),
                            T.ToTensor()])
            full_mask = tf(probs.cpu()).squeeze()
            full_mask = F.one_hot(full_mask.argmax(dim=0), n_classes).permute(2, 0, 1).numpy()
            result_img=np.uint8(full_mask[1]*255)
            img_pre=np.transpose(np.array([result_img,result_img,result_img]),(1,2,0))
            # verctorizaton
            wall_vector_i=wall_vector(image,img_pre)
            wall_vectors.append(wall_vector_i)
        return wall_vectors

    
    def handle(self, data, context):
        start_time=time.perf_counter()
        pre_data=self.preprocess(data)
        preProcess_time=time.perf_counter()
        inf_data=self.inference(pre_data)
        inference_time=time.perf_counter()
        wall_vector=self.postprocess(pre_data, inf_data)
        postProcess_time=time.perf_counter()

        #log time cost
        time_total=postProcess_time-start_time
        time_preProcess=preProcess_time-start_time
        time_inferrence=inference_time-preProcess_time
        time_postProcess=postProcess_time-inference_time
        logger.info("total time : {}".format(time_total))
        logger.info("pre process time : {} , percent : {}%".format(time_preProcess,time_preProcess/time_total*100))
        logger.info("model inferrence time : {} , percent : {}%".format(time_inferrence,time_inferrence/time_total*100))
        logger.info("post process time : {} , percent : {}%".format(time_postProcess,time_postProcess/time_total*100))

        return wall_vector