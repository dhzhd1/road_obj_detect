import _init_paths

import os
import sys
import logging
import pprint
import cv2
from config.config import config, update_config
from utils.image import resize, transform
import numpy as np
from random import random
import matplotlib.pyplot as plt
import math
import glob
import json
from Queue import Queue
from threading import Thread
import time

# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '1'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
update_config('./road_train_all.yaml')

sys.path.insert(0, os.path.join('../external/mxnet', config.MXNET_VERSION))
import mxnet as mx
from core.tester import im_detect, Predictor
from symbols import *
from utils.load_model import load_param
from utils.show_boxes import show_boxes
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper


def show_bbox_frame(bbox_frame_queue):
    cv2.namedWindow('video')
    while True:
        frame = bbox_frame_queue.get()
        if frame is None:
            break
        cv2.imshow('video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue



def get_raw_frame(raw_frame_queue):
    video_path = '../../video/Downtown.mp4'
    # video_path = '../../video/2017_0905_082324_006A.MOV'
    cap = cv2.VideoCapture(video_path)
    fps = math.floor(cap.get(5))
    while (cap.isOpened()):
        frame_id = cap.get(1)
        ret, frame = cap.read()
        # if frame_id % 2 != 0:
        #     raw_frame_queue.put(frame)
        #     continue
        raw_frame_queue.put(frame)
        continue
    cap.release()


def process_video_frame(raw_frame_queue, bbox_frame_queue):
    # get symbol
    pprint.pprint(config)
    config.symbol = 'resnet_v1_101_rfcn'
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)
    arg_params, aux_params = load_param('./output/rfcn/road_obj/road_train_all/all/' + 'rfcn_road', 19, process=True)

    # set up class names; Don't count the background in, even we are treat the background as label '0'
    num_classes = 4
    classes = ['vehicle', 'pedestrian', 'cyclist', 'traffic lights']

    target_size = config.SCALES[0][1]
    max_size = config.SCALES[0][1]

    while True:
        tic()
        i = 0
        data = []
        frame_list = []
        while len(data) < 15:
            frame = raw_frame_queue.get()
            if frame is None:
                continue
            if i < 2:
                i += 1
                frame, im_scale = resize(frame, target_size, max_size, stride=config.network.IMAGE_STRIDE)
                bbox_frame_queue.put(frame)
                continue
            frame, im_scale = resize(frame, target_size, max_size, stride=config.network.IMAGE_STRIDE)
            im_tensor = transform(frame, config.network.PIXEL_MEANS)
            im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
            data.append({'data': im_tensor, 'im_info': im_info})
            frame_list.append(frame)

        # get predictor
        data_names = ['data', 'im_info']
        label_names = []
        data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
        # print('Debug: [data] shape: {}, cont: {}'.format(type(data), data))
        max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
        # print('Debug: [max_data_shape] shape: {}, cont: {}'.format(type(max_data_shape), max_data_shape))
        provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
        # print('Debug: [provide_data] shape: {}, cont: {}'.format(type(provide_data), provide_data))
        provide_label = [None for i in xrange(len(data))]
        # print('Debug: [provide_label] shape: {}, cont: {}'.format(type(provide_label), provide_label))
        predictor = Predictor(sym, data_names, label_names,
                              context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                              provide_data=provide_data, provide_label=provide_label,
                              arg_params=arg_params, aux_params=aux_params)
        nms = gpu_nms_wrapper(config.TEST.NMS, 0)

        # Process video frame
        # image_names = ['frame']
        # for idx, frame in enumerate(frame_list):
        data_batch = mx.io.DataBatch(data=data, label=[], pad=0,
                                     provide_data=provide_data,
                                     provide_label=provide_label)
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
        # print("length: {}".format(len(data_batch.data)))
        # print('Debug: [scales] cont: {}'.format(scales))
        scores_all, boxes_all, data_dict_all = im_detect(predictor, data_batch, data_names, scales, config)
        # print('scores_all: Type: {}, Values: {}, Length: {}'.format(type(scores_all), scores_all, len(scores_all)))
        # print('boxes_all: Type: {}, Values: {}, Length: {}'.format(type(boxes_all), boxes_all, len(boxes_all)))
        # print('data_dict_all: Type: {}, Values: {}, length: {}'.format(type(data_dict_all), data_dict_all, len(data_dict_all)))
        # print('frame_list: Type: {}, Values: {}, Length: {}'.format(type(frame_list), frame_list, len(frame_list)))

        # print('scores_all: Type: {}, Length: {}, Values: {}'.format(type(scores_all[0]), len(scores_all[0]), scores_all[0]))
        # print(scores_all[0].shape)
        # print('boxes_all: Type: {}, Length: {}'.format(type(boxes_all), len(boxes_all)))
        # print(boxes_all[0].shape)
        # print('data_dict_all: Type: {}, length: {}'.format(type(data_dict_all), len(data_dict_all)))
        # print('frame_list: Type: {}, Length: {}'.format(type(frame_list), len(frame_list)))

        for idx, frame in enumerate(frame_list):
            # print('index: {}'.format(str(idx)))
            boxes = boxes_all[0].astype('f')
            scores = scores_all[0].astype('f')
            dets_nms = []
            # print(scores.shape)
            for j in range(1, scores.shape[1]):
                cls_scores = scores[:, j, np.newaxis]
                cls_boxes = boxes[:, 4:8] if config.CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
                cls_dets = np.hstack((cls_boxes, cls_scores))
                keep = nms(cls_dets)
                cls_dets = cls_dets[keep, :]
                cls_dets = cls_dets[cls_dets[:, -1] > 0.7, :]
                dets_nms.append(cls_dets)

            bbox_frame_queue.put(draw_bbox_on_frame(frame, dets_nms, classes, scale=scales[idx]))
        print(toc())

def main():
    raw_frame_queue = Queue()
    bbox_frame_queue = Queue()
    t1 = Thread(target=get_raw_frame, args=(raw_frame_queue,))
    t2 = Thread(target=process_video_frame, args=(raw_frame_queue, bbox_frame_queue,))
    t3 = Thread(target=show_bbox_frame, args=(bbox_frame_queue,))
    t1.start()
    t2.start()
    time.sleep(5)
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    cv2.destroyAllWindows()
    print 'done'


def draw_bbox_on_frame(frame, dets_nms, classes, scale=1.0):
    for cls_idx, cls_name in enumerate(classes):
        cls_dets = dets_nms[cls_idx]
        for det in cls_dets:
            bbox = det[:4] * scale
            # color = (random()*256, random()*256, random()*256)
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (127,127, 255)]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color[cls_idx], 1)
            if cls_dets.shape[1] == 5:
                score = det[-1]
                cv2.putText(frame,
                            '{:s} {:.3f}'.format(cls_name, score),
                            (int(bbox[0]), int(bbox[1])),
                            cv2.FONT_HERSHEY_PLAIN,
                            0.5,
                            color[cls_idx])
                # print('Bbox: {}, Class: {}, Prob: {}'.format(bbox, cls_name, score))
    return frame


if __name__ == '__main__':

    main()
