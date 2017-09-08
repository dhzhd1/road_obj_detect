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

import glob
import json

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


def main(video_file):
    # get symbol
    pprint.pprint(config)
    config.symbol = 'resnet_v1_101_rfcn'
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)
    arg_params, aux_params = load_param('./output/rfcn/road_obj/road_train_all/all/' + 'rfcn_road', 19, process=True)

    # set up class names; Don't count the background in, even we are treat the background as label '0'
    num_classes = 4
    classes = ['vehicle', 'pedestrian', 'cyclist', 'traffic lights']

    cap = cv2.VideoCapture(video_path)
    while (cap.isOpened()):
        ret, frame = cap.read()
        tic()
        data = []
        target_size = config.SCALES[0][1]
        max_size = config.SCALES[0][1]
        frame, im_scale = resize(frame, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(frame, config.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        data.append({'data': im_tensor, 'im_info': im_info})

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
        image_names=['frame']
        for idx, _ in enumerate(image_names):
            data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                         provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                         provide_label=[None])
            scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
            # print('Debug: [scales] cont: {}'.format(scales))

            scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, scales, config)
            boxes = boxes[0].astype('f')
            scores = scores[0].astype('f')
            dets_nms = []
            for j in range(1, scores.shape[1]):
                cls_scores = scores[:, j, np.newaxis]
                cls_boxes = boxes[:, 4:8] if config.CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
                cls_dets = np.hstack((cls_boxes, cls_scores))
                keep = nms(cls_dets)
                cls_dets = cls_dets[keep, :]
                cls_dets = cls_dets[cls_dets[:, -1] > 0.7, :]
                dets_nms.append(cls_dets)

            frame_with_bbox = draw_bbox_on_frame(frame, dets_nms, classes, scale=scales[0])
        cv2.imshow('video', frame_with_bbox)
        print 'Processing frame in {:.4f}s'.format(toc())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print 'done'


def draw_bbox_on_frame(frame, dets_nms, classes, scale=1.0):
    for cls_idx, cls_name in enumerate(classes):
        cls_dets = dets_nms[cls_idx]
        for det in cls_dets:
            bbox = det[:4] * scale
            color = (random()*256, random()*256, random()*256)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1)
            if cls_dets.shape[1] == 5:
                score = det[-1]
                cv2.putText(frame,
                            '{:s} {:.3f}'.format(cls_name, score),
                            (int(bbox[0]), int(bbox[1])),
                            cv2.FONT_HERSHEY_PLAIN,
                            0.5,
                            color)
                # print('Bbox: {}, Class: {}, Prob: {}'.format(bbox, cls_name, score))
    return frame


if __name__ == '__main__':
    video_path = '../../video/Highway.mp4'
    main(video_path)
