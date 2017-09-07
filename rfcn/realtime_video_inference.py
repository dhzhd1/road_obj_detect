import cv2
import numpy as np

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


# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
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

    # Load model
    arg_params, aux_params = load_param('./output/rfcn/road_obj/road_train_all/all/' + 'rfcn_road', 19, process=True)


    # set up class names; Don't count the background in, even we are treat the background as label '0'
    num_classes = 4
    classes = ['vehicle', 'pedestrian', 'cyclist', 'traffic lights']
    video_path = video_file
    cap = cv2.VideoCapture(video_path)
    while (cap.isOpened()):
        tic()
        ret, frame = cap.read()
        # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        data = []
        frame, im_scale = resize(frame, config.SCALES[0][1], config.SCALES[0][1], stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(frame, config.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        data.append({'data': im_tensor, 'im_info': im_info})

        # get predictor
        data_names = ['data', 'im_info']
        label_names = []
        data = [[im_tensor, im_info]]
        max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
        provide_data = [[(k, v.shape) for k, v in zip(data_names, data[0])]]
        provide_label = [None]
        predictor = Predictor(sym, data_names, label_names,
                              context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                              provide_data=provide_data, provide_label=provide_label,
                              arg_params=arg_params, aux_params=aux_params)
        nms = gpu_nms_wrapper(config.TEST.NMS, 0)


        data_batch = mx.io.DataBatch(data=[data[0]], label=[], pad=0, index=0,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[0])]],
                                     provide_label=[None])
        scales = [data_batch.data[0][1].asnumpy()[0, 2]]


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

        frame_with_bbox = draw_bbox_on_frame(frame, dets_nms, classes, scale=1.0)
        print 'Process Frame in {:.4f}s'.format(toc())
        cv2.imshow('video', frame_with_bbox)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def draw_bbox_on_frame(frame, dets_nms, classes, scale=1.0):
    for cls_idx, cls_name in enumerate(classes):
        cls_dets = dets_nms[cls_idx]
        for det in cls_dets:
            bbox = det[:4] * scale
            color = (random(), random(), random())
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            if cls_dets.shape[1] == 5:
                score = det[-1]
                cv2.putText(frame,
                            '{:s} {:.3f}'.format(cls_name, score),
                            (int(bbox[0]), int(bbox[1])),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            4,
                            color,
                            2,
                            cv2.LINE_AA)
    return frame

def draw_bbox_on_frame_2(frame, dets, classes, scale = 1.0):
    plt.cla()
    plt.axis("off")
    plt.imshow(frame)
    for cls_idx, cls_name in enumerate(classes):
        cls_dets = dets[cls_idx]
        for det in cls_dets:
            bbox = det[:4] * scale
            color = (random(), random(), random())
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor=color, linewidth=2.5)
            plt.gca().add_patch(rect)

            if cls_dets.shape[1] == 5:
                score = det[-1]
                plt.gca().text(bbox[0], bbox[1],
                               '{:s} {:.3f}'.format(cls_name, score),
                               bbox=dict(facecolor=color, alpha=0.5), fontsize=9, color='white')
    return frame

if __name__=='__main__':
    video_path = '../../video/Highway.mp4'
    main(video_path)