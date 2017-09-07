import cPickle
import cv2
import os
import io
import json
import numpy as np
from dataset.imdb import IMDB



from utils.tictoc import tic, toc
from bbox.bbox_transform import clip_boxes
import multiprocessing as mp


class RoadImages(IMDB):
    def __init__(self, image_set, root_path, data_path, result_path=None, mask_size=-1, binary_thresh=None):
        """
        fill basic information to initialize imdb
        :param image_set: training_set, validation_set, testing_set
        :param root_path: 'data', will write 'rpn_data', 'cache'
        :param data_path: 'data/rfcn'
        """
        super(RoadImages, self).__init__('RoadImages', image_set, root_path, data_path, result_path)
        self.root_path = root_path
        self.data_path = data_path
        self.notation_file = self._get_ann_file()
        self.notation_list = []


        # deal with class names
        road_obj_type = ['vehicle', 'pedestrian', 'cyclist', 'traffic lights' ]
        self.classes = ['__background__'] + road_obj_type
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, (0, 1, 2, 3, 20)))


        # load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        print 'num_images', self.num_images
        self.mask_size = mask_size
        self.binary_thresh = binary_thresh

        # deal with data name
        self.data_name = image_set

    def _get_ann_file(self):
        return os.path.join(self.root_path, self.data_path, 'all_label.idl')

    def _load_image_set_index(self):
        '''in the label.idl file, the key is "file name", it treated as a index'''
        file_list = []
        notation_list = []
        label_list = []
        with io.open(self.notation_file, 'r') as n:
            notation_content = n.readlines()
        for img_entry in notation_content:
            notation_list.append(json.loads(img_entry))
        print("There are {} records loaded into the list.".format(len(notation_list)))
        for record_entry in notation_list:
            for key in record_entry.keys():
                file_list.append(key)
        self.notation_list = notation_list
        return file_list

    def image_path_from_index(self, index):
        image_path = os.path.join(self.root_path, self.data_path, self.image_set, index)
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def gt_roidb(self):
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self._load_road_annotation(self.notation_list)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def _load_road_annotation(self, notation_list):
        """
        road_ann: [u'image_id', u'bbox', u'category_id']
        bbox:
            [x1, y1, x2, y2]
        :param index: road image id = file name
        :return: roidb entry
        """
        ''' Output field name: 
            image: image path
            boxes:[[index, top_left_x, top_left_y, bottom_right_x, bottom_right_y]...]
            category: [[0], [1], [2],[3],[20]]
        '''
        roi_rec = []
        for notation in self.notation_list:
            for key in notation.keys():
                file_path = self.image_path_from_index(key)
                labels = notation[key]
                bboxes = np.zeros((len(labels), 4), dtype=np.float32)
                classes = np.zeros(len(labels), dtype=np.int32)
                for i in xrange(len(labels)):
                    bboxes[i, :] = [labels[i][0], labels[i][1], labels[i][2], labels[i][3]]
                    classes[i] = 4 if labels[i][4]==20 else labels[i][4]

                roi_rec.append({'image': file_path,
                                'width': 640,
                                'height': 360,
                                'boxes': bboxes,
                                'gt_classes': classes,
                                'flipped': False})
        return roi_rec

    def evaluate_detections(self, detections, ann_type='bbox', all_masks=None):
        print("Not implemented")

