import io
import json
import os
import mxnet as mx
import numpy as np
import cv2
from random import shuffle


def load_notation(notation_path):
    notation_list = []
    with io.open(notation_path, 'r') as n:
        notation_content = n.readlines()
    for img_entry in notation_content:
        notation_list.append(json.loads(img_entry))
    print("There are {} records loaded into the list.".format(len(notation_list)))
    return notation_list

def get_filename_list(notation_list):
    file_names = []
    for entry in notation_list:
        for key in entry.keys():
            file_names.append(key)
    return file_names


class Data:
    def __init__(self, notation_file, dataset_folder, split_ratio=0.8):
        self.dataset_folder = dataset_folder
        self.split_ratio = split_ratio
        self.notation_list = load_notation(notation_file)
        self.rec_folder_name = '../Data_rec/'
        if not os.path.exists(self.rec_folder_name):
            os.mkdir(self.rec_folder_name)


    def generate_rec_file(self, image_list=None):
        # If the image list is None, the data will be split into two rec file: training data and validation data
        # based  on self.split_ratio.
        # If the image list is not None, function will take the image names from list and build a rec file which
        # in this list, for overfitting test.
        if image_list is not None:
            rec_file_name = 'small_dataset.rec'
            idx_file_name = 'small_dataset.idx'
            overfit_dataset = mx.recordio.MXIndexedRecordIO(os.path.join(self.rec_folder_name, idx_file_name),
                                                            os.path.join(self.rec_folder_name, rec_file_name), 'w')
            for i in range(len(image_list)):
                image_name = image_list[i]
                print("Processing {} ...".format(image_name))
                image_notation = self.get_notation(image_name)
                image_content = cv2.imread(os.path.join(self.dataset_folder, image_name))
                resized_image, new_notation = self.image_resize_notation_transform(image_content, image_notation)
                header = mx.recordio.IRHeader(flag=0, label=new_notation.flatten(), id=i, id2=0)
                pack = mx.recordio.pack_img(header, resized_image, quality=100, img_fmt=".jpg")
                overfit_dataset.write_idx(i, pack)
                print("Picture {} has been added into rec-file".format(image_name))

            overfit_dataset.close()
            print("There {} picture(s) have been added into {}".format(str(len(image_list)), rec_file_name))
        else:
            train_rec_name = 'train_dataset.rec'
            train_idx_name = 'train_dataset.idx'
            val_rec_name = 'val_dataset.rec'
            val_idx_name = 'val_dataset.idx'
            train_dataset = mx.recordio.MXIndexedRecordIO(os.path.join(self.rec_folder_name, train_idx_name),
                                                          os.path.join(self.rec_folder_name, train_rec_name), 'w')
            val_dataset = mx.recordio.MXIndexedRecordIO(os.path.join(self.rec_folder_name, val_idx_name),
                                                        os.path.join(self.rec_folder_name, val_rec_name), 'w')
            train_image_num = int(len(self.notation_list) * self.split_ratio)
            shuffle(self.notation_list)
            for i in range(len(self.notation_list)):
                for key in self.notation_list[i].keys():
                    image_name = key
                print("Processing {} ...".format(image_name))
                image_notation = self.get_notation(image_name)
                image_content = cv2.imread(os.path.join(dataset_folder, image_name))
                resized_image, new_notation = self.image_resize_notation_transform(image_content, image_notation)
                header = mx.recordio.IRHeader(flag=0, label=new_notation.flatten(), id=i, id2=0)
                pack = mx.recordio.pack_img(header, resized_image, quality=100, img_fmt=".jpg")
                if i < train_image_num:
                    train_dataset.write_idx(i, pack)
                    print("Picture {} has been added into training dataset.".format(image_name))
                else:
                    val_dataset.write_idx(i-train_image_num, pack)
                    print("Picture {} has been added into validation dataset.".format(image_name))
            train_dataset.close()
            val_dataset.close()
            print("There are {} images have been added into training dataset".format(str(train_image_num)))
            print("There are {} images have been added into validation dataset".format(str(len(self.notation_list)-train_image_num)))


    def get_notation(self, image_name):
        for notation in self.notation_list:
            if image_name in notation.keys():
                return notation[image_name]


    def image_resize_notation_transform(self, image, notations, grid_size, target_size=224, dscale=16):
        # TODO: Need to read the paper to implement the correct notation/image resize function
        # Image resize. Ref: https://github.com/giorking/mx-rfcn/blob/master/helper/processing/image_processing.py
        # dscale = 16 : On the author caffe configuration, the spatial_scale is 0.0625 (14/224) is 16-times down scale

        im_h, im_w = image.shape[:2]
        resized_image = cv2.resize(image, dsize=(target_size, target_size))
        new_notations = np.zeros(grid_size)
        for notation in notations:
            # normalize the coordinates
            top_left_x, top_left_y, bottom_right_x, bottom_right_y, category= notation
            top_left_x = 1.0 * top_left_x / im_w
            top_left_y = 1.0 * top_left_y / im_h
            bottom_right_x = 1.0 * bottom_right_x / im_w
            bottom_right_y = 1.0 * bottom_right_y / im_h
            assert top_left_y<=1 and top_left_x<=1 and bottom_right_y<=1 and bottom_right_x<=1, \
                "The coordinates haven't be normalized in (0,1]"
            i, j, tl_x, tl_y, k, l, br_x, br_y = \
                self.coordinate_transform_on_score_map([top_left_x, top_left_y, bottom_right_x, bottom_right_y],
                                                       grid_size, dscale, target_size )
            label_vector = np.asarray([category, tl_x, tl_y, br_x, br_y])
            

        return resized_image, new_notations

    def coordinate_transform_on_score_map(self, cordsXY, grid_size, dscale=16, target_size=224 ):
        tl_x, tl_y, br_x, br_y = cordsXY
        # grid_size(w, h)
        i = int(np.floor(tl_x/(1.0/grid_size[0])))
        j = int(np.floor(tl_y/(1.0/grid_size[1])))
        k = int(np.floor(br_x / (1.0 / grid_size[0])))
        l = int(np.floor(br_y / (1.0 / grid_size[1])))
        tl_x_trans = (tl_x * target_size - i * dscale) / dscale
        tl_y_trans = (tl_y * target_size - j * dscale) / dscale
        br_x_trans = (br_x * target_size - k * dscale) / dscale
        br_y_trans = (br_y * target_size - l * dscale) / dscale
        return [i, j, tl_x_trans, tl_y_trans, k, l, br_x_trans, br_y_trans]


if __name__ == '__main__':
    from random import shuffle
    import random
    train_dataset_folder = './data/RoadImages/train'
    val_dataset_folder = './data/RoadImages/val'
    dataset_root = './data/RoadImages'
    val_ratio = 0.2
    notation_file_path = os.path.join('./data/RoadImages', 'label.idl')
    val_label_name = 'val_label.idl'
    train_label_name = 'train_label.idl'
    # small_overfit_images = ["60091-1.jpg", "67996-1.jpg","68015-1.jpg", "70076-1.jpg", "68785-1.jpg", "68808-1.jpg",
    #                       "68040-1.jpg", "67999-1.jpg", "68192-1.jpg", "70070-1.jpg"]
    # data = Data(notation_file_path, dataset_folder, split_ratio=0.8)
    # data.generate_rec_file(image_list=small_overfit_images)
    notation_content = load_notation(notation_file_path)
    shuffle(notation_content)
    val_notation = []
    train_notation = []
    val_image_num = int(len(notation_content) * 0.2)  #26339
    for i in xrange(len(notation_content)):
        idx = random.randint(0, len(notation_content))
        print("random index {}".format(str(i)))
        if i < val_image_num:
            val_notation.append(notation_content[i])
            for key in notation_content[i].keys():
                path_from = os.path.join(train_dataset_folder, key)
                path_to = os.path.join(val_dataset_folder, key)
                os.rename(path_from, path_to)
        else:
            train_notation.append(notation_content[i])

    with open(os.path.join(dataset_root, val_label_name), 'w') as n:
        for item in val_notation:
            json_str = json.dumps(item)
            n.write(json_str + '\n')

    with open(os.path.join(dataset_root, train_label_name), 'w') as n:
        for item in train_notation:
            json_str = json.dumps(item)
            n.write(json_str + '\n')