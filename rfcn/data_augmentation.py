import os
import io
import json
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

class DataAugmentation:
    def __init__(self, notation_file, training_folder, wild_testing_folder, new_train_dataset):
        self.notation_file = notation_file
        self.training_folder = training_folder
        self.wild_testing_folder = wild_testing_folder
        self.notation_list = []
        self.new_notation_list = []
        self.new_train_set_path = new_train_dataset
        if not os.path.isdir(self.new_train_set_path):
            os.mkdir(self.new_train_set_path)
            print("Creating new image set under {}".format(self.new_train_set_path))

    def load_notations(self):
        with io.open(self.notation_file,'r') as n:
            notation_content = n.readlines()

        for img_entry in notation_content:
            self.notation_list.append(json.loads(img_entry))

        self.notation_list = self.screen_bad_label(self.notation_list)

    def save_notations(self):
        with open(os.path.join(self.new_train_set_path, 'new_label.idl'), 'w') as n:
            for item in self.new_notation_list:
                json_str = json.dumps(item)
                n.write(json_str + '\n')
        print("New Label Notation has been saved!")

    def image_transform_mirror(self, image, org_coordinate, image_size):
        mirror_img = ImageOps.mirror(image)
        new_coordinate = self.coordinate_transform(org_coordinate, image_size)
        return mirror_img, new_coordinate

    def coordinate_transform(self, org_cords, image_size):
        """
        Transform the notation coordinate from original to mirror image
        :param image_size: original image size, Format {'width': 640, 'height':360}
        :param org_cords: Format of coordinate [[top_left_x, top_left_y, bottom_right_x, bottom_right_y, category],]
        :return: Transferred coordinate with same Format as input
        """
        new_cords = []
        for cord in org_cords:
            top_left_x = image_size['width'] - cord[2]
            top_left_y = cord[1]
            bottom_right_x = image_size['width'] - cord[0]
            bottom_right_y = cord[3]
            new_cords.append([top_left_x, top_left_y, bottom_right_x, bottom_right_y, cord[4]])
        return new_cords


    def image_transform_contrast(self, image):
        """
        This method will generate image with two contrasted image, enhance rate is (2, 0.5)
        :param image: image loaded by PIL.Image.open
        :return: list of images which in numpy array format
        """
        contrast_images = []
        contrast = ImageEnhance.Contrast(image)
        contrast_images.append(contrast.enhance(2))
        contrast_images.append(contrast.enhance(0.5))
        return contrast_images

    def image_transform_brightness(self, image):
        """
        This method will generate image with 2 brighten pictures enhance factor (2, 0.3)
        :param image: image which loaded by PIL.Image.open
        :return: list of images
        """
        brighten_images = []
        brightness = ImageEnhance.Brightness(image)
        brighten_images.append(brightness.enhance(2))
        brighten_images.append(brightness.enhance(0.5))
        return brighten_images


    def image_transform_blur(self, image):
        """
        generate two blur images for training set with Gaussian and MedianFilter
        :type image: image which loaded by PIL.Image.open
        :return: 2 blur images
        """
        blur_images = [image.filter(ImageFilter.GaussianBlur(1)),
                       image.filter(ImageFilter.MedianFilter(3  ))]
        return blur_images


    def screen_bad_label(self, notation_list):
        """
        drop the labels which doesn't include any object notation
        :param notation_list: list of images and notation list[dict]
        :return: new label list without empty labels
        """
        print("There is {} records before screen the empty notation".format(len(notation_list)))
        i = 0
        new_notation_list = []
        for image_detail in notation_list:
            '''
            Data format: {u'69139.jpg': [[419.66656, 133.99992, 532.33344, 248.00003999999998, 1], 
                                         [232.49984, 161.66664, 368.0, 257.83344, 1]]}
            '''
            for key in image_detail.keys():
                if len(image_detail[key]) != 0:
                    new_notation_list.append(image_detail)
                    i += 1
        print("{} empty notations have been removed from list. {} useful data left."
              .format(str(len(notation_list) - len(new_notation_list)), len(new_notation_list)))
        return new_notation_list


    def new_training_set(self):
        # Image size for this project is 640*360
        image_size = {'width': 640, 'height': 360}
        for image_detail in self.notation_list:
            image_path = None
            notation = None
            image_name =None
            for key in image_detail.keys():
                image_name = key
                notation = image_detail[key]
            image_path = os.path.join(self.training_folder, image_name)
            print("Loading image {}".format(image_path))
            original_image = Image.open(image_path)
            mirror_image, new_notation = self.image_transform_mirror(original_image, notation, image_size)
            contrast_images = self.image_transform_contrast(original_image)
            contrast_mirror_images = self.image_transform_contrast(mirror_image)
            blur_images = self.image_transform_blur(original_image)
            blur_mirror_images = self.image_transform_blur(mirror_image)
            brightness_images = self.image_transform_brightness(original_image)
            brightness_mirror_images = self.image_transform_brightness(mirror_image)

            image_list = []

            image_list.append([original_image, notation])

            for contrast_image in contrast_images:
                image_list.append([contrast_image, notation])

            for blur_image in blur_images:
                image_list.append([blur_image, notation])

            for brightness_image in brightness_images:
                image_list.append([brightness_image, notation])

            image_list.append([mirror_image, new_notation])

            for contrast_mirror_image in contrast_mirror_images:
                image_list.append([contrast_mirror_image, new_notation])

            for blur_mirror_image in blur_mirror_images:
                image_list.append([blur_mirror_image, new_notation])

            for brightness_mirror_image in brightness_mirror_images:
                image_list.append([brightness_mirror_image, new_notation])

            self.save_image(image_list, image_name.split('.')[0])

        print(len(self.new_notation_list))
        self.save_notations()


    def save_image(self, images, image_prefix):
        i = 1
        for entry in images:
            image_name = image_prefix + "-" + str(i) + ".jpg"
            image_path = os.path.join(self.new_train_set_path, image_name)
            entry[0].save(image_path)
            self.new_notation_list.append({image_name: entry[1]})
            i += 1



if __name__ == '__main__':
    # Create a larger dataset and save the new label notation file

    new_train_dataset = '../dataset/new_train_set/'
    train_folder = '../dataset/training/'
    test_folder = '../dataset/testing/'
    notation_file_name = 'label.idl'
    new_notation_file_name = 'new_label.idl'
    # notation_file_path = os.path.join(train_folder,notation_file_name)
    # data = DataAugmentation(notation_file_path, train_folder, test_folder, new_train_dataset)
    # data.load_notations()
    # data.new_training_set()
    # print("Total {} images in new training dataset".format(len(data.new_notation_list)))


    # validate the new notation file if it is readable and correct
    new_notation_file_path = os.path.join(new_train_dataset, new_notation_file_name)
    new_data = DataAugmentation(new_notation_file_path, train_folder, test_folder, new_train_dataset)
    new_data.load_notations()
    print("Verify new training data notation, {} record inside!".format(len(new_data.notation_list)))
    i = 0
    for entry in new_data.notation_list:
        if i < 14:
            print(entry)
            i += 1
        else:
            break
