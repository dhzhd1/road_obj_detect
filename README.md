# Road Object Detector 
This project is based on **Deformable Convolutional Network**  
This repository used code from [MXNet rcnn example](https://github.com/dmlc/mxnet/tree/master/example/rcnn) , [mx-rfcn](https://github.com/giorking/mx-rfcn) and [Deformable-ConvNets](https://github.com/msracver/Deformable-ConvNets).


## Introduction

**R-FCN** is initially described in a [NIPS 2016 paper](https://arxiv.org/abs/1605.06409).


## Requirements: Software

Please refer to [requirements.txt](https://github.com/dhzhd1/road_obj_detect/blob/master/rfcn/requirements.txt)


## Requirements: Hardware

Any NVIDIA GPUs with at least 4GB memory should be OK. This project I used Quadro P5000.

## Installation

1. Clone this repository to your system
```
git clone https://github.com/dhzhd1/road_obj_detect.git
```

2. For Windows users, run ``cmd .\init.bat``. For Linux user, run `sh ./init.sh`. The scripts will build cython module automatically and create some folders.

3. Install MXNet:

	3.1 Clone MXNet and checkout to [MXNet@(commit 62ecb60)](https://github.com/dmlc/mxnet/tree/62ecb60) by
	```
	git clone --recursive https://github.com/dmlc/mxnet.git
	git checkout -b 62ecb60
	git submodule update --remote --merge
	```
	3.2 Copy operators in `$(DCN_ROOT)/rfcn/operator_cxx` to `$(YOUR_MXNET_FOLDER)/src/operator/contrib` by
	```
	cp -r $(DCN_ROOT)/rfcn/operator_cxx/* $(MXNET_ROOT)/src/operator/contrib/
	```
	3.3 Compile MXNet
	```
	cd ${MXNET_ROOT}
	make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
	```
	3.4 Install the MXNet Python binding by
	
	```
	cd python
	sudo python setup.py install
	```

## Preparation for Training & Testing

For R-FCN/Faster R-CNN\:
1. Check out the code, run the code of [prepare_raw_data.py](https://github.com/dhzhd1/road_obj_detect/blob/master/rfcn/prepare_raw_data.py) . It will download the train and wild testing data and extracted into below folder:
```
	./rfcn/data/RoadImages/train
	./rfcn/data/RoadImages/test
```

2. Please download ImageNet-pretrained ResNet-v1-101 model manually from [OneDrive](https://1drv.ms/u/s!Am-5JzdW2XHzhqMEtxf1Ciym8uZ8sg), and put it under folder `./rfcn/model/pretrained_model`. Make sure it looks like this:
	```
	./rfcn/model/pretrained_model/resnet_v1_101-0000.params
	```
3. Run the [data_augmentation.py](https://github.com/dhzhd1/road_obj_detect/blob/master/rfcn/data_augmentation.py) to make more training sample. This file will create a mirror image. For the orignial image and mirrored image, will generate 6 differnt images by adjust the brightness x2, blur x2, constract x2. After augmentation, there are 131698 traing samples (remove the invalid training samples)

## Usage

1. All of our experiment settings (GPU #, dataset, etc.) are kept in yaml config files at folder `./experiments/rfcn/cfgs`. You also can use the yaml file which located in the ./rfcn/road_train_all.yaml.
2. After change the yaml configuration file properly, run the below command to start training:
```
	nohup python ./road_train.py --cfg road_train_all.yaml &
```
the output file will be stored under ./rfcn/output

3. After train stage finished, you can use the [inference.py](https://github.com/dhzhd1/road_obj_detect/blob/master/rfcn/inference.py) to do the wild testing. 


## Result
The project I was using a NVIDIA Quadro P5000 under CUDA 8.  The wild testing result is based on EPOCH 19. It took 4 days.

Result file was in JSON format. It could be found under ./rfcn/results.json. [results.json](https://github.com/dhzhd1/road_obj_detect/blob/master/rfcn/results.json)



## Citing Deformable ConvNets

If you find Deformable ConvNets useful in your research, please consider citing:
```
@article{dai17dcn,
    Author = {Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, Yichen Wei},
    Title = {Deformable Convolutional Networks},
    Journal = {arXiv preprint arXiv:1703.06211},
    Year = {2017}
}
@inproceedings{dai16rfcn,
    Author = {Jifeng Dai, Yi Li, Kaiming He, Jian Sun},
    Title = {{R-FCN}: Object Detection via Region-based Fully Convolutional Networks},
    Conference = {NIPS},
    Year = {2016}
}
```
## License

© Microsoft, 2017. Licensed under an Apache-2.0 license.
