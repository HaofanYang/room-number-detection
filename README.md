### Students
1. Haofan Yang
2. Xin Shen
3. Yumeng Zhu

## Introduction
This project involves room number recognitions in videos by indicating locations of room numbers appearing in a frame as well as specifying the room number being detected. The **[SSD (Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325)** model was utilized to localize door plates in raw videos or images and **[Google Cloud Vision APIs](https://cloud.google.com/vision/docs/libraries?hl=zh-cn)** were used to recognize room numbers appearing in bounding boxes produced by *SSD*. 

## How to run
Two examples were provided within `results` of which each includes a original raw video file, processed output video file and a csv file that contains text coordinates in corresponding frames. Before you run `main.py`, please make sure required packges have been installed
1. **Environment and Dependency Overview**
	1. Google Cloud Vision APIs
	2. OpenCV
	3. Tensorflow
	4. Python 3
	5. Numpy

2. Since this project uses **Google Cloud Vision APIs**, authentication is required to run the client library

	`$ export GOOGLE_APPLICATION_CREDENTIALS=$(pwd)/COSI-149-e0440f419720.json` 

3. Install Google Cloud Vision client library

	`$ pip install --upgrade google-cloud-vision`

4. Run the python script

	`$ python main.py --input_video={input_video_path} --output_video={output_video_path} --csv={csv_file_path}`

	Parameters: `{input_video_path}` is the path to your raw video and resulting output video will be saved to `{output_video_path}`. `{csv_file_path}` contains detection results for the input video.

## Training
### SSD model
1. **Architecture of SSD**

	Liu et al. proposed a model *SSD* for detecting objects in images and videos using a single deep neural network. It implements a multipile feature maps at the top followed by multi-scale convolutional bounding box outputs, which is the key feature of this model (**Figure 1**). As reported by authors, there is a significant improvement in the detection speed and accuracy, compared against current leading models such as *YOLO* and *Faster R-CNN*. For more details, please refer to the [original paper](https://arxiv.org/abs/1512.02325).
							
	![SSD Architecture](/images/SSD_1.png)

<center>**Figure 1.** The architecture of SSD.</center>

2. A **SSD mobile v2** model pre-trained with the **[COCO dataset](http://cocodataset.org/#home)** was downloaded from [this site](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and this pre-trained model was used as the starting point for the training step.

3. Training parameters (e.g. Learning rate, number steps & epoches, number of classes)
	* Learning Rate: 0.004 (0 - 10000 steps) and 0.001 (10000 - 20000 steps)
	* Number of steps: 20000
	* Number of classes: 1
	* Batch size: 16

### Training and test sets
483 images containing door plates were captured in the volen building at Brandeis Univeristy and used as the dataset for this project. The training and test set were split into 8:2 ratio (386 images in the training set and 97 images in the test set). An image annotation tool **[LabelImg](https://github.com/tzutalin/labelImg)** was used to lable images.

### Traning result
The localization loss on the training set decreased as the number of steps increased and finally converged at 0.075 (**Figure 2**) whereas the loss on the test set converged at 0.453 (**Figure 3**)
							
![Loss curve on the training set](/images/loss_1.png)

_**Figure 2.** Loss curve on the training set._
							
![Loss on the test set](/images/loss_1.png)

 **Figure 3.** Loss on the test set. 

## Summary
A pre-trained **SSD mobile v2** model was re-trained on 386 images captured in the volen building at Brandeis University, giving loss functions at 0.075 and 0.453 one the training and test set, respectively. This model successfully localized bounding boxes for door plates appearing in raw videos (prediction results are placed in the `Results` file) and room numbers within each bounding box were recognized using **Google Cloud Vision APIs**.
