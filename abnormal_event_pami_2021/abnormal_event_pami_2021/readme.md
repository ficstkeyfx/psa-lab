## A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video
### Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu, Mubarak Shah
### IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021

### Required libraries
- tested with Python 3.6 and 3.7
- numpy (tested with version 1.18.5 and 1.19.1)
- tensorflow (tested with tf1.13, tf1.14 and tf1.15)
- opencv (tested with 4.5.1)
- tested on Linux OS and Windows OS 

### Required pre-trained models
- YoloV3 (https://github.com/wizyoung/YOLOv3_TensorFlow)
- MaskRCNN  (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md - mask_rcnn_inception_v2_coco)
- SelFlow (https://github.com/ppliuboy/SelFlow)

### Set the input and output paths in the ```args.py``` file.
    '''
    database_name = 'ShanghaiTech'
    output_folder_base = '/home/lili/datasets/abnormal_event/shanghai/output_yolo_0.80'
    input_folder_base = '/home/lili/datasets/abnormal_event/shanghai'
    adversarial_images_path = "/home/lili/datasets/adversarial_images"
    '''

### Run ```mask_extraction.py```

It requires the mask rcnn model in the folder models/mask_rcnn/frozen_inference_graph.pb

### Run ```object_extraction.py```
It requires the yolov3 model in the folder models/yolov3/yolov3.ckpt

### Run ```write_mask_on_objects.py```

### In order to extract the optical flow (the files are taken from SelFlow official repository):
- from ```optical-flow-sel-flow/args.py``` set the input and the output paths (the same as ```args.py```)
- from optical-flow-sel-flow directory, run ```main.py```, but
    - selflow_model.py line 182 and 183
      ```is_adv = False; folder_name = 'train'```
       by default it will extract the optical flow for training without adversarial.
       - first run ```main.py```
       - second change ```is_adv = True``` to extract the adversarial maps for training, then run ```main.py```.
       - third change ```is_adv = False; folder_name = 'test' ``` to extract the maps for test, then run ```main.py```.
    
### Run ```write_motion_on_objects.py```
- by default, it will write the motion crops for training.
- change ```is_adv = True (line 48)``` to write the adversarial motion crops for training
- change ```is_adv = False (line 48); folder_name = 'test'``` to write the motion crops for testing

### Run ```run.py```


#### Evaluation
- The evaluation code requires to have the ground-truth frame level annotation in args.output_folder_base/test/video_name/ground_truth_frame_level.txt
- The 2D and 3D filters (compute_features.py lines 196 and 180) are fine-tuned for the ShanghaiTech dataset.


