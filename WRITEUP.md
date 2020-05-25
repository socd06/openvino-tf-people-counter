# Deploy a People Counter App at the Edge

## Inference

### Convert a Model into an Intermediate Representation with the Model Optimizer

Model chosen was the `SSD MobileNet V2 COCO` model since it is an object detection model trained in the COCO dataset which contains `People` as one of its classes. It's also one of the supported TensorFlow frozen topologies which means it is compatible with the OpenVINO model optimizer.

The `SSD MobileNet V2 COCO` model was downloaded from the [Converting a Tensorflow Model OpenVINO documentation](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
using the following command:

`wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz`

The model files were uncompressed using:
`tar -xvzf ssd_mobilenet_v2_coco_2018_03_29.tar.gz`

The model was converted to its intermediate representation using the following command:
`cd ssd_mobilenet_v2_coco_2018_03_29`
`python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_support.json`

Renamed *.xml* and *.bin* files to `tf_ssdm2.bin` and `tf_ssdm2.xml` respectively and put into the `models` folder.

### Loading Model Intermediate Representation into the Inference Engine
Model is loaded in the `load_model` function in `inference.py`.

### Custom Layers
Code checks and recommends using a CPU Extension. Refer to `load_model` function in `inference.py`.

### Handle Asynchronous Requests
Asynchronous requests are handled in the `exec_net` in `inference.py`.

### Output Results
The app can be ran using:

`python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/tf_ssdm2.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.3 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm`

Confidence threshold is set to 0.3 since the second person (in black) is not detected at a higher threshold and can cause false positives due to the verification code.

## Processing Video

### Handle Different Input Streams and Preprocess the Input

Input streams are resized, transposed and reshaped in `main.py` while the capture `isOpened()` and the user is notified if a given input is unsupported.

Processed input is then loaded to the inference network.

### Extract Information from Model Output

The output of the network is processed with bounding boxes when the probability threshold is met or exceeded. See `ssd_out` function in `main.py` for details.

### Calculate Relevant Statistics

Detections are verified along 2 seconds, yielding a total of 6 people detected and an average duration of 27 seconds of screen per person.
See [video-demo.m4v](https://github.com/socd06/openvino-tf-people-counter/blob/master/video-demo.m4v) or run app to verify results yourself.

## Sending Data to Servers

### Statistics Sent to MQTT Server

Port 3001 was used in the classroom workspace to publish the `person/duration` and `person` topics and sent through JSON to the MQTT server.
See [video-demo.m4v](https://github.com/socd06/openvino-tf-people-counter/blob/master/video-demo.m4v) or run app to verify results yourself.

### Image Frame Sent to FFmpeg Server

Frames were sent and flushed to the FFmpeg server. See [video-demo.m4v](https://github.com/socd06/openvino-tf-people-counter/blob/master/video-demo.m4v) or run app to verify results yourself.

### Video and Statistics Viewable in UI

Provided UI used and video and statistics were properly visualized.

## Write-Up

## Explaining Custom Layers

No custom layers were used.

## Comparing Model Performance

Model performance was not discussed during the course though [Optimizing neural networks for production with Intel’s OpenVINO](https://medium.com/hackernoon/optimizing-neural-networks-for-production-with-intels-openvino-a7ee3a6883d) makes a head-to-head comparison between vanilla Tensorflow models and Tensorflow models optimized using the `OpenVINO Model Optimizer`, with results showing that the `Model Optimizer` can increase inference speed from 30 to 50%.

An OpenVINO edge app is the best option for a retail application since it would not incur in recurrent cloud service charges. Though it may represent a high initial overhead, depending on the application, an OpenVINO app can be deployed in bare-bones scenarios or embedded devices, much cheaper than other all-in-one or embedded plus cloud systems.  

## Assess Model Use Cases

Some of the potential use cases of the people counter app are retail stores and malls, where the amount of customers could be tracked per hour to help determine better shop staffing and hours.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. For this example, the person with the darkest skin and outfit was not always detected by the detection model, showing there is a bias in the `person` class of the COCO dataset. A dedicated people counter app would need a dedicated human detection model where the bias is accounted for and minimized.
