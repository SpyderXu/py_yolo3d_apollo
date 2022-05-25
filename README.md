# py_yolo3d_apollo
Python implementation of yolo3d in Apollo

Apollo provides a yolo3d detector which could do below tasks simultaneously:
* **2D Object Detection** : The pixel position of the objects.
* **Orientation Estimation** : Regress the local object orientation.
* **3D Dims Regression** : Regress the object 3d dims.
* **Lane Segmentation** : Lane line segmentation.

With the help of above information and camera intristric parameters, the 3d position of the objects could be estimated. I strip this part code from apollo and change the C++ code to python.

To run this code, Caffe should be installed and the python interface should be compiled. The camera intristric parameters in config.py should be edited. The caffe model and deploy file should be supplied. Corresponding path in config.py should also be edited. You could edit the test.py or test_video.py to run this code on your own image or video.

I have compared the outputs of the python code and original c++ code to ensure that they are same.

I give a short introduction for each part of this project here:
* **D2toD3** : Provide the camera model and the code for searching the 3D position.
* **yolo3d** : Encapsulates the Yolo3D camera detector.

You can obtain the model weights and deploy file from the Apollo project.

More details could be found on [Apollo](https://github.com/ApolloAuto/apollo).

