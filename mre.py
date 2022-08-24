#code for duplicate uncompatible openvino_2021.4
import cv2
import depthai as dai
import numpy as np
import time

import requests

url = "http://blobconverter.luxonis.com/compile"  # change if running against other URL
payload = {
    'compile_type': 'model',
    'model_type': 'onnx',
    'intermediate_compiler_params': '--data_type=FP16',
    'compiler_params': '-ip U8 -op FP16 -VPU_NUMBER_OF_SHAVES 6'
}
files = {
    'model': open('model.onnx', 'rb'),
}
params = {
    'version': '2021.4',  # OpenVINO version, can be "2021.1", "2020.4", "2020.3", "2020.2", "2020.1", "2019.R3"
}
response = requests.post(url, data=payload, files=files, params=params)
print(response)

import blobconverter

modelPath = "model.blob"; input_shape = (180,320)

blob = dai.OpenVINO.Blob(modelPath)
print("blob.version=",blob.version)
print('Inputs')
[print(f"Name: {name}, Type: {vec.dataType}, Shape: {vec.dims}") for name, vec in blob.networkInputs.items()]
print('Outputs')
[print(f"Name: {name}, Type: {vec.dataType}, Shape: {vec.dims}") for name, vec in blob.networkOutputs.items()]

def create_pipeline(input_shape):

    # Create pipeline
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

    # Define sources and outputs
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)

    rect_left = pipeline.create(dai.node.XLinkOut)
    rect_right = pipeline.create(dai.node.XLinkOut)

    rect_left.setStreamName("rect_left")
    rect_right.setStreamName("rect_right")

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # ImageManip for cropping (face detection NN requires input image of 300x300) and to change frame type
    manipLeft = pipeline.create(dai.node.ImageManip)
    manipLeft.initialConfig.setResize(input_shape[1],input_shape[0])
    # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
    manipLeft.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
    
    # ImageManip for cropping (face detection NN requires input image of 300x300) and to change frame type
    manipRight = pipeline.create(dai.node.ImageManip)
    manipRight.initialConfig.setResize(input_shape[1], input_shape[0])
    # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
    manipRight.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
    
    # StereoDepth
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    
    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    stereo.rectifiedLeft.link(manipLeft.inputImage)
    stereo.rectifiedRight.link(manipRight.inputImage)
    
    stereo.rectifiedLeft.link(rect_left.input)
    stereo.rectifiedRight.link(rect_right.input)

    # NN that detects faces in the image
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(modelPath)
    nn.setNumInferenceThreads(0)
           
    manipLeft.out.link(nn.inputs['left'])
    manipRight.out.link(nn.inputs['right'])
    
    # Send bouding box from the NN to the host via XLink
    nn_xout = pipeline.create(dai.node.XLinkOut)
    nn_xout.setStreamName("nn")
    nn.out.link(nn_xout.input)
    
    return pipeline
    
# Get Depthai pipeline
pipeline = create_pipeline(input_shape)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    nn_que = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    while True:
        mre_map = nn_que.get()
