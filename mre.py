#code for duplicate uncompatible openvino_2021.4
import cv2
import depthai as dai
import numpy as np
import time

nnflag=True #True: turn on NN, False: turn off NN

# import blobconverter

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
    # pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2020_4)
    pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)
    # pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2022_1)
    # Define sources and outputs
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)

    rect_left = pipeline.create(dai.node.XLinkOut)
    rect_right = pipeline.create(dai.node.XLinkOut)
    mre = pipeline.create(dai.node.XLinkOut)

    rect_left.setStreamName("rect_left")
    rect_right.setStreamName("rect_right")
    mre.setStreamName("mre")

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
    stereo.disparity.link(mre.input)
    
    if nnflag:
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

def depthRelative(pred):
    # Scale depth to get relative depth
    d_min = np.min(pred)
    d_max = np.max(pred)
    depth_relative = (pred - d_min) / (d_max - d_min)

    # Color it
    depth_relative = np.array(depth_relative) * 255
    depth_relative = depth_relative.astype(np.uint8)
    depth_relative = 255 - depth_relative
    depth_relative = cv2.applyColorMap(depth_relative, cv2.COLORMAP_INFERNO)
    return depth_relative
    

# Model options (not all options supported together)
iters = 2            # Lower iterations are faster, but will lower detail. 
                     # Options: 2, 5, 10, 20 

                           # Options: (240,320), (320,480), (380, 480), (360, 640), (480,640), (720, 1280)

version = "init" # The combined version does 2 passes, one to get an initial estimation and a second one to refine it.
                     # Options: "init", "combined"

# Camera options: baseline (m), focal length (pixel) and max distance for OAK-D Lite
# Ref: https://docs.luxonis.com/en/latest/pages/faq/#how-do-i-calculate-depth-from-mre
# TODO: Modify values corrsponding with YOUR BOARD info

max_distance = 3

# Initialize model
# model_path = f'models/crestereo_{version}_iter{iters}_{input_shape[0]}x{input_shape[1]}.onnx'
# depth_estimator = CREStereo(model_path, camera_config=camera_config, max_dist=max_distance)

# Get Depthai pipeline
pipeline = create_pipeline(input_shape)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    rectified_left_queue = device.getOutputQueue(name="rect_left", maxSize=4, blocking=False)
    rectified_right_queue = device.getOutputQueue(name="rect_right", maxSize=4, blocking=False)
    mre_que = device.getOutputQueue(name="mre", maxSize=4, blocking=False)
    
    if nnflag:
        nn_que = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    start_time = time.time()
    counter = 0
    fps=0
    while True:
        in_left_rect = rectified_left_queue.get()
        in_right_rect = rectified_right_queue.get()
        oak_mre = mre_que.get()
        
        if nnflag:
            mre_map = nn_que.get()
            
        left_rect_img = in_left_rect.getCvFrame()
        right_rect_img = in_right_rect.getCvFrame()
        
        # Show FPS
        frame = left_rect_img
        color_black, color_white = (0, 0, 0), (255, 255, 255)
        label_fps = "Fps: {:.2f}".format(fps)
        (w1, h1), _ = cv2.getTextSize(label_fps, cv2.FONT_HERSHEY_TRIPLEX, 0.4, 1)
        cv2.rectangle(frame, (0, frame.shape[0] - h1 - 6), (w1 + 2, frame.shape[0]), color_white, -1)
        cv2.putText(frame, label_fps, (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX,
                    0.4, color_black)

            
        combined_image = np.hstack((frame, right_rect_img))
        #cv2.imwrite("output.jpg", combined_image)

        cv2.imshow("left/right img", combined_image)
        
        counter += 1
        if (time.time() - start_time) > 1:
            fps = counter / (time.time() - start_time)

            counter = 0
            start_time = time.time()

        if cv2.waitKey(1) == ord('q'):
            break