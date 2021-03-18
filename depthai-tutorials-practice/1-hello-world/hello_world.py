import numpy as np # numpy - manipulate the packet data returned by depthai
import cv2 # opencv - display the video stream
import depthai # access the camera and its data packets

########## initialyze pipeline ################
pipeline = depthai.Pipeline()

# add color camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300) # define the size acordinggly to the neuralnetwork image size
cam_rgb.setInterleaved(False)

# define neuralnetwork
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath("/home/utilizador/OAK_code/depthai-tutorials-practice/mobilenet-ssd.blob")

# connect color camera tro neural network
cam_rgb.preview.link(detection_nn.input)

# compile device result into host machine, using xlinkout
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)
###########################################

############# Initialize DepthAI ##############
# USB3.0
device = depthai.Device(pipeline)
device.startPipeline()

# USB2.0 
# device = depthai.Device(pipeline, True)

# hostside outputs
q_rgb = device.getOutputQueue("rgb")
q_nn = device.getOutputQueue("nn")

# object boxes
frame = None
bboxes = []

# convert 0..1 values into pixel positions
def frame_norm(frame, bbox):
    return (np.array(bbox) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)

while True:
    # fetching node and colorcamera
    in_rgb = q_rgb.tryGet() # return the last result or None(when the queue is empty)
    in_nn = q_nn.tryGet()

    # convet 1D frames to 3D
    if in_rgb is not None:
        shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
        frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

    # filter last 4 agruments in nn vector plus set the confifence value to 0.8
    if in_nn is not None:
        bboxes = np.array(in_nn.getFirstLayerFp16())
        bboxes = bboxes[:np.where(bboxes == -1)[0][0]]
        bboxes = bboxes.reshape((bboxes.size // 7, 7))
        bboxes = bboxes[bboxes[:, 2] > 0.8][:, 3:7]

    # display results
    if frame is not None:
        for raw_bbox in bboxes:
            bbox = frame_norm(frame, raw_bbox)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.imshow("preview", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break