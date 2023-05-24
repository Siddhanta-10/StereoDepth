import cv2
import threading
import time
import numpy as np
from PIL import Image

# camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('stereoMap70_gpu_320def.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
print(type(stereoMapL_x))

# create gpuMat for stereoMaps
cuMapLX = cv2.cuda_GpuMat(cv2.CV_32FC1)
cuMapLY = cv2.cuda_GpuMat(cv2.CV_32FC1)
cuMapRX = cv2.cuda_GpuMat(cv2.CV_32FC1)
cuMapRY = cv2.cuda_GpuMat(cv2.CV_32FC1)

# upload cumaps to gpu
cuMapLX.upload(stereoMapL_x)
cuMapLY.upload(stereoMapL_y)
cuMapRX.upload(stereoMapR_x)
cuMapRY.upload(stereoMapR_y)





#B = 3.95               #Distance between the cameras [cm]
B = 9.5               #Distance between the cameras [cm]
f = 0.304               #Camera lense's focal length [mm]
theta0 = 62.2        #Camera field of view in the horisontal plane [degrees]

# Create StereoSGBM and prepare all parameters
num_disparities = 16
block_size = 3
bp_ndisp = 64
min_disparity = 2
uniqueness_ratio = 30
stereo_sgm_cuda = cv2.cuda.createStereoBM(numDisparities=num_disparities,
                                           blockSize=block_size
                                           )



# WLS FILTER Parameters
lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0
 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_sgm_cuda)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)


# Mouse pointer location
point = (300, 200)

def onMouse(event, x, y, flags, disparity_normalized):
    global point
    point = (x, y)
    distance = disparity_normalized[y][x]
    if distance==0:
        zDepth = 0
    else:
        zDepth = (B * 1280)/(2 * np.tan(theta0/2) * distance)
    print("Distance {} mm".format(zDepth*10))

# Create mouse event
#cv2.namedWindow("Left Image")
#cv2.namedWindow("Right Image")
#cv2.setMouseCallback("Left Image", show_distance)
#cv2.setMouseCallback("Right Image", show_distance)
cv2.namedWindow("DepthMap")


def find_depth(dispL):
    x0 = dispL.shape[1]
    # CALCULATE DEPTH z:
    zDepth = (B * x0)/(2 * np.tan(theta0/2) * dispL)             #Depth in [cm]
    #f_pixel = (640 * 0.5) / np.tan(theta0 * 0.5 * np.pi/180)
    #zDepth = (B * f_pixel)/(dispL)
    #print(type(zDepth))
    return zDepth


class StereoWrapper:
    """
    This class takes care of the CUDA input such that such that images
    can be provided as numpy array
    """
    def __init__(self,
                 num_disparities: int = 64,
                 block_size: int = 5,
                 bp_ndisp: int = 64,
                 min_disparity: int = 2,
                 uniqueness_ratio: int = 5
                 ) -> None:
        self.stereo_bm_cuda = cv2.cuda.createStereoBM(numDisparities=num_disparities,
                                                  blockSize=block_size)
        self.stereo_bp_cuda = cv2.cuda.createStereoBeliefPropagation(ndisp=bp_ndisp)
        self.stereo_bcp_cuda = cv2.cuda.createStereoConstantSpaceBP(min_disparity)
        self.stereo_sgm_cuda = cv2.cuda.createStereoSGM(minDisparity=min_disparity,
                                                    numDisparities=num_disparities,
                                                    uniquenessRatio=uniqueness_ratio
                                                    )
    @staticmethod
    def __numpy_to_gpumat(np_image: np.ndarray) -> cv2.cuda_GpuMat:
        """
        This method converts the numpy image matrix to a matrix that
        can be used by opencv cuda.
        Args:
            np_image: the numpy image matrix
        Returns:
            The image as a cuda matrix
        """
        image_cuda = cv2.cuda_GpuMat(cv2.CV_32FC1)
        image_cuda.upload(cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY))
        return image_cuda

    def compute_disparity(self, left_img: np.ndarray,
                          right_img: np.ndarray,
                          algorithm_name: str = "stereo_bm_cuda"
                          ) -> np.ndarray:
        """
        Computes the disparity map using the named algorithm.
        Args:
            left_img: the numpy image matrix for the left camera
            right_img: the numpy image matrix for the right camera
            algorithm_name: the algorithm to use for calculating the disparity map
        Returns:
            The disparity map
        """
        algorithm = getattr(self, algorithm_name)
        left_cuda = self.__numpy_to_gpumat(left_img)
        right_cuda = self.__numpy_to_gpumat(right_img)
        if algorithm_name == "stereo_sgm_cuda":
            disparity_sgm_cuda_2 = cv2.cuda_GpuMat()
            disparity_sgm_cuda_1 = algorithm.compute(left_cuda,
                                                     right_cuda,
                                                     disparity_sgm_cuda_2)
            return disparity_sgm_cuda_1.download()
        else:
            disparity_cuda = algorithm.compute(left_cuda, right_cuda, cv2.cuda_Stream.Null())
            return disparity_cuda.download()



wrapper = StereoWrapper()




class CSI_Camera:

    def __init__(self):
        # Initialize instance variables
        # OpenCV video capture element
        self.video_capture = None
        # The last captured image from the camera
        self.frame = None
        self.grabbed = False
        # The thread where the video capture runs
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False

    def open(self, gstreamer_pipeline_string):
        try:
            self.video_capture = cv2.VideoCapture(
                gstreamer_pipeline_string, cv2.CAP_GSTREAMER
            )
            # Grab the first frame to start the video capturing
            self.grabbed, self.frame = self.video_capture.read()

        except RuntimeError:
            self.video_capture = None
            print("Unable to open camera")
            print("Pipeline: " + gstreamer_pipeline_string)


    def start(self):
        if self.running:
            print('Video capturing is already running')
            return None
        # create a thread to read the camera image
        if self.video_capture != None:
            self.running = True
            self.read_thread = threading.Thread(target=self.updateCamera)
            self.read_thread.start()
        return self

    def stop(self):
        self.running = False
        # Kill the thread
        self.read_thread.join()
        self.read_thread = None

    def updateCamera(self):
        # This is the thread to read images from the camera
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except RuntimeError:
                print("Could not read image from camera")
        # FIX ME - stop and cleanup thread
        # Something bad happened

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture != None:
            self.video_capture.release()
            self.video_capture = None
        # Now kill the thread
        if self.read_thread != None:
            self.read_thread.join()


""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080
"""


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=2,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def run_cameras():
    window_title = "Dual CSI Cameras"
    left_camera = CSI_Camera()
    left_camera.open(
        gstreamer_pipeline(
            sensor_id=0,
            capture_width=1280,
            capture_height=720,
            flip_method=2,
            display_width=640,
            display_height=480,
        )
    )
    left_camera.start()

    right_camera = CSI_Camera()
    right_camera.open(
        gstreamer_pipeline(
            sensor_id=1,
            capture_width=1280,
            capture_height=720,
            flip_method=2,
            display_width=640,
            display_height=480,
        )
    )
    right_camera.start()

    if left_camera.video_capture.isOpened() and right_camera.video_capture.isOpened():

        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
        prev_time = 0
        try:
            while True:
                ret_right, frame_left = left_camera.read()
                ret_left, frame_right = right_camera.read()


                #print(frame_left.shape)

                gpu_frame_left = cv2.cuda_GpuMat(cv2.CV_32FC1)
                gpu_frame_right = cv2.cuda_GpuMat(cv2.CV_32FC1)

                
                
                gpu_frame_left.upload(frame_left)
                gpu_frame_right.upload(frame_right)


                #gpu_frame_left = cv2.cuda.cvtColor(gpu_frame_left, cv2.COLOR_BGR2GRAY)
                #gpu_frame_right = cv2.cuda.cvtColor(gpu_frame_right, cv2.COLOR_BGR2GRAY)

                



                left_cuda = cv2.cuda.remap(gpu_frame_left, cuMapLX, cuMapLY, interpolation=cv2.INTER_LINEAR)
                right_cuda = cv2.cuda.remap(gpu_frame_right, cuMapRX, cuMapRY, interpolation=cv2.INTER_LINEAR)

                left_numpy_image = left_cuda.download()
                right_numpy_image = right_cuda.download()

                # Compute the 2 images for the Depth_image
                disparity= wrapper.compute_disparity(left_numpy_image,right_numpy_image)#.astype(np.float32)/ 16
                #print(disparity.dtype) # uint8


                disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
                image = np.array(disparity_normalized, dtype=np.uint8)
                disparity_color = cv2.applyColorMap(image, cv2.COLORMAP_JET)
                #print(disparity_normalized.dtype) # uint8
                cv2.setMouseCallback("DepthMap", onMouse, disparity_normalized)
                cv2.imshow("DepthMap", image)
                cv2.imshow("DepthMapColored", disparity_color)
                

                curr_time = time.time()
                diff = curr_time - prev_time
                if diff <= 0:
                    fps = 0
                else:
                    fps = 1/diff
                    
                cv2.putText(left_numpy_image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
                cv2.putText(right_numpy_image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
                prev_time = curr_time



                # Show distance for a specific point
                #cv2.circle(left_numpy_image, point, 4, (0, 0, 255))
                #cv2.circle(right_numpy_image, point, 4, (0, 0, 255))

                #depth_frame = find_depth(disparity)
                #distance = abs(depth_frame[point[0]][point[1]])
                #cv2.imshow('Disparity Map', disparity)


                #cv2.putText(left_numpy_image, "{}mm".format(distance), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
                #cv2.putText(right_numpy_image, "{}mm".format(distance), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

                # Mouse click
                #cv2.setMouseCallback("Filtered Color Depth",find_depth,dispL)

                
                
                
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow("Left_image", left_numpy_image)
                    cv2.imshow("Right_image", right_numpy_image)
                else:
                    break
                # This also acts as
                keyCode = cv2.waitKey(1) & 0xFF
                # Stop the program on the ESC key
                if keyCode == 27:
                    break
        finally:

            left_camera.stop()
            left_camera.release()
            right_camera.stop()
            right_camera.release()
        cv2.destroyAllWindows()
    else:
        print("Error: Unable to open both cameras")
        left_camera.stop()
        left_camera.release()
        right_camera.stop()
        right_camera.release()



if __name__ == "__main__":
    run_cameras()
