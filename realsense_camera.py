#https://pysource.com
import pyrealsense2 as rs
import numpy as np
import cv2

class RealsenseCamera:
def imDetect(self):

		frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()

        # Load imgae, grayscale, Gaussian blur, Otsu's threshold
		#image = cv2.imread('images/Test2.png')
		cam = cv2.VideoCapture(0)
		image = aligned_frames.get_color_frame()
        original = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9,9), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
		
        # Morph close
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours and filter for QR code
        cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            x,y,w,h = cv2.boundingRect(approx)
            area = cv2.contourArea(c)
            ar = w / float(h)
            if len(approx) == 4 and area > 1000 and (ar > .85 and ar < 1.3):
                cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 3)
                ROI = original[y:y+h, x:x+w]
                cv2.imwrite('ROI.png', ROI)
				float obDepth = depth.get_distance(x/2,y/2)
		print(obDepth)
        cv2.imshow('thresh', thresh)
        cv2.imshow('close', close)
        cv2.imshow('image', image)
        cv2.imshow('ROI', ROI)
        cv2.waitKey()     

    def __init__(self):
        # Configure depth and color streams
        print("Loading Intel Realsense Camera")
        self.pipeline = rs.pipeline()

        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)

        # Start streaming
        self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)


    def get_frame_stream(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            # If there is no frame, probably camera not connected, return False
            print("Error, impossible to get the frame, make sure that the Intel Realsense camera is correctly connected")
            return False, None, None
        
        # Apply filter to fill the Holes in the depth image
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.holes_fill, 3)
        filtered_depth = spatial.process(depth_frame)

        hole_filling = rs.hole_filling_filter()
        filled_depth = hole_filling.process(filtered_depth)

        
        # Create colormap to show the depth of the Objects
        colorizer = rs.colorizer()
        depth_colormap = np.asanyarray(colorizer.colorize(filled_depth).get_data())

        
        # Convert images to numpy arrays
        # distance = depth_frame.get_distance(int(50),int(50))
        # print("distance", distance)
        depth_image = np.asanyarray(filled_depth.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # cv2.imshow("Colormap", depth_colormap)
        # cv2.imshow("depth img", depth_image)

        return True, color_image, depth_image
    
    def release(self):
        self.pipeline.stop()
        #print(depth_image)
        
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.10), 2)

        # Stack both images horizontally
        
        #images = np.hstack((color_image, depth_colormap))




		def imDetect(image):
        # Load imgae, grayscale, Gaussian blur, Otsu's threshold
        original = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9,9), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Morph close
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours and filter for QR code
        cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            x,y,w,h = cv2.boundingRect(approx)
            area = cv2.contourArea(c)
            ar = w / float(h)
            if len(approx) == 4 and area > 1000 and (ar > .85 and ar < 1.3):
                cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 3)
                ROI = original[y:y+h, x:x+w]
                cv2.imwrite('ROI.png', ROI)

        cv2.imshow('thresh', thresh)
        cv2.imshow('close', close)
        cv2.imshow('image', image)
        cv2.imshow('ROI', ROI)
        cv2.waitKey()     