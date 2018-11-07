import pyrealsense2 as rs
import cv2
import numpy as np
import glob
import math
import os, operator
from random import randint
from scipy import spatial

output_landmarks = []
refPt = []

#Gives coordinates on mouse click in a opencv window
def click_coordinates(event, x, y, flags, param):
    global refPt

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x,y))
    elif event == cv2.EVENT_LBUTTONUP:
        print(refPt)
        print(len(refPt))

#save landmarks to file
def save_landmarks(path, coords):
    with open(path, "w+") as file:
        for coord in coords:
            file.write("%i," % coord[0])
            file.write("%i\n" % coord[1])

#Find face in a image
def find_face(frame_image):
    face_cascade = cv2.CascadeClassifier('frontal_face_features.xml')
    gray_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
    gray_image -= np.min(gray_image[:])
    gray_image /= np.max(gray_image[:])
    gray_image_uint = gray_image * 255
    gray_image_uint = gray_image_uint.astype(np.uint8)
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
    return faces

#returns a 300x300 image of the face bounding box
def resampling(frame, faces):
    for (x, y, w, h) in faces:
        ROI = frame[y:y+h, x:x+w]
        resize = cv2.resize(ROI, (300,300), fx=1, fy=1, interpolation = cv2.INTER_CUBIC)
    return resize

#Draws white circles on a image according to coordinates input
def draw_landmarks(img, coords):
    for (i,j) in coords:
        cv2.circle(img,(i,j), 5, (255,225,255), 1)
    return img

#Computes SIFT descriptors from a set of landmarks
def keypoints(image, landmarks):
    descriptor = cv2.xfeatures2d.SIFT_create()
    keypoints = []

    for point in landmarks:
        # convert the x, y point into an open-cv key point, 10 is the size of the key point
        point_converted_to_key_point = cv2.KeyPoint(point[0], point[1], 10)
        keypoints.append(point_converted_to_key_point)

    # Compute keypoint descriptors
    kp, key_point_descriptor = descriptor.compute(image, keypoints)

    return kp, key_point_descriptor

#Read a .txt file with image landmark positions
def get_image_landmarks(path):
    landmarks = []
    base = os.path.splitext(path)[0]
    path = path.replace('.png', '.txt')
    with open(path, "r") as file:
        for line in file:
            x, y = line.split(',')
            landmarks.append((int(x),int(y)))
    return landmarks

#Scale landmarks from 300x300 image to original image size
def scale_landmarks(landmarks, faces):
    scaled_landmarks = []
    for (x, y, w, h) in faces:
        scale_x = w/(300 * 1.0)
        scale_y = h/(300 * 1.0)
        for landmark in landmarks:
            re_x, re_y = landmark
            re_x = (re_x * scale_x) + x
            re_y = (re_y * scale_y) + y
            scaled_landmarks.append((int(re_x),int(re_y)))
    return scaled_landmarks

#Compares descriptors of frame image and template image
def compare_random_keypoints(fkp, fd, td):
    tree = spatial.KDTree(fd)
    closest = tree.query(td)
    pos = closest[1]
    return (int(fkp[pos].pt[0]),int(fkp[pos].pt[1]))

#Generate 22 random points in a square around each landmark
def generate_random_points(landmarks):
    random_landmarks = []
    dist = 5 #size of square around each landmark for random sampling
    for point in landmarks:
        random_points = []
        for i in range(22):
            rand_x = randint(point[0]-dist, point[0]+dist)
            rand_y = randint(point[1]-dist, point[1]+dist)
            random_points.append((rand_x, rand_y))
        random_landmarks.append(random_points)
    return random_landmarks

#Get the average distance error for landmark positions
def accuracy(final_landmarks, ground_truth_landmarks):
    distances = []
    for i in range(len(final_landmarks)):
        x1, y1 = ground_truth_landmarks[i]
        x2, y2 = final_landmarks[i]
        distances.append(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )/len(final_landmarks))
    return sum(distances)/len(distances)

#Return average distance error on our dataset of still images
def accuracy_test():
    # the trained haar-cascade classifier data
    face_cascade = cv2.CascadeClassifier('frontal_face_features.xml')

    #load template picture and find face
    paths_original  = glob.glob('emotion_dataset/dataset/*.png')
    path = 'emotion_dataset/Happy/happy1_Color.png'
    template_landmarks = get_image_landmarks(path)
    template = cv2.imread(path).astype(np.float32)
    template -= np.min(template[:])
    template /= np.max(template[:])
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_gray -= np.min(template_gray[:])
    template_gray /= np.max(template_gray[:])
    template_gray_uint = template_gray * 255
    template_gray_uint = template_gray_uint.astype(np.uint8)
    template_face = face_cascade.detectMultiScale(template_gray_uint, 1.3, 5)
    template_face[0,3] += 50 #increase the height of face bounding box

    #Random sample around template landmarks
    random_landmarks = generate_random_points(template_landmarks)

    #get template keypoints and descriptors
    if len(template_face) > 0:
        template_gray_uint = resampling(template_gray_uint, template_face)
        template_kp, template_descriptors = keypoints(template_gray_uint, template_landmarks)
        #cv2.imshow('template',template_gray_uint)
        #cv2.waitKey(0)

    landmark_accuracy = []

    for path_original in paths_original:
        print(path_original)
        # image to display
        image = cv2.imread(path_original).astype(np.float32)
        image -= np.min(image[:])
        image /= np.max(image[:])

        #Find face in current frame
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image -= np.min(gray_image[:])
        gray_image /= np.max(gray_image[:])
        gray_image_uint = gray_image * 255
        gray_image_uint = gray_image_uint.astype(np.uint8)
        faces = face_cascade.detectMultiScale(gray_image_uint, 1.3, 5)

        #get current image ground truth
        ground_truth_landmarks = scale_landmarks(get_image_landmarks(path_original), faces)

        if len(faces) > 0:
            faces[0,3] += 50 #increase height of face bounding box

            for (x,y,w,h) in faces:
                # draw a rectangle where a face is detected
                cv2.rectangle(image, (x,y),(x+w,y+h), (255,0,0), 2)

            #Resize gray frame image to a 300x300 image of the detected face
            gray_image_uint = resampling(gray_image_uint, faces)

            final_landmarks = []
            #Compare random sampled points descriptors to template point descriptors
            #Save the random sampled descriptors closest to the temmplate descriptors
            for i in range(22):
                r_landmarks = random_landmarks[i]
                curr_descriptor = template_descriptors[i]
                frame_kp, frame_descriptors = keypoints(gray_image_uint, r_landmarks)
                final_landmarks.append(compare_random_keypoints(frame_kp, frame_descriptors, curr_descriptor))

            #Scale landmark coordinates from 300x300 to original frame size
            final_landmarks = scale_landmarks(final_landmarks, faces)

            #Draw the landmarks on the face
            image = draw_landmarks(image, final_landmarks)

            #print(accuracy(final_landmarks, ground_truth_landmarks))
            landmark_accuracy.append(accuracy(final_landmarks, ground_truth_landmarks))
    print("Average distance error: ")
    print(sum(landmark_accuracy)/len(landmark_accuracy))

            # Show images
            #cv2.imshow('RealSense', image)
            #cv2.waitKey(0)

def main():
    image_size = (640,480)
    background = np.zeros((image_size[1],image_size[0],3))
    clipping_distance_in_meters = 0.15

    # the openCV window
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

    # the trained haar-cascade classifier data
    face_cascade = cv2.CascadeClassifier('frontal_face_features.xml')

    # configure the realsense camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, image_size[0], image_size[1], rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, image_size[0], image_size[1], rs.format.z16, 30)
    frame_aligner = rs.align(rs.stream.color)

    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    clipping_distance = clipping_distance_in_meters / depth_scale

    #load template picture and find face
    paths_original  = glob.glob('emotion_dataset/Happy/*.png')
    path = 'emotion_dataset/Happy/happy1_Color.png'
    landmarks = get_image_landmarks(path)
    template = cv2.imread(path).astype(np.float32)
    template -= np.min(template[:])
    template /= np.max(template[:])
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_gray -= np.min(template_gray[:])
    template_gray /= np.max(template_gray[:])
    template_gray_uint = template_gray * 255
    template_gray_uint = template_gray_uint.astype(np.uint8)
    template_face = face_cascade.detectMultiScale(template_gray_uint, 1.3, 5)
    template_face[0,3] += 50 #increase the height of face bounding box

    #get template keypoints and descriptors
    if len(template_face) > 0:
        template_gray_uint = resampling(template_gray_uint, template_face)
        template_kp, template_descriptors = keypoints(template_gray_uint, landmarks)
        template = cv2.drawKeypoints(template_gray_uint,template_kp, template, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #cv2.imshow('template', template)

    #Random sample around template landmarks
    random_landmarks = generate_random_points(landmarks)

    try:
        while True:
            # Wait for a new frame and align the frame
            frames = pipeline.wait_for_frames()
            aligned_frames = frame_aligner.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # image to display
            image = np.asanyarray(color_frame.get_data()).astype(np.float32)
            image -= np.min(image[:])
            image /= np.max(image[:])

            #Find face in current frame
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_image -= np.min(gray_image[:])
            gray_image /= np.max(gray_image[:])
            gray_image_uint = gray_image * 255
            gray_image_uint = gray_image_uint.astype(np.uint8)
            faces = face_cascade.detectMultiScale(gray_image_uint, 1.3, 5)

            if len(faces) > 0:
                faces[0,3] += 50 #increase height of face bounding box

                for (x,y,w,h) in faces:
                    # draw a rectangle where a face is detected
                    cv2.rectangle(image, (x,y),(x+w,y+h), (255,0,0), 2)

                #Resize gray frame image to a 300x300 image of the detected face
                gray_image_uint = resampling(gray_image_uint, faces)

                final_landmarks = []
                #Compare random sampled points descriptors to template point descriptors
                #Save the random sampled descriptors closest to the temmplate descriptors
                for i in range(22):
                    r_landmarks = random_landmarks[i]
                    curr_descriptor = template_descriptors[i]
                    frame_kp, frame_descriptors = keypoints(gray_image_uint, r_landmarks)
                    final_landmarks.append(compare_random_keypoints(frame_kp, frame_descriptors, curr_descriptor))

                #Scale landmark coordinates from 300x300 to original frame size
                final_landmarks = scale_landmarks(final_landmarks, faces)
                #write to file
                save_landmarks('current.txt', final_landmarks)

                #Draw the landmarks on the face
                image = draw_landmarks(image, final_landmarks)

            # Show images
            cv2.imshow('RealSense', image)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        print("\nShutting down -- Good Bye")
    finally:
        # Stop streaming
        pipeline.stop()

main()
#accuracy_test()
