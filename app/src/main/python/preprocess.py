import numpy as np
import cv2
from PIL import Image
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import json
import os
import sys
from os.path import dirname, join
# from com.arthenica.mobileffmpeg import FFmpeg
# import ffmpeg
#from keras.models import model_from_json
from com.arthenica.mobileffmpeg import FFmpeg
from com.arthenica.mobileffmpeg import FFprobe

print("cv2 vwrsion is",cv2.__version__)
#video_path = "/storage/emulated/0/Movies/VID_20211028_151758.mp4"

# if os.path.exists(video_path):
#     print(video_path)
# else:
#     sys.exit("video path does not exist")
print(tf.__version__)
print(keras.__version__)
filename = join(dirname(__file__), "model_D_new.json")
json_file = open(filename, 'r')
print('json loaded')
# loaded_model_json = json_file.read()
# json_file.close()
data = json.load(open(filename))
jtopy=json.dumps(data)
print("json completed")
loaded_model = tf.keras.models.model_from_json(jtopy)
# # # load weights into new model
print("model loading started")
model_name=join(dirname(__file__), "model_D_new.h5")
loaded_model.load_weights(model_name)
print('model loaded')
print(loaded_model)
median_flow_tracker =cv2.legacy.TrackerMedianFlow_create()
print(median_flow_tracker)
print('created tracker')


# set the sizes of mouth region (ROI) -> input shape
WIDTH = 24
HEIGHT = 32
DEPTH = 28

face_path=join(dirname(__file__), "haarcascade_frontalface_default.xml")
print(face_path)
# Haar cascade classifiers - frontal face, profile face and mouth detection
face_cascade = cv2.CascadeClassifier(face_path)
# print(face_cascade.load(face_path))
face_profile_cascade = cv2.CascadeClassifier(join(dirname(__file__), 'haarcascade_profileface.xml'))
mouth_cascade = cv2.CascadeClassifier(join(dirname(__file__), 'haarcascade_mouth.xml'))
initial_path='/storage/emulated/0/DCIM'
frame_gap=1
vid_length_in_seconds=4
def video_to_npy_array(video):
    count = 0
    lip_frames = []
    frames = []
    found_first_frame = False
    found_lips = False
    found_face = False
    video_array = None
    vid_length_in_seconds=5
#     print(video)
#     cap = cv2.VideoCapture(video)
#     if (cap.isOpened()== False):
#         print("Error opening video stream or file")
    FFmpeg.execute("-i " + video + " -r " + str(frame_gap) + " -f image2 " + initial_path + "/image-%2d.png")
    image_read_counter = 1
    while image_read_counter:
        str_image_read_counter = '%02d' % image_read_counter
        image_path = initial_path + '/image-' + str_image_read_counter + '.png'
        img = cv2.imread(image_path)
#         print(img)
        if img is None:
            print(image_read_counter)
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 3, minSize=(100, 100))
        print('length of faces',len(faces))
        face = faces[0]
        face[3] += 20
        for (x,y,w,h) in faces:
            print("in for loop")
            lower_face = int(h * 0.5)
            print("h is",h)
            lower_face_roi = gray[y + lower_face:y + h, x:x + w]
        mouths = mouth_cascade.detectMultiScale(lower_face_roi, 1.3, 15)
        image_read_counter=image_read_counter+1
        if len(faces) == 0:
             found_face = False
             print('face not found')
        else:
            found_face=True
#         if not found_lips:
#              break
#         else:
#              found_face = True
#         if found_face:
#              print('found face')
#              face = faces[0]
#              face[3] += 20
#         lower_face = int(h * 0.5)
#         lower_face_roi = gray[y + lower_face:y + h, x:x + w]
#         mouths = mouth_cascade.detectMultiScale(lower_face_roi, 1.3, 15)
        if len(mouths) > 0:
             mouth = mouths[0]
             mouth[0] += x  # add face x to fix absolute pos
             mouth[1] += y + lower_face  # add face y to fix absolute pos
        else:
            print('mouth not detected')
        if not found_lips:
            print('lips not found')
            lip_track = mouth
            lip_track[0] -= 10
            lip_track[1] -= 20
            lip_track[2] += 20
            lip_track[3] += 30
            median_flow_tracker.init(img, tuple(lip_track))
            found_lips = True
        else:
            print('lips found')
            ok, bbox = median_flow_tracker.update(img)
            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
#                      cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)
                lips_roi = gray[
                                    int(bbox[1]):int(bbox[1]) + int(bbox[3]),
                                    int(bbox[0]):int(bbox[0]) + int(bbox[2])
                                    ]
                if lips_roi.size == 0:
                    print("roi size 0")
                    break
                print('size before',lips_roi.shape)
                lips_resized = cv2.resize(lips_roi, (HEIGHT, WIDTH), interpolation=cv2.INTER_AREA)
                print('size afrer',lips_resized.shape)
                lip_frames.append(lips_resized)
        if len(frames) != DEPTH:
               frames.append(img)
        count += 1
        print('count is',count)
        if count:
               print("appending array")
               video_array = np.array(lip_frames, dtype="uint8")
               break
#         video_array = np.array(lip_frames, dtype="uint8")
#         print(len(faces))
#     stream=ffmpeg.input('img.png')
#     out, err = (ffmpeg.input(video).filter_('select', 'gte(n,{})'.format(1)).output('pipe:', vframes=1, format='image2', vcodec='mjpeg').run(capture_stdout=True))
#     if out is null:
#         print('out is null')
#     else:
#         print('out is not null')
#     count = 0


    # initialize MedianFlow tracker for tracking mouth region
    #medianflow_tracker = cv2.TrackerMedianFlow_create()



#     while cap.isOpened():
#         # Capture frame-by-frame
#         ret, frame = cap.read()
#         if not ret:
#             if len(lip_frames) > 0:
#                 video_array = np.array(lip_frames, dtype="uint8")
#             break
#         # convert frames to grayscale
#         if frame is null:
#             print('it is null')
#         else:
#             print('it is not null')
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#         if not found_lips:
#             # use Haar classifier to find the frontal face
#
#             print('not found lips')
#             faces = face_cascade.detectMultiScale(gray, 1.2, 3, minSize=(100, 100))
#             if len(faces) == 0:
#
#                 print('frontal face not found')
#                 # if frontal face is not found then try to detect profile face
#                 faces = face_profile_cascade.detectMultiScale(gray, 1.2, 3, minSize=(100, 100))
#                 if len(faces) == 0:
#                     found_face = False
#                     print('face not found')
#                     if not found_lips:
#
#                         break
#                 else:
#                     found_face = True
#             else:
#                 found_face = True
#
#             if found_face:
#                 print('found face')
#                 face = faces[0]
#                 face[3] += 20
#
#                 for (x,y,w,h) in faces:
#                     # drawing rectangle for face
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#
#                 lower_face = int(h * 0.5)
#                 lower_face_roi = gray[y + lower_face:y + h, x:x + w]
#
#                 # detect mouth region in lower half of the face
#                 mouths = mouth_cascade.detectMultiScale(lower_face_roi, 1.3, 15)
#                 if len(mouths) > 0:
#                     # if first mouth is found
#                     mouth = mouths[0]
#                     mouth[0] += x  # add face x to fix absolute pos
#                     mouth[1] += y + lower_face  # add face y to fix absolute pos
#
#                     m = mouth
#                     # drawing rectangle for mouth
#                     cv2.rectangle(frame, (m[0], m[1]), (m[0] + m[2], m[1] + m[3]), (0, 255, 0), 2)
#
#                     # initialized the init tracker
#                     if not found_lips:
#                         lip_track = mouth
#                         # extend tracking area
#                         lip_track[0] -= 10
#                         lip_track[1] -= 20
#                         lip_track[2] += 20
#                         lip_track[3] += 30
#                         medianflow_tracker.init(frame, tuple(lip_track))
#                         found_lips = True
#
#                     if count == 0:
#                         found_first_frame = True
#
#                     if not found_first_frame:
#                         cap = cv2.VideoCapture(video)
#                         found_first_frame = True
#                         continue
#
#                 # skip the sample, if the face is not found
#                 else:
#                     if not found_lips:
#
#                         break
#         # Update medianflow tracker
#         else:
#             ok, bbox = medianflow_tracker.update(frame)
#             # if tracker is successfully matched in following frame
#             if ok:
#                 p1 = (int(bbox[0]), int(bbox[1]))
#                 p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
#                 cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)
#
#                 lips_roi = gray[
#                            int(bbox[1]):int(bbox[1]) + int(bbox[3]),
#                            int(bbox[0]):int(bbox[0]) + int(bbox[2])
#                            ]
#
#                 # prevent crash when tracker goes out of frame
#                 # and skip video if this occurs (eg. waved hand in front of mouth...)
#                 if lips_roi.size == 0:
#                     break
#
#                 lips_resized = cv2.resize(lips_roi, (HEIGHT, WIDTH), interpolation=cv2.INTER_AREA)
#                 lip_frames.append(lips_resized)
#
#
#             # if tracker is lost, skip the sample
#             else:
#
#                 break
#
#         if len(frames) != DEPTH:
#             #cv2.imwrite('outputs/haar/frame-' + str(count) + ".png", frame)
#             frames.append(frame)
#         count += 1
#         if count > DEPTH:
#             video_array = np.array(lip_frames, dtype="uint8")
#             break

    print('end of function')
    print(video_array)
    img_array = np.expand_dims(video_array, axis=3)
    print(img_array.shape)
    img_array = np.expand_dims(img_array, axis=0)
    print(img_array)
    print(video_array.shape)
    print(img_array.shape)
#     prediction = loaded_model.predict(img_array)
#     print(prediction)

#     cap.release()
    return video_array


# img_array = video_to_npy_array(video_path)
# img_array = np.expand_dims(img_array, axis=0)
#
#         #Calling the predict function using keras
# prediction = loaded_model.predict(img_array)#[0][0]
# print(prediction)
# while True:
#         _, frame = video.read()

#         #Convert the captured frame into RGB
#         im = Image.fromarray(frame, 'RGB')

#         #Resizing into dimensions you used while training
#         im = im.resize((24,32))
#         img_array = np.array(im)

#         #Expand dimensions to match the 4D Tensor shape.
# img_array = video_to_npy_array(video_path)
# img_array = np.expand_dims(img_array, axis=0)
#
# #Calling the predict function using keras
# prediction = loaded_model.predict(img_array)#[0][0]
# print(prediction)
# #Customize this part to your liking...
# if(prediction == 1 or prediction == 0):
#     print("No Human")
# elif(prediction < 0.5 and prediction != 0):
#     print("Female")
# elif(prediction > 0.5 and prediction != 1):
#     print("Male")

# cv2.imshow("Prediction", frame)
# key=cv2.waitKey(1)
# if key == ord('q'):
#         break

# video.release()
# cv2.destroyAllWindows()