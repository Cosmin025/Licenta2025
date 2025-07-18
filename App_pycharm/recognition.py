import cv2
import numpy as np
from ultralytics import YOLO
from utils import hip_center_and_normalize,resource_path
import keras
import tensorflow as tf
import pickle
from collections import deque
import mediapipe as mp


rec_save_file = resource_path("saves\\recording_detections.avi")
live_save_file = resource_path("saves\\live_detections.avi")




action_model_file="action_recognition_models\\1d_cnn_pose.keras"
label_encoder_file = "action_recognition_models\\label_encoder.pkl"

mp_pose = mp.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=1,  # 0 = light, 1 = normal, 2 = heavy
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

action_model = keras.models.load_model(
    resource_path(action_model_file)
)

with open(resource_path(label_encoder_file), 'rb') as file:
    label_encoder = pickle.load(file)

period = 0
period_counter = 0
NUMBER_OF_FRAMES = 15
YOLO_MIN_DETECTION_CONFIDENCE = 0.6

class Box:
     def __init__(self, id):
         self.id = id
         self.pl_queue = deque(maxlen=NUMBER_OF_FRAMES)
         self.frames_queue = deque(maxlen=NUMBER_OF_FRAMES)

boxes_dict = {}


def action_recognition(live:bool,play: bool, save_output: bool, video_path: str):
    yolo_model = YOLO(resource_path('yolo_models\\yolo11n.pt'))
    if not live and video_path == "":
        print("No video path specified")
        return

    if live:
        video_path=0

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if live:
            writer = cv2.VideoWriter(live_save_file, fourcc, fps, (width, height))
        else:
            writer = cv2.VideoWriter(rec_save_file, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = yolo_model.track(frame, persist=True)[0]

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()
            classes = result.boxes.cls.int().cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()


            filtered = [(box, tid) for box, tid, cls, conf in zip(boxes, track_ids, classes, confidences) if
                        cls == 0 and conf > YOLO_MIN_DETECTION_CONFIDENCE]

            if len(filtered) == 0:
                continue


            current_ids = set(tid for _, tid in filtered)
            for key in list(boxes_dict.keys()):
                if key not in current_ids:
                    del boxes_dict[key]

            for box, track_id in filtered:
                if track_id not in boxes_dict:
                    boxes_dict[track_id] = Box(track_id)

            for box, track_id in filtered:

                x1, y1, x2,y2 = box

                person_crop = frame[int(y1):int(y2),int(x1):int(x2)]

                r = mp_pose.process(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))

                box_dict=None
                label = 'unknown'
                action_conf = 0

                if r.pose_landmarks is not None:
                    frame_landmarks = r.pose_landmarks.landmark
                    box_dict = boxes_dict[track_id]
                    box_dict.pl_queue.append([(lm.x, lm.y, lm.z, lm.visibility, lm.presence) for lm in frame_landmarks])

                if box_dict is not None and len(box_dict.pl_queue) == NUMBER_OF_FRAMES:
                    pose_data = np.array(box_dict.pl_queue)
                    pose_data = pose_data.reshape(1, NUMBER_OF_FRAMES, 33, 5)
                    pose_data = hip_center_and_normalize(pose_data)
                    pose_data = pose_data.reshape(1, NUMBER_OF_FRAMES, 33 * 5)

                    prediction = action_model.predict(pose_data)[0]
                    predicted_index = np.argmax(prediction)
                    label = label_encoder.inverse_transform([predicted_index])[0]
                    action_conf = prediction[predicted_index]

                s = f"{label} : {action_conf:.2f}"

                cv2.rectangle(frame, (int(x1),int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, str(s),(int(x1),int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 255, 0), 1)

        if play:
            cv2.imshow("Multi-Person Detection", frame)

        if save_output and writer:
            writer.write(frame)

        if cv2.getWindowProperty("Multi-Person Detection", cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    if cap:
        cap.release()

    print("Recognition window closed")
