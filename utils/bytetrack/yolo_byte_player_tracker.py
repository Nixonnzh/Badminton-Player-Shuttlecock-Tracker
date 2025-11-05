from ultralytics import YOLO
from utils.bytetrack import read_video, save_video
import cv2
import os
import pickle

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.bytetrack_yaml_path = 'utils/bytetrack/bytetrack.yaml'

    def detect_frame(self, frame):
        """This function returns a dictionary containing the key of each player and the value of [bbox, confidence]."""
        model = self.model
        tracker = model.track(frame, 
                              persist=True, 
                              tracker=self.bytetrack_yaml_path)[0]
        tracker_id = tracker.names  # dict
        player_dict = {}
        for box in tracker.boxes:
            # print(box)
            box_id = int(box.id.tolist()[0])
            xyxy = box.xyxy.tolist()[0]
            confidence = float(box.conf.tolist()[0])  # Extract confidence score
            player_id = box.cls.tolist()[0]
            player_name = tracker_id[player_id]
            if player_name == "Player1":
                player_dict[box_id] = [xyxy, confidence]
            else:
                player_dict[box_id] = [xyxy, confidence]
            # print(player_dict)
        return player_dict

    def detect_player(self, frames, last_detect=False, path_of_last_detect=None):
        """This function detects the player in each frame and returns it as a list of dictionaries containing bbox and confidence."""
        # read last detect player
        if last_detect and path_of_last_detect is not None:
            with open(path_of_last_detect, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        player_detections = []
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if path_of_last_detect is not None:
            with open(path_of_last_detect, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def player_positions(frames, detections):
        c_positions = {}
        c_bboxes = []
        for k, bbox in zip(detections.keys(), detections.values()):
            x1,y1,x2,y2=det
            c_x = int(x2-x1)/2
            c_y = int(y2-y1)/2
            id = k
            c_bboxes.append(c_x)
            c_bboxes.append(c_y)
            c_positions = {id : c_bboxes}
        return c_positions
    
    def draw_player_bbox(self, frames, player_detections):
        # player_detections = self.detect_player(frames)
        player_frames = []
        for frame, player_detect in zip(frames, player_detections):
            # Create a copy of the frame to avoid modifying the original
            frame_copy = frame.copy()
            for id, detection_data in player_detect.items():
                # Handle both old format (just bbox) and new format (bbox + confidence)
                if isinstance(detection_data, list) and len(detection_data) == 2:
                    box, confidence = detection_data
                    x1, y1, x2, y2 = box
                    conf_text = f"Player: {id} ({confidence:.2f})"
                else:
                    # Handle old format for backward compatibility
                    box = detection_data
                    x1, y1, x2, y2 = box
                    conf_text = f"Player: {id}"
                
                if id == 1:
                    cv2.putText(frame_copy, conf_text, (int(box[0]), int(box[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                else:
                    cv2.putText(frame_copy, conf_text, (int(box[0]), int(box[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            player_frames.append(frame_copy)
        return player_frames