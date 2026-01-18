import numpy as np
from ultralytics import YOLO
from .interfaces import PoseEstimator

class YOLOPoseEstimator(PoseEstimator):
    """
    Adapter for YOLOv11-pose models.
    """
    def __init__(self, model_path: str = 'yolo11n-pose.pt', max_people: int = 6):
        """
        Args:
            model_path: Path to the YOLO model file.
            max_people: Maximum number of people to track (padding/truncating output).
        """
        self.model = YOLO(model_path)
        self.max_people = max_people

    def estimate(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # YOLO inference
        results = self.model(frame, verbose=False)
        
        if not results:
             return self._empty_output()

        result = results[0]
        
        # Get frame dimensions for normalization
        h, w = frame.shape[:2]

        # Extract Keypoints
        # Use simple property if available for normalized xy.
        # result.keypoints.xyn gives (N, 17, 2) normalized [x, y]
        # result.keypoints.conf gives (N, 17) confidence
        if result.keypoints is not None:
             # Retrieve normalized coordinates directly provided by Ultralytics
             # Note: We must move to CPU/numpy for compatibility with downstream logic
             kpts_xy_norm = result.keypoints.xyn.cpu().numpy() # (N, 17, 2)
             kpts_conf = result.keypoints.conf.cpu().numpy()   # (N, 17)
             
             # Stack to create (N, 17, 3) -> [x_norm, y_norm, conf]
             # Expand conf to (N, 17, 1) first
             kpts_conf = kpts_conf[..., np.newaxis]
             kpts = np.concatenate((kpts_xy_norm, kpts_conf), axis=2)
             
        else:
             kpts = np.zeros((0, 17, 3))

        # Extract Boxes
        # Use xyxyn for normalized xyxy [x1, y1, x2, y2]
        if result.boxes is not None:
             # xyxyn format [x1, y1, x2, y2] normalized
             boxes_xyxy_norm = result.boxes.xyxyn.cpu().numpy()
             confs = result.boxes.conf.cpu().numpy()
             
             # Create format (N, 5) -> (ymin, xmin, ymax, xmax, score)
             # boxes_xyxy_norm is [xmin, ymin, xmax, ymax]
             
             num_boxes = len(boxes_xyxy_norm)
             formatted_boxes = np.zeros((num_boxes, 5))
             formatted_boxes[:, 0] = boxes_xyxy_norm[:, 1] # ymin
             formatted_boxes[:, 1] = boxes_xyxy_norm[:, 0] # xmin
             formatted_boxes[:, 2] = boxes_xyxy_norm[:, 3] # ymax
             formatted_boxes[:, 3] = boxes_xyxy_norm[:, 2] # xmax
             formatted_boxes[:, 4] = confs
             
        else:
             formatted_boxes = np.zeros((0, 5))

        # Process Keypoints: [x, y, conf] -> [y, x, conf] (Normalized)
        if len(kpts) > 0:
            # Current kpts is [x, y, conf]
            # Swap x and y to get [y, x, conf]
            kpts[:, :, [0, 1]] = kpts[:, :, [1, 0]]
            
        # Pad or Truncate to self.max_people
        final_kpts = np.zeros((self.max_people, 17, 3))
        final_boxes = np.zeros((self.max_people, 5))
        
        num_detected = min(len(kpts), self.max_people)
        
        if num_detected > 0:
            final_kpts[:num_detected] = kpts[:num_detected]
            final_boxes[:num_detected] = formatted_boxes[:num_detected]
            
        return final_kpts, final_boxes

    def _empty_output(self):
        return np.zeros((self.max_people, 17, 3)), np.zeros((self.max_people, 5))
