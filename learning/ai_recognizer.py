# learning/ai_recognizer.py
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import os

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss_fixed

class ASLRecognition:
    def __init__(self, model_path):
        # Load model
        self.model = load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss()})
        
        # Classes chỉ bao gồm chữ cái
        self.CLASSES = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'del', 'nothing', 'space'
        ]
        
        self.IMG_SIZE = 224
        
        # Mediapipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        
        print("✅ ASL Recognition model loaded successfully!")
    
    def process_frame(self, frame):
        """
        Xử lý frame và nhận diện ký hiệu
        Trả về: (prediction, confidence, bbox)
        """
        try:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get hand bounding box
                    h, w, _ = frame.shape
                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]
                    x_min = int(min(x_coords) * w) - 20
                    x_max = int(max(x_coords) * w) + 20
                    y_min = int(min(y_coords) * h) - 20
                    y_max = int(max(y_coords) * h) + 20
                    
                    # Ensure within frame bounds
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(w, x_max), min(h, y_max)
                    
                    # Extract ROI
                    roi = frame[y_min:y_max, x_min:x_max]
                    if roi.size == 0:
                        continue
                    
                    # Preprocess image
                    img = cv2.resize(roi, (self.IMG_SIZE, self.IMG_SIZE))
                    img = img.astype("float32") / 255.0
                    img = np.expand_dims(img, axis=0)
                    
                    # Predict
                    preds = self.model.predict(img, verbose=0)
                    class_id = np.argmax(preds)
                    confidence = float(np.max(preds))
                    prediction = self.CLASSES[class_id]
                    
                    # Chỉ trả về chữ cái, bỏ qua 'del', 'nothing', 'space'
                    if prediction in ['del', 'nothing', 'space']:
                        prediction = "Không nhận diện"
                        confidence = 0.0
                    
                    return prediction, confidence, (x_min, y_min, x_max, y_max)
            
            return "Không có tay", 0.0, None
            
        except Exception as e:
            print(f"❌ Lỗi trong process_frame: {e}")
            return "Lỗi nhận diện", 0.0, None
    
    def close(self):
        """Giải phóng tài nguyên"""
        self.hands.close()

# Global instance
asl_recognizer = None

def init_recognizer(model_path):
    """Khởi tạo recognizer toàn cục"""
    global asl_recognizer
    try:
        asl_recognizer = ASLRecognition(model_path)
        return True
    except Exception as e:
        print(f"❌ Lỗi khởi tạo recognizer: {e}")
        return False

def get_recognizer():
    """Lấy instance của recognizer"""
    return asl_recognizer