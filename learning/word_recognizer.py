# learning/word_recognizer.py
import os
import time
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import traceback
from collections import deque
import base64
from django.conf import settings

# ==================== MODEL ARCHITECTURE (giống file của bạn) ====================
class BackgroundSuppressionModule(nn.Module):
    def __init__(self, cnn_dim=1280, reduced_dim=256, dropout=0.2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(cnn_dim, cnn_dim // 4),
            nn.LayerNorm(cnn_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cnn_dim // 4, cnn_dim),
            nn.Sigmoid()
        )
        self.reducer = nn.Sequential(
            nn.Linear(cnn_dim, reduced_dim),
            nn.LayerNorm(reduced_dim),
            nn.GELU(),
            nn.Dropout(dropout * 1.5)
        )
    def forward(self, cnn_feat):
        gate_weights = self.gate(cnn_feat)
        suppressed = cnn_feat * gate_weights
        reduced = self.reducer(suppressed)
        return reduced

class MotionAttentionModule(nn.Module):
    def __init__(self, motion_dim=12, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.diff_encoder = nn.Sequential(
            nn.Linear(motion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        self.motion_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.GELU()
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, motion_feat):
        x = self.diff_encoder(motion_feat)
        x_conv = self.temporal_conv(x.transpose(1,2)).transpose(1,2)
        x = x + x_conv
        attn_weights = F.softmax(self.attention(x), dim=1)
        attended = (x * attn_weights).sum(dim=1)
        global_motion = self.motion_pool(x.transpose(1,2)).squeeze(-1)
        return torch.cat([attended, global_motion], dim=-1)

class EnhancedHandAttention(nn.Module):
    def __init__(self, hand_dim, num_heads=8, dropout=0.2):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hand_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hand_dim)
        self.dropout = nn.Dropout(dropout)
        self.cross_attention = nn.MultiheadAttention(embed_dim=hand_dim // 2, num_heads=max(1, num_heads // 2), dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hand_dim // 2)
    def forward(self, hand_feat):
        attended, _ = self.attention(hand_feat, hand_feat, hand_feat)
        hand_feat = self.norm(hand_feat + self.dropout(attended))
        B, T, D = hand_feat.shape
        left = hand_feat[:, :, :D//2]
        right = hand_feat[:, :, D//2:]
        left_enh, _ = self.cross_attention(left, right, right)
        right_enh, _ = self.cross_attention(right, left, left)
        left = self.cross_norm(left + self.dropout(left_enh))
        right = self.cross_norm(right + self.dropout(right_enh))
        return torch.cat([left, right], dim=-1)

class AdvancedFeatureProcessor(nn.Module):
    def __init__(self, cnn_dim, pose_dim, hand_dim, shape_dim, motion_dim, hidden_dim=768, dropout=0.35, num_hand_heads=8):
        super().__init__()
        self.bg_suppressor = BackgroundSuppressionModule(cnn_dim, hidden_dim // 3, dropout)
        self.pose_proc = nn.Sequential(nn.Linear(pose_dim, hidden_dim // 8), nn.LayerNorm(hidden_dim // 8), nn.GELU(), nn.Dropout(dropout))
        self.hand_proc = nn.Sequential(nn.Linear(hand_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(dropout * 0.4))
        self.hand_attention = EnhancedHandAttention(hidden_dim // 2, num_heads=num_hand_heads, dropout=dropout * 0.4)
        self.shape_proc = nn.Sequential(nn.Linear(shape_dim, hidden_dim // 4), nn.LayerNorm(hidden_dim // 4), nn.GELU(), nn.Dropout(dropout * 0.6))
        self.motion_attention = MotionAttentionModule(motion_dim, hidden_dim // 6, dropout)
        total_dim = (hidden_dim // 3 + hidden_dim // 8 + hidden_dim // 2 + hidden_dim // 4 + hidden_dim // 6 + hidden_dim // 12)
        self.output_proj = nn.Sequential(nn.Linear(total_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
    def forward(self, cnn_feat, pose_feat, hand_feat, shape_feat, motion_feat):
        B, T, _ = cnn_feat.shape
        cnn_out = self.bg_suppressor(cnn_feat)
        pose_out = self.pose_proc(pose_feat)
        hand_out = self.hand_proc(hand_feat)
        hand_out = self.hand_attention(hand_out)
        shape_out = self.shape_proc(shape_feat)
        motion_global = self.motion_attention(motion_feat)
        motion_global = motion_global.unsqueeze(1).expand(-1, T, -1)
        combined = torch.cat([cnn_out, pose_out, hand_out, shape_out, motion_global], dim=-1)
        out = self.output_proj(combined)
        return out

class AdvancedMobileNetBiLSTM(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=768, lstm_layers=3, dropout=0.35, num_hand_heads=8):
        super().__init__()
        self.cnn_dim = 1280
        self.pose_dim = 99
        self.left_hand_dim = 63
        self.right_hand_dim = 63
        self.left_shape_dim = 10
        self.right_shape_dim = 10

        base_dim = (self.cnn_dim + self.pose_dim + self.left_hand_dim + self.right_hand_dim + self.left_shape_dim + self.right_shape_dim)
        remaining = input_dim - base_dim

        if remaining >= 24:
            self.left_edge_dim = 10
            self.right_edge_dim = 10
            self.motion_dim = remaining - 20
            self.has_edge = True
        else:
            self.left_edge_dim = 0
            self.right_edge_dim = 0
            self.motion_dim = remaining
            self.has_edge = False

        self.hand_dim = self.left_hand_dim + self.right_hand_dim
        self.shape_dim = self.left_shape_dim + self.right_shape_dim
        if self.has_edge:
            self.shape_dim += self.left_edge_dim + self.right_edge_dim

        self.feature_processor = AdvancedFeatureProcessor(self.cnn_dim, self.pose_dim, self.hand_dim, self.shape_dim, self.motion_dim, hidden_dim, dropout, num_hand_heads)

        self.conv_blocks = nn.ModuleList([
            self._make_conv_block(hidden_dim, hidden_dim, stride=1),
            self._make_conv_block(hidden_dim, hidden_dim, stride=2),
            self._make_conv_block(hidden_dim, hidden_dim, stride=1),
            self._make_conv_block(hidden_dim, hidden_dim, stride=1),
        ])
        self.lstm_norm = nn.LayerNorm(hidden_dim)
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=lstm_layers, batch_first=True, bidirectional=True, dropout=dropout if lstm_layers > 1 else 0)
        self.lstm_dropout = nn.Dropout(dropout)
        self.attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Dropout(dropout * 0.5), nn.Linear(hidden_dim // 2, 1))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.LayerNorm(hidden_dim // 4), nn.GELU(), nn.Dropout(dropout * 0.8),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        self._init_weights()

    def _make_conv_block(self, in_ch, out_ch, stride):
        return nn.Sequential(nn.Conv1d(in_ch, out_ch, 3, stride, 1, bias=False), nn.BatchNorm1d(out_ch), nn.GELU(), nn.Dropout(0.1))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, x):
        B, T, C = x.shape
        idx = 0
        cnn_feat = x[:,:,idx:idx+self.cnn_dim]; idx += self.cnn_dim
        pose_feat = x[:,:,idx:idx+self.pose_dim]; idx += self.pose_dim
        left_hand = x[:,:,idx:idx+self.left_hand_dim]; idx += self.left_hand_dim
        right_hand = x[:,:,idx:idx+self.right_hand_dim]; idx += self.right_hand_dim
        hand_feat = torch.cat([left_hand, right_hand], dim=-1)
        left_shape = x[:,:,idx:idx+self.left_shape_dim]; idx += self.left_shape_dim
        right_shape = x[:,:,idx:idx+self.right_shape_dim]; idx += self.right_shape_dim
        shape_feat = torch.cat([left_shape, right_shape], dim=-1)
        if self.has_edge:
            left_edge = x[:,:,idx:idx+self.left_edge_dim]; idx += self.left_edge_dim
            right_edge = x[:,:,idx:idx+self.right_edge_dim]; idx += self.right_edge_dim
            shape_feat = torch.cat([shape_feat, left_edge, right_edge], dim=-1)
        motion_feat = x[:,:,idx:]
        x = self.feature_processor(cnn_feat, pose_feat, hand_feat, shape_feat, motion_feat)
        x = x.transpose(1,2)
        for block in self.conv_blocks:
            x = block(x)
        x = x.transpose(1,2)
        x = self.lstm_norm(x)
        lstm_out, _ = self.bilstm(x)
        lstm_out = self.lstm_dropout(lstm_out)
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        pooled = (lstm_out * attn_weights).sum(dim=1)
        out = self.classifier(pooled)
        return out

# ==================== FEATURE EXTRACTOR ====================
class RealtimeFeatureExtractor:
    def __init__(self, device, motion_dim=11):
        self.device = device
        self.target_frames = 120
        self.img_size = 224
        self.motion_dim = motion_dim
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(static_image_mode=False, model_complexity=1,
                                                  enable_segmentation=False, min_detection_confidence=0.5,
                                                  min_tracking_confidence=0.5)
        try:
            from torchvision.models import mobilenet_v2
            from torchvision import transforms
            self.mobilenet = mobilenet_v2(pretrained=True).features.to(device).eval()
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
            print("[OK] MobileNetV2 loaded for CNN features")
        except Exception as e:
            print(f"[WARNING] MobileNetV2 not available: {e}")
            self.mobilenet = None
            self.transform = None

    def normalize_landmarks(self, landmarks, expected_len):
        if landmarks is None:
            return np.zeros(expected_len, dtype=np.float32)
        arr = np.array(landmarks).reshape(-1, 3)
        if arr.size == 0:
            return np.zeros(expected_len, dtype=np.float32)
        reference = arr[0].copy()
        arr = arr - reference
        scale = np.max(np.abs(arr)) + 1e-6
        arr = arr / scale
        flat = arr.flatten().astype(np.float32)
        if flat.shape[0] == expected_len:
            return flat
        if flat.shape[0] < expected_len:
            pad = np.zeros(expected_len - flat.shape[0], dtype=np.float32)
            return np.concatenate([flat, pad], axis=0)
        else:
            return flat[:expected_len]

    def compute_hand_shape_features(self, hand_landmarks):
        if hand_landmarks is None:
            return np.zeros(10, dtype=np.float32)
        try:
            lm = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark])
            features = []
            palm = np.linalg.norm(lm[0] - lm[9])
            features.append(palm)
            finger_tips = [4,8,12,16,20]; finger_bases=[2,5,9,13,17]
            for t,b in zip(finger_tips,finger_bases):
                features.append(np.linalg.norm(lm[t]-lm[b]))
            for i in range(4):
                v1 = lm[finger_tips[i]] - lm[0]; v2 = lm[finger_tips[i+1]] - lm[0]
                cos = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
                features.append(cos)
            return np.array(features, dtype=np.float32)
        except Exception:
            return np.zeros(10, dtype=np.float32)

    def extract_frame_features(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # CNN features
        if self.mobilenet is not None and self.transform is not None:
            try:
                t = self.transform(frame_rgb).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    cnn = self.mobilenet(t)
                    cnn = F.adaptive_avg_pool2d(cnn, 1).squeeze().cpu().numpy()
            except Exception:
                cnn = np.zeros(1280, dtype=np.float32)
        else:
            cnn = np.zeros(1280, dtype=np.float32)
        
        # MediaPipe landmarks
        results = self.holistic.process(frame_rgb)
        
        if getattr(results, "pose_landmarks", None):
            pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
            pose_norm = self.normalize_landmarks(pose, expected_len=99)
        else:
            pose_norm = np.zeros(99, dtype=np.float32)
        
        if getattr(results, "left_hand_landmarks", None):
            left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
            left_norm = self.normalize_landmarks(left_hand, expected_len=63)
        else:
            left_norm = np.zeros(63, dtype=np.float32)
        
        if getattr(results, "right_hand_landmarks", None):
            right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
            right_norm = self.normalize_landmarks(right_hand, expected_len=63)
        else:
            right_norm = np.zeros(63, dtype=np.float32)
        
        left_shape = self.compute_hand_shape_features(getattr(results, "left_hand_landmarks", None))
        right_shape = self.compute_hand_shape_features(getattr(results, "right_hand_landmarks", None))
        
        motion_feat = np.zeros(self.motion_dim, dtype=np.float32)
        
        features = np.concatenate([cnn, pose_norm, left_norm, right_norm, left_shape, right_shape, motion_feat], axis=0).astype(np.float32)
        return features, results

    def compute_motion_features(self, buffer):
        if len(buffer) < 2: 
            return
        curr = buffer[-1]; prev = buffer[-2]
        base = 1280 + 99
        curr_left = curr[base: base + 63]; prev_left = prev[base: base + 63]
        curr_right = curr[base + 63: base + 126]; prev_right = prev[base + 63: base + 126]
        
        left_vel = float(np.linalg.norm(curr_left - prev_left))
        right_vel = float(np.linalg.norm(curr_right - prev_right))
        
        if len(buffer) >= 3:
            prev2 = buffer[-3]
            prev2_left = prev2[base: base + 63]
            left_acc = abs(left_vel - float(np.linalg.norm(prev_left - prev2_left)))
        else:
            left_acc = 0.0
        
        hands_dist = float(np.linalg.norm(curr_left[:3] - curr_right[:3]))
        
        base_motion = [
            left_vel, right_vel, left_acc, 0.0, hands_dist,
            left_vel + right_vel, abs(left_vel - right_vel), left_vel * right_vel
        ]
        
        motion = np.array(base_motion + [0.0] * (self.motion_dim - len(base_motion)), dtype=np.float32)
        
        if buffer[-1].shape[0] >= self.motion_dim:
            buffer[-1][-self.motion_dim:] = motion

    def get_motion_magnitude(self, features):
        motion_start = len(features) - self.motion_dim
        motion_features = features[motion_start:]
        if len(motion_features) >= 2:
            return float(motion_features[0] + motion_features[1])
        return 0.0

    def close(self):
        try:
            self.holistic.close()
        except Exception:
            pass

# ==================== SLIDING WINDOW DETECTOR ====================
class SlidingWindowDetector:
    def __init__(self, window_size=30, stride=10, min_motion=0.001):
        self.window_size = window_size
        self.stride = stride
        self.min_motion = min_motion
        
        self.frame_buffer = deque(maxlen=window_size)
        self.frame_count = 0
        self.motion_history = deque(maxlen=10)
        self.is_active = False
        
        print(f"[INIT] SlidingWindowDetector: window_size={window_size}, stride={stride}")
    
    def update(self, motion_magnitude, frame_features):
        self.motion_history.append(motion_magnitude)
        
        if len(self.motion_history) >= 3:
            avg_motion = np.mean(list(self.motion_history)[-5:])
            self.is_active = avg_motion > self.min_motion
        
        self.frame_buffer.append(frame_features)
        self.frame_count += 1
        
        should_predict = (
            len(self.frame_buffer) >= self.window_size and
            self.frame_count % self.stride == 0 and
            self.is_active
        )
        
        if should_predict:
            return True, list(self.frame_buffer)
        
        return False, []
    
    def get_debug_info(self):
        avg_motion = np.mean(list(self.motion_history)) if self.motion_history else 0.0
        return {
            'motion': avg_motion,
            'buffer_size': len(self.frame_buffer),
            'frame_count': self.frame_count,
            'is_active': self.is_active
        }
    
    def reset(self):
        self.frame_buffer.clear()
        self.motion_history.clear()
        self.frame_count = 0
        self.is_active = False

# ==================== MAIN WORD RECOGNIZER CLASS ====================
class ASLWordRecognizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.extractor = None
        self.detector = None
        self.class_list = []
        self.is_initialized = False
        self.buffer = []
        self.prediction_history = deque(maxlen=3)
        self.last_prediction = "Chưa nhận diện"
        self.last_confidence = 0.0
        
    def initialize(self, model_path=None, class_names_path=None):
        """Khởi tạo model nhận diện từ"""
        try:
            print("🚀 Initializing ASL Word Recognizer...")
            
            # Tìm model path
            if model_path is None:
                model_paths = [
                    "models/asl_improved_finetuned.pth",
                    "models/asl_improved.pth",
                    "asl_improved_finetuned.pth",
                    "asl_improved.pth"
                ]
                model_path = next((p for p in model_paths if os.path.exists(p)), None)
            
            if model_path is None or not os.path.exists(model_path):
                print("❌ Model file not found")
                return False

            # Load class names
            if class_names_path and os.path.exists(class_names_path):
                with open(class_names_path, "r", encoding="utf-8") as f:
                    self.class_list = json.load(f)
            else:
                # Fallback class names
                self.class_list = ["hello", "thank you", "please", "sorry", "help", "love", "family", "friend"] + \
                                [f"word_{i}" for i in range(500)]
            
            print(f"📖 Loaded {len(self.class_list)} class names")

            # Load model
            ckpt = torch.load(model_path, map_location=self.device)
            state = ckpt.get("model_state_dict", ckpt)
            
            # Infer parameters
            motion_dim = 11  # Default
            base_dim = 1280 + 99 + 63 + 63 + 10 + 10
            input_dim = base_dim + motion_dim
            
            num_classes = len(self.class_list)
            
            # Create model
            self.model = AdvancedMobileNetBiLSTM(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dim=768,
                lstm_layers=3,
                dropout=0.35,
                num_hand_heads=8
            ).to(self.device)
            
            self.model.load_state_dict(state)
            self.model.eval()
            
            # Initialize feature extractor and detector
            self.extractor = RealtimeFeatureExtractor(self.device, motion_dim=motion_dim)
            self.detector = SlidingWindowDetector(window_size=30, stride=5, min_motion=0.0008)
            
            self.is_initialized = True
            print("✅ ASL Word Recognizer initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing ASL Word Recognizer: {e}")
            traceback.print_exc()
            return False
    
    def process_frame(self, frame):
        """Xử lý frame và trả về kết quả nhận diện"""
        if not self.is_initialized:
            return "Model chưa khởi tạo", 0.0
        
        try:
            # Extract features
            features, _ = self.extractor.extract_frame_features(frame)
            
            # Update buffer
            self.buffer.append(features)
            if len(self.buffer) >= 2:
                self.extractor.compute_motion_features(self.buffer)
            if len(self.buffer) > self.extractor.target_frames * 2:
                self.buffer.pop(0)
            
            # Get motion magnitude
            motion_mag = self.extractor.get_motion_magnitude(features)
            
            # Update sliding window
            should_predict, window_data = self.detector.update(motion_mag, features)
            
            # Predict if needed
            if should_predict and len(window_data) > 0:
                target_frames = self.extractor.target_frames
                
                # Resample window
                if len(window_data) >= target_frames:
                    indices = np.linspace(0, len(window_data) - 1, target_frames, dtype=int)
                    sampled = np.array([window_data[i] for i in indices])
                else:
                    sampled = np.array(window_data)
                    pad_len = target_frames - len(sampled)
                    padding = np.tile(sampled[-1:], (pad_len, 1))
                    sampled = np.vstack([sampled, padding])
                
                x = torch.from_numpy(sampled).unsqueeze(0).float().to(self.device)
                
                with torch.no_grad():
                    out = self.model(x)
                    probs = torch.softmax(out, dim=1)
                    conf, idx = torch.max(probs, dim=1)
                    
                    pred_idx = idx.item()
                    pred_conf = float(conf.item())
                    
                    if pred_idx < len(self.class_list):
                        pred_class = self.class_list[pred_idx]
                    else:
                        pred_class = f"Class_{pred_idx}"
                    
                    # Update prediction with confidence threshold
                    if pred_conf > 0.3:
                        self.prediction_history.append((pred_class, pred_conf))
                        
                        # Voting from history
                        if len(self.prediction_history) >= 2:
                            votes = {}
                            for p, c in self.prediction_history:
                                votes[p] = votes.get(p, 0) + c
                            best_pred = max(votes.items(), key=lambda x: x[1])
                            self.last_prediction = best_pred[0]
                            self.last_confidence = best_pred[1] / len(self.prediction_history)
                        else:
                            self.last_prediction = pred_class
                            self.last_confidence = pred_conf
            
            return self.last_prediction, self.last_confidence
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            return "Lỗi xử lý", 0.0
    
    def reset(self):
        """Reset buffer và lịch sử nhận diện"""
        if self.detector:
            self.detector.reset()
        self.prediction_history.clear()
        self.buffer.clear()
        self.last_prediction = "Đã reset"
        self.last_confidence = 0.0
    
    def close(self):
        """Giải phóng tài nguyên"""
        if self.extractor:
            self.extractor.close()
        self.is_initialized = False

# Global instance
word_recognizer = ASLWordRecognizer()

def init_word_recognizer(model_path=None, class_names_path=None):
    """Khởi tạo recognizer toàn cục"""
    return word_recognizer.initialize(model_path, class_names_path)

def get_word_recognizer():
    """Lấy instance của recognizer"""
    return word_recognizer if word_recognizer.is_initialized else None