from django.shortcuts import render
from django.http import JsonResponse
import os
import base64
import cv2
import numpy as np

from .word_recognizer import init_word_recognizer, get_word_recognizer


# Import AI recognizers
from .ai_recognizer import init_recognizer, get_recognizer
from .word_recognizer import init_word_recognizer, get_word_recognizer

# Khởi tạo model AI khi server start
ASL_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'mobilenetv2_asl_final.h5')
WORD_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'mobilenet_asl_v1_attention_focal.h5')

# Kiểm tra và khởi tạo ASL model
if os.path.exists(ASL_MODEL_PATH):
    print("🚀 Đang khởi tạo ASL Recognition Model...")
    if init_recognizer(ASL_MODEL_PATH):
        print("✅ ASL Model khởi tạo thành công!")
    else:
        print("❌ Không thể khởi tạo ASL Model")
else:
    print(f"❌ ASL Model file không tồn tại: {ASL_MODEL_PATH}")

# Kiểm tra và khởi tạo Word model
if os.path.exists(WORD_MODEL_PATH):
    print("🚀 Đang khởi tạo Word Recognition Model...")
    if init_word_recognizer(WORD_MODEL_PATH):
        print("✅ Word Model khởi tạo thành công!")
    else:
        print("❌ Không thể khởi tạo Word Model")
else:
    print(f"❌ Word Model file không tồn tại: {WORD_MODEL_PATH}")

def home(request):
    """Trang chủ"""
    return render(request, 'learning/home.html')

def learn_alphabet(request):
    """Trang học bảng chữ cái"""
    return render(request, 'learning/letters.html')

def alphabet_detail(request, letter):
    """Chi tiết chữ cái"""
    context = {'letter': letter.upper()}
    return render(request, 'learning/alphabet_detail.html', context)

def learn_words(request):
    """Trang học từ vựng"""
    return render(request, 'learning/words.html')

def word_detail(request, word):
    """Chi tiết từ vựng"""
    context = {'word': word}
    return render(request, 'learning/word_detail.html', context)

def practice(request):
    """Trang luyện tập chính"""
    return render(request, 'learning/practice.html')

def practice_camera(request):
    """Trang luyện tập với camera - SỬ DỤNG AI THẬT"""
    # Kiểm tra model có sẵn sàng không
    model_ready = get_recognizer() is not None
    context = {'model_ready': model_ready}
    return render(request, 'learning/practice_camera.html', context)

# API để nhận diện từ frontend
def api_recognize(request):
    """API nhận diện ASL từ frame ảnh"""
    if request.method == 'POST':
        try:
            recognizer = get_recognizer()
            if not recognizer:
                return JsonResponse({
                    'success': False,
                    'prediction': 'Model chưa sẵn sàng',
                    'confidence': 0
                })
            
            # Nhận frame ảnh từ frontend
            image_data = request.POST.get('image')
            if not image_data:
                return JsonResponse({
                    'success': False,
                    'prediction': 'Không có dữ liệu ảnh',
                    'confidence': 0
                })
            
            # Decode base64 image
            format, imgstr = image_data.split(';base64,')
            image_bytes = base64.b64decode(imgstr)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return JsonResponse({
                    'success': False,
                    'prediction': 'Không thể decode ảnh',
                    'confidence': 0
                })
            
            # Nhận diện
            prediction, confidence, bbox = recognizer.process_frame(frame)
            
            return JsonResponse({
                'success': True,
                'prediction': prediction,
                'confidence': confidence * 100,  # Convert to percentage
                'bbox': bbox
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'prediction': f'Lỗi: {str(e)}',
                'confidence': 0
            })
    
    return JsonResponse({'success': False, 'prediction': 'Method not allowed', 'confidence': 0})


# Khởi tạo word recognizer (thêm vào phần khởi tạo)
WORD_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'asl_improved_finetuned.pth')  # Thay bằng model của bạn

# Kiểm tra và khởi tạo
if os.path.exists(WORD_MODEL_PATH):
    print("🚀 Đang khởi tạo ASL Word Recognition Model...")
    if init_word_recognizer(WORD_MODEL_PATH):
        print("✅ ASL Word Model khởi tạo thành công!")
    else:
        print("❌ Không thể khởi tạo ASL Word Model")
else:
    print(f"❌ Word model file không tồn tại: {WORD_MODEL_PATH}")

# Thêm view mới cho nhận diện từ
def practice_words_camera(request):
    """Trang luyện tập nhận diện từ với camera"""
    model_ready = get_word_recognizer() is not None
    context = {
        'model_ready': model_ready,
        'practice_type': 'words'
    }
    return render(request, 'learning/practice_words_camera.html', context)

# API cho nhận diện từ
def api_recognize_words(request):
    """API nhận diện từ vựng ASL từ frame ảnh"""
    if request.method == 'POST':
        try:
            recognizer = get_word_recognizer()
            if not recognizer:
                return JsonResponse({
                    'success': False,
                    'prediction': 'Word model chưa sẵn sàng',
                    'confidence': 0
                })
            
            # Nhận frame ảnh từ frontend
            image_data = request.POST.get('image')
            if not image_data:
                return JsonResponse({
                    'success': False,
                    'prediction': 'Không có dữ liệu ảnh',
                    'confidence': 0
                })
            
            # Decode base64 image
            format, imgstr = image_data.split(';base64,')
            image_bytes = base64.b64decode(imgstr)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return JsonResponse({
                    'success': False,
                    'prediction': 'Không thể decode ảnh',
                    'confidence': 0
                })
            
            # Nhận diện từ
            prediction, confidence = recognizer.process_frame(frame)
            
            return JsonResponse({
                'success': True,
                'prediction': prediction,
                'confidence': confidence * 100,  # Convert to percentage
                'type': 'word'
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'prediction': f'Lỗi: {str(e)}',
                'confidence': 0
            })
    
    return JsonResponse({'success': False, 'prediction': 'Method not allowed', 'confidence': 0})
# learning/views.py - CẬP NHẬT API nhận diện từ
def api_recognize_words(request):
    """API nhận diện từ vựng ASL từ frame ảnh"""
    if request.method == 'POST':
        try:
            # Kiểm tra reset request
            if request.POST.get('reset') == 'true':
                recognizer = get_word_recognizer()
                if recognizer:
                    recognizer.reset()
                    return JsonResponse({
                        'success': True,
                        'message': 'Reset thành công'
                    })
                else:
                    return JsonResponse({
                        'success': False,
                        'message': 'Recognizer not available'
                    })
            
            # Xử lý nhận diện bình thường
            recognizer = get_word_recognizer()
            if not recognizer:
                return JsonResponse({
                    'success': False,
                    'prediction': 'Word model chưa sẵn sàng',
                    'confidence': 0
                })
            
            # Nhận frame ảnh từ frontend
            image_data = request.POST.get('image')
            if not image_data:
                return JsonResponse({
                    'success': False,
                    'prediction': 'Không có dữ liệu ảnh',
                    'confidence': 0
                })
            
            # Decode base64 image
            format, imgstr = image_data.split(';base64,')
            image_bytes = base64.b64decode(imgstr)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return JsonResponse({
                    'success': False,
                    'prediction': 'Không thể decode ảnh',
                    'confidence': 0
                })
            
            # Nhận diện từ
            prediction, confidence = recognizer.process_frame(frame)
            
            return JsonResponse({
                'success': True,
                'prediction': prediction,
                'confidence': confidence * 100,  # Convert to percentage
                'type': 'word'
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'prediction': f'Lỗi: {str(e)}',
                'confidence': 0
            })
    
    return JsonResponse({'success': False, 'prediction': 'Method not allowed', 'confidence': 0})