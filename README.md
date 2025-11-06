# 🌳 Cấu trúc dự án
```
WEB/
├── 📂 asLweb/ (Django Project)
│   ├── settings.py - Cấu hình Django
│   ├── urls.py - Routing chính
│   └── wsgi.py, asgi.py - Deployment
│
├── 📂 learning/ (Django App chính)
│   ├── 📂 templates/learning/ (Giao diện)
│   │   ├── base.html - Template chính
│   │   ├── home.html - Trang chủ
│   │   ├── practice.html - Trang luyện tập chính
│   │   ├── practice_camera.html - Nhận diện chữ cái
│   │   └── practice_words_camera.html - Nhận diện từ vựng
│   │
│   ├── 📂 static/learning/ (Tài nguyên)
│   │   ├── css/style.css - Styling
│   │   ├── js/camera_real.js - Xử lý camera chữ cái
│   │   └── js/words_camera.js - Xử lý camera từ vựng
│   │
│   ├── views.py - Xử lý request & logic
│   ├── urls.py - Routing app
│   ├── models.py - Database models (tùy chọn)
│   ├── ai_recognizer.py - AI nhận diện chữ cái
│   └── word_recognizer.py - AI nhận diện từ vựng
│
├── 📂 models/ (Thư mục model AI)
│   ├── mobilenet_asl_v1_attention_focal.h5 - Model chữ cái
│   └── asl_improved_finetuned.pth - Model từ vựng
│
├── manage.py - Quản lý Django
└── requirements.txt - Dependencies
