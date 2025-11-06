WEB/
├── aslweb/                     # Django Project
│   ├── settings.py             # Cấu hình Django
│   ├── urls.py                 # Routing chính
│   └── wsgi.py, asgi.py        # Deployment
│
├── learning/                   # Django App chính
│   ├── templates/learning/     # Giao diện HTML
│   │   ├── base.html           # Template chung
│   │   ├── home.html           # Trang chủ
│   │   ├── practice.html       # Trang luyện tập
│   │   ├── practice_camera.html         # Nhận diện chữ cái
│   │   └── practice_words_camera.html   # Nhận diện từ vựng
│   │
│   ├── static/learning/        # Tài nguyên tĩnh
│   │   ├── css/style.css       # Styling
│   │   ├── js/camera_real.js   # JS xử lý camera (chữ cái)
│   │   └── js/words_camera.js  # JS xử lý camera (từ vựng)
│   │
│   ├── views.py                # Xử lý request & logic
│   ├── urls.py                 # Routing của app
│   ├── models.py               # Database models (tùy chọn)
│   ├── ai_recognizer.py        # AI nhận diện chữ cái
│   └── word_recognizer.py      # AI nhận diện từ vựng
│
├── models/                     # Thư mục chứa model AI
│   ├── mobilenet_asl_v1_attention_focal.h5    # Model chữ cái (Keras)
│   └── asl_improved_finetuned.pth             # Model từ vựng (PyTorch)
│
├── manage.py                   # Quản lý Django
└── requirements.txt            # Danh sách dependencies
