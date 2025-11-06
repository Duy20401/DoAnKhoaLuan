# 🌳 Cấu trúc dự án
```
project-root
├─ app/
│  ├─ Http/
│  │  ├─ Controllers/    # nhận request → gọi service/model → trả response
│  │  └─ Middleware/     # chặn/lọc request (auth, throttle...)
│  ├─ Models/            # Eloquent (hasMany/belongsTo...)
│  └─ Providers/         # đăng ký service, event, policy
├─ bootstrap/            # boot + cache runtime
├─ config/               # app.php, database.php, cache.php, mail.php, ...
├─ database/
│  ├─ migrations/        # tạo/sửa bảng
│  ├─ seeders/           # dữ liệu mẫu
│  └─ factories/         # dữ liệu giả (testing)
├─ public/               # document root (index.php, assets Vite)
├─ resources/
│  ├─ views/             # Blade templates (.blade.php)
│  ├─ js/                # front-end (Vite, ESM)
│  └─ css/
├─ routes/
│  ├─ web.php            # web (session, CSRF, Blade)
│  ├─ api.php            # API (stateless, prefix /api)
│  ├─ console.php        # lệnh Artisan tự định nghĩa
│  └─ channels.php       # broadcast channels
├─ storage/
│  ├─ app/               # vị dụ: app/public để lưu upload
│  ├─ framework/         # cache view, sessions, routes, compiled
│  └─ logs/              # laravel.log
├─ tests/                # Feature/Unit tests
└─ vendor/               # Composer packages
```
