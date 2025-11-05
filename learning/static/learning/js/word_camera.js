// learning/static/learning/js/words_camera.js
class ASLWordsRecognizer {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.startBtn = document.getElementById('start-btn');
        this.stopBtn = document.getElementById('stop-btn');
        this.resetBtn = document.getElementById('reset-btn');
        this.speakBtn = document.getElementById('speak-btn');
        this.toggleSpeechBtn = document.getElementById('toggle-speech-btn');
        this.result = document.getElementById('prediction-result');
        this.confidence = document.getElementById('confidence');
        this.confidenceBar = document.getElementById('confidence-bar');
        this.status = document.getElementById('status');
        this.speechStatus = document.getElementById('speech-status');
        
        this.stream = null;
        this.isRunning = false;
        this.autoSpeech = false;
        this.recognitionInterval = null;
        this.lastSpokenPrediction = ''; // Theo dõi từ vừa đọc
        
        this.speechSynth = window.speechSynthesis;
        this.voices = [];
        
        // Thiết lập kích thước canvas
        this.canvas.width = 640;
        this.canvas.height = 480;
        
        this.initializeEventListeners();
        this.loadVoices();
    }
    
    initializeEventListeners() {
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.stopBtn.addEventListener('click', () => this.stopCamera());
        this.resetBtn.addEventListener('click', () => this.resetRecognition());
        this.speakBtn.addEventListener('click', () => this.speakText());
        this.toggleSpeechBtn.addEventListener('click', () => this.toggleAutoSpeech());
    }
    
    loadVoices() {
        this.speechSynth.onvoiceschanged = () => {
            this.voices = this.speechSynth.getVoices();
            console.log('Voices loaded:', this.voices.length);
        };
    }
    
    async startCamera() {
        try {
            console.log('🚀 Starting camera for word recognition...');
            this.updateStatus('Đang khởi động camera...');
            
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user',
                    frameRate: { ideal: 30 }
                } 
            });
            
            this.video.srcObject = this.stream;
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.resetBtn.disabled = false;
            this.speakBtn.disabled = false;
            this.isRunning = true;
            
            this.video.onloadedmetadata = () => {
                this.updateStatus('Camera đã sẵn sàng. Đang nhận diện từ vựng...');
                this.startRecognition();
            };
            
        } catch (error) {
            console.error('❌ Lỗi khi truy cập camera:', error);
            this.updateStatus('Lỗi: Không thể truy cập camera');
            alert('Không thể truy cập camera. Vui lòng kiểm tra quyền truy cập và thử lại.');
        }
    }
    
    startRecognition() {
        console.log('🎯 Starting word recognition...');
        this.updateStatus('AI đang nhận diện từ vựng...');
        
        this.recognitionInterval = setInterval(() => {
            if (this.isRunning && this.video.readyState === this.video.HAVE_ENOUGH_DATA) {
                this.captureAndRecognize();
            }
        }, 500); // Process every 500ms
    }
    
    async captureAndRecognize() {
        try {
            // Vẽ video lên canvas với flip horizontal
            this.ctx.save();
            this.ctx.scale(-1, 1);
            this.ctx.drawImage(this.video, -this.canvas.width, 0, this.canvas.width, this.canvas.height);
            this.ctx.restore();
            
            const imageData = this.canvas.toDataURL('image/jpeg', 0.8);
            
            const response = await this.sendToServer(imageData);
            
            if (response.success) {
                this.updateResult(response.prediction, response.confidence);
                
                // Tự động phát âm nếu enabled VÀ độ tin cậy >= 60% VÀ từ mới khác từ cũ
                if (this.autoSpeech && 
                    response.confidence >= 60 && 
                    response.prediction !== this.lastSpokenPrediction &&
                    response.prediction !== '--' &&
                    response.prediction !== 'Lỗi nhận diện' &&
                    response.prediction !== 'Lỗi kết nối') {
                    
                    this.speakText();
                    this.lastSpokenPrediction = response.prediction;
                }
            } else {
                this.updateResult('Lỗi nhận diện', 0);
            }
            
        } catch (error) {
            console.error('❌ Lỗi trong captureAndRecognize:', error);
            this.updateResult('Lỗi kết nối', 0);
        }
    }
    
    async sendToServer(imageData) {
        try {
            const response = await fetch('/api/recognize/words/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'X-CSRFToken': this.getCSRFToken()
                },
                body: `image=${encodeURIComponent(imageData)}`
            });
            
            return await response.json();
            
        } catch (error) {
            console.error('❌ Lỗi kết nối server:', error);
            return { success: false, prediction: 'Lỗi kết nối server', confidence: 0 };
        }
    }
    
    updateResult(prediction, confidence) {
        this.result.textContent = prediction;
        this.confidence.textContent = `Độ tin cậy: ${confidence.toFixed(1)}%`;
        this.confidenceBar.style.width = `${confidence}%`;
        
        // Đổi màu thanh confidence và text
        if (confidence >= 80) {
            this.confidenceBar.style.background = 'linear-gradient(90deg, #28a745, #20c997)';
            this.result.className = 'display-4 fw-bold text-success mb-3 pulse-animation';
        } else if (confidence >= 60) {
            this.confidenceBar.style.background = 'linear-gradient(90deg, #ffc107, #fd7e14)';
            this.result.className = 'display-4 fw-bold text-warning mb-3 pulse-animation';
        } else if (confidence > 0) {
            this.confidenceBar.style.background = 'linear-gradient(90deg, #dc3545, #e83e8c)';
            this.result.className = 'display-4 fw-bold text-danger mb-3';
        } else {
            this.confidenceBar.style.background = '#e9ecef';
            this.result.className = 'display-4 fw-bold text-secondary mb-3';
        }
        
        // Thêm animation khi có kết quả mới
        if (confidence > 0 && prediction !== '--') {
            this.result.classList.add('pulse-animation');
            setTimeout(() => {
                this.result.classList.remove('pulse-animation');
            }, 500);
        }
        
        // Hiển thị trạng thái đọc tự động
        let statusMessage = `Đã nhận diện: ${prediction}`;
        if (this.autoSpeech && confidence >= 60 && prediction !== this.lastSpokenPrediction) {
            statusMessage += ' 🔊 Tự động đọc...';
        } else if (this.autoSpeech && confidence < 60) {
            statusMessage += ' ⚠️ Độ tin cậy thấp';
        }
        
        this.updateStatus(statusMessage);
    }
    
    speakText() {
        const text = this.result.textContent;
        
        if (text && text !== '--' && text !== 'Lỗi nhận diện' && text !== 'Lỗi kết nối') {
            // Dừng speech hiện tại nếu có
            this.speechSynth.cancel();
            
            const utterance = new SpeechSynthesisUtterance(text);
            
            // Cấu hình giọng đọc - dùng tiếng Anh cho từ vựng ASL
            utterance.rate = 0.8;    // Tốc độ chậm
            utterance.pitch = 1;     // Cao độ
            utterance.volume = 1;    // Âm lượng
            utterance.lang = 'en-US'; // Luôn dùng tiếng Anh cho từ vựng
            
            // Tìm giọng tiếng Anh tốt
            const englishVoice = this.voices.find(voice => 
                voice.lang.includes('en') && voice.name.includes('Female')
            ) || this.voices.find(voice => voice.lang.includes('en'));
            
            if (englishVoice) {
                utterance.voice = englishVoice;
            }
            
            // Xử lý sự kiện
            utterance.onstart = () => {
                this.speakBtn.innerHTML = '<i class="fas fa-volume-up me-2"></i>ĐANG ĐỌC...';
                this.speakBtn.disabled = true;
                this.updateStatus(`🔊 Đang đọc: ${text}`);
            };
            
            utterance.onend = () => {
                this.speakBtn.innerHTML = '<i class="fas fa-volume-up me-2"></i>ĐỌC KẾT QUẢ';
                this.speakBtn.disabled = false;
                this.updateStatus(`Đã đọc: ${text}`);
            };
            
            utterance.onerror = (event) => {
                console.error('Speech synthesis error:', event);
                this.speakBtn.innerHTML = '<i class="fas fa-volume-up me-2"></i>ĐỌC KẾT QUẢ';
                this.speakBtn.disabled = false;
                this.updateStatus('Lỗi phát âm');
            };
            
            this.speechSynth.speak(utterance);
            console.log(`🔊 Phát âm từ vựng: ${text}`);
        }
    }
    
    toggleAutoSpeech() {
        this.autoSpeech = !this.autoSpeech;
        
        if (this.autoSpeech) {
            this.speechStatus.textContent = 'BẬT';
            this.speechStatus.className = 'badge bg-success ms-2';
            this.toggleSpeechBtn.classList.remove('btn-outline-info');
            this.toggleSpeechBtn.classList.add('btn-info');
            this.updateStatus('Tự động phát âm đã BẬT (≥60%)');
        } else {
            this.speechStatus.textContent = 'TẮT';
            this.speechStatus.className = 'badge bg-secondary ms-2';
            this.toggleSpeechBtn.classList.remove('btn-info');
            this.toggleSpeechBtn.classList.add('btn-outline-info');
            this.updateStatus('Tự động phát âm đã TẮT');
            this.lastSpokenPrediction = ''; // Reset khi tắt auto speech
        }
    }
    
    resetRecognition() {
        console.log('🔄 Resetting word recognition...');
        
        // Reset biến theo dõi
        this.lastSpokenPrediction = '';
        
        // Gửi reset request đến server (nếu có endpoint reset)
        fetch('/api/recognize/words/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'X-CSRFToken': this.getCSRFToken()
            },
            body: 'reset=true'
        }).catch(error => {
            console.error('Reset error:', error);
        });
        
        // Reset UI
        this.updateResult('--', 0);
        this.updateStatus('Đã reset nhận diện');
        
        // Dừng speech nếu đang đọc
        this.speechSynth.cancel();
    }
    
    updateStatus(message) {
        if (this.status) {
            this.status.innerHTML = `<i class="fas fa-circle text-success me-2"></i>${message}`;
        }
    }
    
    getCSRFToken() {
        const name = 'csrftoken';
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
    
    stopCamera() {
        console.log('🛑 Stopping camera and word recognition...');
        
        if (this.recognitionInterval) {
            clearInterval(this.recognitionInterval);
        }
        
        // Dừng speech
        this.speechSynth.cancel();
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.video.srcObject = null;
        }
        
        // Reset UI state
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.resetBtn.disabled = true;
        this.speakBtn.disabled = true;
        this.isRunning = false;
        
        // Reset display
        this.updateResult('--', 0);
        this.confidenceBar.style.background = '#e9ecef';
        this.result.className = 'display-4 fw-bold text-success mb-3';
        this.updateStatus('Đã dừng nhận diện');
        
        // Reset tracking
        this.lastSpokenPrediction = '';
    }
}

// Khởi tạo khi trang được load
document.addEventListener('DOMContentLoaded', function() {
    console.log('📄 Page loaded, initializing ASL Words Recognizer...');
    new ASLWordsRecognizer();
});