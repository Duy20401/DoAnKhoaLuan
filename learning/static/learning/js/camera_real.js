// learning/static/learning/js/camera_real.js
class ASLRealRecognizer {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.startBtn = document.getElementById('start-btn');
        this.stopBtn = document.getElementById('stop-btn');
        this.speakBtn = document.getElementById('speak-btn');
        this.toggleSpeechBtn = document.getElementById('toggle-speech-btn');
        this.testSpeechBtn = document.getElementById('test-speech-btn');
        this.result = document.getElementById('prediction-result');
        this.confidence = document.getElementById('confidence');
        this.confidenceBar = document.getElementById('confidence-bar');
        this.status = document.getElementById('status');
        this.speechStatus = document.getElementById('speech-status');
        
        this.stream = null;
        this.isRunning = false;
        this.autoSpeech = false;
        this.lastPrediction = '';
        this.recognitionInterval = null;
        
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
        this.speakBtn.addEventListener('click', () => this.speakText());
        this.toggleSpeechBtn.addEventListener('click', () => this.toggleAutoSpeech());
        this.testSpeechBtn.addEventListener('click', () => this.testSpeech());
    }
    
    loadVoices() {
        this.speechSynth.onvoiceschanged = () => {
            this.voices = this.speechSynth.getVoices();
            console.log('Voices loaded:', this.voices.length);
        };
    }
    
    async startCamera() {
        try {
            console.log('🚀 Starting camera and AI recognition...');
            this.updateStatus('Đang khởi động camera...');
            
            // Sử dụng facingMode: 'user' cho camera trước và thêm constraints
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user', // Luôn dùng camera trước
                    frameRate: { ideal: 30 }
                } 
            });
            
            this.video.srcObject = this.stream;
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.speakBtn.disabled = false;
            this.isRunning = true;
            
            // Đợi video ready
            this.video.onloadedmetadata = () => {
                this.updateStatus('Camera đã sẵn sàng. Đang nhận diện...');
                this.startRecognition();
            };
            
        } catch (error) {
            console.error('❌ Lỗi khi truy cập camera:', error);
            this.updateStatus('Lỗi: Không thể truy cập camera');
            alert('Không thể truy cập camera. Vui lòng kiểm tra quyền truy cập và thử lại.');
        }
    }
    
    startRecognition() {
        console.log('🎯 Starting real-time AI recognition...');
        this.updateStatus('AI đang nhận diện...');
        
        this.recognitionInterval = setInterval(() => {
            if (this.isRunning && this.video.readyState === this.video.HAVE_ENOUGH_DATA) {
                this.captureAndRecognize();
            }
        }, 500);
    }
    
    async captureAndRecognize() {
        try {
            // Vẽ video lên canvas với flip horizontal để khớp với video
            this.ctx.save();
            this.ctx.scale(-1, 1); // Flip horizontal
            this.ctx.drawImage(this.video, -this.canvas.width, 0, this.canvas.width, this.canvas.height);
            this.ctx.restore();
            
            const imageData = this.canvas.toDataURL('image/jpeg', 0.8);
            
            const response = await this.sendToServer(imageData);
            
            if (response.success) {
                this.updateResult(response.prediction, response.confidence);
                
                // Tự động phát âm nếu enabled
                if (this.autoSpeech && response.confidence > 70 && 
                    response.prediction !== this.lastPrediction &&
                    response.prediction.length === 1) {
                    this.speakText();
                }
                
                this.lastPrediction = response.prediction;
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
            const response = await fetch('/api/recognize/', {
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
        
        // Đổi màu thanh confidence
        if (confidence >= 80) {
            this.confidenceBar.style.background = 'linear-gradient(90deg, #28a745, #20c997)';
            this.result.className = 'display-1 fw-bold text-success mb-3 pulse-animation';
        } else if (confidence >= 60) {
            this.confidenceBar.style.background = 'linear-gradient(90deg, #ffc107, #fd7e14)';
            this.result.className = 'display-1 fw-bold text-warning mb-3 pulse-animation';
        } else if (confidence > 0) {
            this.confidenceBar.style.background = 'linear-gradient(90deg, #dc3545, #e83e8c)';
            this.result.className = 'display-1 fw-bold text-danger mb-3';
        } else {
            this.confidenceBar.style.background = '#e9ecef';
            this.result.className = 'display-1 fw-bold text-secondary mb-3';
        }
        
        this.result.classList.add('pulse-animation');
        setTimeout(() => {
            this.result.classList.remove('pulse-animation');
        }, 500);
        
        this.updateStatus(`Đã nhận diện: ${prediction}`);
    }
    
    speakText() {
        const text = this.result.textContent;
        
        if (text && text !== '--' && text !== 'Lỗi nhận diện' && text !== 'Lỗi kết nối') {
            this.speechSynth.cancel();
            
            const utterance = new SpeechSynthesisUtterance(text);
            
            utterance.rate = 0.8;
            utterance.pitch = 1;
            utterance.volume = 1;
            
            const vietnameseVoice = this.voices.find(voice => 
                voice.lang.includes('vi') || voice.lang.includes('VN')
            );
            
            if (vietnameseVoice) {
                utterance.voice = vietnameseVoice;
                utterance.lang = 'vi-VN';
            } else {
                utterance.lang = 'en-US';
            }
            
            utterance.onstart = () => {
                this.speakBtn.innerHTML = '<i class="fas fa-volume-up me-2"></i>ĐANG ĐỌC...';
                this.speakBtn.disabled = true;
            };
            
            utterance.onend = () => {
                this.speakBtn.innerHTML = '<i class="fas fa-volume-up me-2"></i>ĐỌC KẾT QUẢ';
                this.speakBtn.disabled = false;
            };
            
            this.speechSynth.speak(utterance);
            console.log(`🔊 Phát âm: ${text}`);
        }
    }
    
    toggleAutoSpeech() {
        this.autoSpeech = !this.autoSpeech;
        
        if (this.autoSpeech) {
            this.speechStatus.textContent = 'BẬT';
            this.speechStatus.className = 'badge bg-success ms-2';
            this.toggleSpeechBtn.classList.remove('btn-outline-info');
            this.toggleSpeechBtn.classList.add('btn-info');
            this.updateStatus('Tự động phát âm đã BẬT');
        } else {
            this.speechStatus.textContent = 'TẮT';
            this.speechStatus.className = 'badge bg-secondary ms-2';
            this.toggleSpeechBtn.classList.remove('btn-info');
            this.toggleSpeechBtn.classList.add('btn-outline-info');
            this.updateStatus('Tự động phát âm đã TẮT');
        }
    }
    
    testSpeech() {
        const testText = "Xin chào! Hệ thống nhận diện ASL đã sẵn sàng";
        const utterance = new SpeechSynthesisUtterance(testText);
        
        utterance.rate = 0.8;
        utterance.volume = 1;
        
        this.speechSynth.speak(utterance);
        this.updateStatus('Đang kiểm tra âm thanh...');
        
        utterance.onend = () => {
            this.updateStatus('Kiểm tra âm thanh hoàn tất');
        };
    }
    
    updateStatus(message) {
        if (this.status) {
            this.status.innerHTML = `<i class="fas fa-circle text-primary me-2"></i>${message}`;
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
        console.log('🛑 Stopping camera and recognition...');
        
        if (this.recognitionInterval) {
            clearInterval(this.recognitionInterval);
        }
        
        this.speechSynth.cancel();
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.video.srcObject = null;
        }
        
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.speakBtn.disabled = true;
        this.isRunning = false;
        
        this.result.textContent = '--';
        this.confidence.textContent = 'Độ tin cậy: 0%';
        this.confidenceBar.style.width = '0%';
        this.confidenceBar.style.background = '#e9ecef';
        this.result.className = 'display-1 fw-bold text-primary mb-3';
        this.updateStatus('Đã dừng nhận diện');
    }
}

document.addEventListener('DOMContentLoaded', function() {
    console.log('📄 Page loaded, initializing ASL Real Recognizer with Speech...');
    new ASLRealRecognizer();
});