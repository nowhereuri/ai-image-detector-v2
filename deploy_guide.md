# AI Image Detector 웹사이트 배포 가이드

## 🚀 로컬에서 실행하기

### 1. 환경 설정
```bash
# 가상환경 생성
python -m venv ai_detector_env
ai_detector_env\Scripts\activate  # Windows
# source ai_detector_env/bin/activate  # Linux/Mac

# 패키지 설치
pip install -r requirements.txt
```

### 2. 웹 애플리케이션 실행
```bash
python web_app.py
```

### 3. 브라우저에서 접속
```
http://localhost:5000
```

## 🌐 클라우드 배포 옵션

### 옵션 1: Heroku (무료 티어)
1. **Heroku CLI 설치**
2. **Procfile 생성**:
```
web: python web_app.py
```
3. **배포**:
```bash
heroku create your-app-name
git push heroku main
```

### 옵션 2: Google Cloud Platform
1. **App Engine 설정**
2. **app.yaml 파일 생성**:
```yaml
runtime: python39
entrypoint: python web_app.py
```
3. **배포**:
```bash
gcloud app deploy
```

### 옵션 3: AWS EC2
1. **EC2 인스턴스 생성**
2. **Docker 컨테이너 사용**
3. **Dockerfile 생성**:
```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "web_app.py"]
```

## 💰 비용 예상

### 무료 옵션
- **Heroku**: 월 550시간 무료 (제한적)
- **Google Cloud**: $300 크레딧 제공
- **AWS**: 12개월 무료 티어

### 유료 옵션
- **Heroku**: $7/월부터
- **Google Cloud**: 사용량 기반
- **AWS**: $5-20/월 (인스턴스 크기에 따라)

## 🔧 성능 최적화

### 1. 모델 최적화
```python
# GPU 사용 (CUDA 환경)
ai_pipe = pipeline('image-classification', model=model_name, device=0)

# 배치 처리
def predict_batch(images):
    return ai_pipe(images)
```

### 2. 캐싱 추가
```python
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@cache.memoize(timeout=300)
def predict_image_cached(image_hash):
    return predict_image(image_path)
```

### 3. CDN 사용
- CloudFlare 무료 플랜
- 이미지 최적화
- 전역 배포

## 📊 모니터링 및 분석

### 1. 사용량 추적
```python
import logging
from datetime import datetime

@app.route('/upload', methods=['POST'])
def upload_file():
    # 사용량 로깅
    logging.info(f"Upload at {datetime.now()}")
    # ... 기존 코드
```

### 2. Google Analytics 추가
```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
```

## 🛡️ 보안 고려사항

### 1. 파일 업로드 보안
```python
# 파일 크기 제한
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 파일 형식 검증
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
```

### 2. Rate Limiting
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/upload', methods=['POST'])
@limiter.limit("10 per minute")
def upload_file():
    # ... 기존 코드
```

## 📈 수익화 전략

### 1. 프리미엄 기능
- 고해상도 이미지 분석
- 배치 처리
- API 액세스
- 상세 분석 리포트

### 2. 광고 수익
- Google AdSense
- 관련 서비스 광고

### 3. API 서비스
- 개발자용 API 제공
- 사용량 기반 과금

## 🎯 마케팅 전략

### 1. SEO 최적화
- 메타 태그 추가
- 구조화된 데이터
- 모바일 최적화

### 2. 소셜 미디어
- Twitter, Instagram 공유 기능
- 바이럴 마케팅

### 3. 콘텐츠 마케팅
- AI 관련 블로그 포스트
- 사용 사례 공유

