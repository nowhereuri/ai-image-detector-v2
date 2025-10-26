# 🚀 AI Image Detector 배포 가이드

## 📋 배포 방법별 가이드

### 1. 🐳 Docker 배포 (추천)

#### 로컬 Docker 실행
```bash
# Docker 이미지 빌드 및 실행
docker build -t ai-image-detector .
docker run -p 8000:8000 ai-image-detector

# 또는 Docker Compose 사용
docker-compose up -d
```

#### 클라우드 Docker 배포
- **AWS ECS**: ECS 클러스터에 컨테이너 배포
- **Google Cloud Run**: 서버리스 컨테이너 실행
- **Azure Container Instances**: 간단한 컨테이너 배포

### 2. ☁️ 클라우드 플랫폼 배포

#### Heroku
```bash
# Heroku CLI 설치 후
heroku create your-app-name
git add .
git commit -m "Deploy AI Image Detector"
git push heroku main
```

#### AWS EC2
```bash
# EC2 인스턴스에서
sudo apt update
sudo apt install python3-pip nginx
pip3 install -r requirements.txt
pip3 install gunicorn

# Gunicorn으로 실행
gunicorn -w 4 -b 0.0.0.0:8000 web_app:app
```

#### Google Cloud Platform
```bash
# App Engine 배포
gcloud app deploy

# 또는 Compute Engine에서 Docker 실행
gcloud compute instances create-with-container ai-detector \
  --container-image=ai-image-detector
```

### 3. 🖥️ VPS/서버 호스팅

#### Ubuntu/Debian 서버
```bash
# 1. 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# 2. Python 및 필수 패키지 설치
sudo apt install python3-pip python3-venv nginx

# 3. 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# 4. 의존성 설치
pip install -r requirements.txt
pip install gunicorn

# 5. 애플리케이션 실행
gunicorn -w 4 -b 127.0.0.1:8000 web_app:app

# 6. Nginx 설정
sudo cp nginx.conf /etc/nginx/sites-available/ai-detector
sudo ln -s /etc/nginx/sites-available/ai-detector /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# 7. Systemd 서비스 등록
sudo cp systemd/ai-image-detector.service /etc/systemd/system/
sudo systemctl enable ai-image-detector
sudo systemctl start ai-image-detector
```

## 🔧 환경 설정

### 환경 변수
```bash
export FLASK_ENV=production
export FLASK_APP=web_app.py
export PYTHONPATH=/opt/ai-image-detector
```

### 보안 설정
- SSL 인증서 설치 (Let's Encrypt)
- 방화벽 설정 (포트 80, 443만 개방)
- 정기적인 보안 업데이트

## 📊 모니터링

### 로그 확인
```bash
# Docker
docker-compose logs -f

# Systemd
sudo journalctl -u ai-image-detector -f

# Nginx
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### 성능 모니터링
- CPU/메모리 사용량 모니터링
- 응답 시간 측정
- 에러율 추적

## 🔄 업데이트 및 백업

### 업데이트
```bash
# Docker
docker-compose pull
docker-compose up -d

# 일반 서버
git pull
sudo systemctl restart ai-image-detector
```

### 백업
```bash
# 데이터베이스 백업
cp feedback.db feedback_backup_$(date +%Y%m%d).db

# 업로드 파일 백업
tar -czf uploads_backup_$(date +%Y%m%d).tar.gz uploads/
```

## 🚨 문제 해결

### 일반적인 문제들
1. **포트 충돌**: 다른 포트 사용
2. **메모리 부족**: 워커 수 조정
3. **파일 업로드 실패**: Nginx client_max_body_size 설정
4. **모델 로딩 실패**: 충분한 메모리 할당

### 성능 최적화
- Gunicorn 워커 수 조정 (CPU 코어 수 * 2 + 1)
- Nginx 캐싱 설정
- 정적 파일 CDN 사용
- 데이터베이스 최적화

## 📞 지원

배포 중 문제가 발생하면:
1. 로그 파일 확인
2. 시스템 리소스 상태 점검
3. 네트워크 연결 확인
4. 방화벽 설정 검토

