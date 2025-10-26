# AI Image Detector - 배포 가이드

## 🚀 배포 옵션

### 1. Heroku (추천)
Flask 애플리케이션에 가장 적합한 플랫폼입니다.

#### 배포 단계:
1. **Heroku CLI 설치**
```bash
# Windows
winget install Heroku.HerokuCLI

# 또는 공식 사이트에서 다운로드
# https://devcenter.heroku.com/articles/heroku-cli
```

2. **Heroku 로그인 및 앱 생성**
```bash
heroku login
heroku create ai-image-detector-app
```

3. **Procfile 생성**
```
web: gunicorn web_app:app
```

4. **배포**
```bash
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

### 2. Railway
현대적이고 사용하기 쉬운 배포 플랫폼입니다.

#### 배포 단계:
1. **Railway 계정 생성**: https://railway.app
2. **GitHub 연동**: 저장소 연결
3. **자동 배포**: 코드 푸시 시 자동 배포

### 3. Render
무료 티어가 있는 클라우드 플랫폼입니다.

#### 배포 단계:
1. **Render 계정 생성**: https://render.com
2. **Web Service 생성**
3. **GitHub 저장소 연결**
4. **빌드 명령어 설정**: `pip install -r requirements.txt`
5. **시작 명령어 설정**: `gunicorn web_app:app`

### 4. Google Cloud Platform (GCP)
Google의 클라우드 플랫폼입니다.

#### 배포 단계:
1. **Cloud Run 사용**
2. **Docker 컨테이너 배포**
3. **자동 스케일링**

### 5. AWS (Amazon Web Services)
가장 널리 사용되는 클라우드 플랫폼입니다.

#### 배포 옵션:
- **Elastic Beanstalk**: 간편한 배포
- **ECS (Elastic Container Service)**: Docker 컨테이너
- **Lambda**: 서버리스 (제한적)

## 📋 배포 전 준비사항

### 1. requirements.txt 업데이트
```txt
Flask>=2.0.0
gunicorn>=20.0.0
torch>=1.9.0
transformers>=4.20.0
Pillow>=8.0.0
scikit-learn>=1.0.0
imbalanced-learn>=0.8.0
datasets>=2.0.0
accelerate>=0.20.0
```

### 2. 환경 변수 설정
```bash
# AI 모델 설정
MODEL_NAME=dima806/ai_vs_real_image_detection
DEBUG=False
FLASK_ENV=production
```

### 3. 정적 파일 최적화
- 이미지 압축
- CSS/JS 최적화
- CDN 사용 고려

## 🔧 Netlify 대안 (정적 사이트용)

현재 프로젝트는 Flask 백엔드가 필요하므로 Netlify는 적합하지 않습니다. 하지만 정적 프론트엔드만 배포하려면:

### Netlify 배포 (정적 파일만)
1. **빌드 명령어**: `npm run build` (만약 Node.js 사용)
2. **배포 폴더**: `dist/` 또는 `build/`
3. **GitHub 연동**: 자동 배포 설정

## 💡 추천 배포 플랫폼

### 🥇 1순위: Heroku
- **장점**: Flask 최적화, 간편한 배포, 무료 티어
- **단점**: 무료 티어 제한 (월 550시간)

### 🥈 2순위: Railway
- **장점**: 현대적 UI, 자동 배포, GitHub 연동
- **단점**: 무료 티어 제한

### 🥉 3순위: Render
- **장점**: 무료 티어, 간편한 설정
- **단점**: 성능 제한

## 🚀 즉시 배포 가능한 방법

### Heroku 빠른 배포:
1. **Heroku CLI 설치**
2. **다음 명령어 실행**:
```bash
# Heroku 앱 생성
heroku create ai-image-detector-app

# Procfile 생성
echo "web: gunicorn web_app:app" > Procfile

# 배포
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

### Railway 빠른 배포:
1. **https://railway.app 접속**
2. **GitHub 로그인**
3. **저장소 선택**: `nowhereuri/ai-image-detector-v2`
4. **자동 배포 완료**

## 📞 지원

배포 과정에서 문제가 발생하면:
1. **GitHub Issues**: 버그 리포트
2. **플랫폼 문서**: 각 플랫폼의 공식 가이드 참조
3. **커뮤니티**: Stack Overflow, Reddit 등

---

**추천**: Heroku 또는 Railway를 사용하여 빠르게 배포하세요!