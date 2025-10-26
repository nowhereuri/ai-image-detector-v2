# Heroku 배포 가이드

## 1. Heroku CLI 설치 및 로그인
```bash
# Windows에서 Heroku CLI 설치 (이미 완료됨)
# https://devcenter.heroku.com/articles/heroku-cli

# 로그인 (웹 브라우저에서 인증)
heroku login
```

## 2. Heroku 앱 생성
```bash
# 새 앱 생성
heroku apps:create ai-image-detector-v2

# 또는 기존 앱 사용
heroku git:remote -a ai-image-detector-v2
```

## 3. 환경변수 설정 (필요시)
```bash
# 예: API 키나 데이터베이스 URL 설정
heroku config:set FLASK_ENV=production
```

## 4. 배포
```bash
# Git에 변경사항 커밋
git add .
git commit -m "Prepare for Heroku deployment"

# Heroku에 배포
git push heroku main
```

## 5. 앱 실행
```bash
# 앱 열기
heroku open

# 로그 확인
heroku logs --tail
```

## 6. 문제 해결
```bash
# 앱 상태 확인
heroku ps

# 환경변수 확인
heroku config

# 앱 재시작
heroku restart
```

## 주의사항
- AI 모델 파일들은 용량이 크므로 Heroku의 파일 시스템 제한에 주의
- 필요시 외부 스토리지 서비스 사용 고려
- Heroku는 무료 플랜에서 30분 후 슬립 모드로 전환됨
