#!/bin/bash

# AI Image Detector 배포 스크립트

echo "🚀 AI Image Detector 배포 시작..."

# Docker 이미지 빌드
echo "📦 Docker 이미지 빌드 중..."
docker build -t ai-image-detector .

# 기존 컨테이너 중지 및 제거
echo "🛑 기존 컨테이너 정리 중..."
docker-compose down

# 새 컨테이너 시작
echo "▶️ 새 컨테이너 시작 중..."
docker-compose up -d

# 상태 확인
echo "✅ 배포 완료! 상태 확인 중..."
sleep 5
docker-compose ps

echo "🌐 웹 애플리케이션이 http://localhost:8000 에서 실행 중입니다."
echo "📊 로그 확인: docker-compose logs -f"
echo "🛑 중지: docker-compose down"

