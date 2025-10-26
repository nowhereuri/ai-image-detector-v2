#!/bin/bash

echo "========================================"
echo "AI Image Detector Model VIT 설치 스크립트"
echo "========================================"
echo

echo "Python 버전 확인 중..."
python3 --version
if [ $? -ne 0 ]; then
    echo "오류: Python3가 설치되지 않았거나 PATH에 추가되지 않았습니다."
    echo "Python 3.8 이상을 설치해주세요."
    exit 1
fi

echo
echo "가상환경 생성 중..."
python3 -m venv ai_image_detector
if [ $? -ne 0 ]; then
    echo "오류: 가상환경 생성에 실패했습니다."
    exit 1
fi

echo
echo "가상환경 활성화 중..."
source ai_image_detector/bin/activate

echo
echo "pip 업그레이드 중..."
pip install --upgrade pip

echo
echo "필요한 패키지 설치 중..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "오류: 패키지 설치에 실패했습니다."
    exit 1
fi

echo
echo "========================================"
echo "설치가 완료되었습니다!"
echo "========================================"
echo
echo "사용법:"
echo "1. 훈련: python train_ai_image_detector.py --data_path '데이터경로'"
echo "2. 예측: python predict_ai_image.py --model_path '모델경로' --image_path '이미지경로'"
echo
echo "자세한 사용법은 README.md 파일을 참고하세요."
echo
echo "가상환경을 활성화하려면 다음 명령어를 실행하세요:"
echo "source ai_image_detector/bin/activate"
echo

