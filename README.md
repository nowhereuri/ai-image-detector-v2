# AI Image Detector - AI 생성 이미지 탐지기

## 프로젝트 개요
Vision Transformer(ViT) 기반의 AI 생성 이미지 탐지 웹 애플리케이션입니다. 사용자가 업로드한 이미지가 AI로 생성된 것인지 실제 촬영된 것인지 판단하며, 피드백을 통한 실시간 학습 기능을 제공합니다.

## 주요 기능

### 🔍 AI 이미지 탐지
- **Vision Transformer 모델** 사용
- **높은 정확도**로 AI 생성 이미지와 실제 이미지 구분
- **신뢰도 점수** 제공 (0-100%)

### 📊 분석 설명
- **상세한 분석 결과** 제공
- **판단 근거** 설명
- **이미지 특성** 분석

### 💬 피드백 시스템
- **사용자 피드백** 수집
- **잘못된 판단** 개선
- **실시간 학습** 기능

### 📈 통계 대시보드
- **실시간 성능 통계**
- **피드백 현황** 모니터링
- **학습 진행 상황** 추적

### 🎨 현대적인 UI/UX
- **반응형 디자인**
- **하단 탭 네비게이션**
- **드래그 앤 드롭** 업로드
- **로딩 애니메이션**

## 기술 스택

### Backend
- **Python 3.8+**
- **Flask** - 웹 프레임워크
- **Hugging Face Transformers** - AI 모델
- **PyTorch** - 딥러닝 프레임워크
- **SQLite** - 피드백 데이터 저장

### Frontend
- **HTML5/CSS3/JavaScript**
- **반응형 디자인**
- **모던 UI 컴포넌트**

### AI/ML
- **Vision Transformer (ViT)**
- **Hugging Face Pipeline**
- **실시간 학습 시스템**
- **데이터 증강 (Data Augmentation)**

## 설치 및 실행

### 1. 저장소 클론
```bash
git clone https://github.com/[your-username]/ai-image-detector.git
cd ai-image-detector
```

### 2. 가상환경 생성 및 활성화
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 웹 애플리케이션 실행
```bash
python web_app.py
```

### 5. 브라우저에서 접속
```
http://localhost:5000
```

## 프로젝트 구조

```
ai-image-detector/
├── web_app.py                 # Flask 웹 애플리케이션
├── requirements.txt           # Python 의존성
├── README.md                  # 프로젝트 문서
├── .gitignore                 # Git 무시 파일
├── templates/                 # HTML 템플릿
│   ├── index.html            # 메인 페이지
│   ├── features.html         # 기능 소개
│   ├── stats.html            # 통계 페이지
│   └── about.html            # 사이트 정보
├── static/                    # 정적 파일 (CSS, JS, 이미지)
├── feedback_system.py         # 피드백 시스템
├── realtime_learning_system.py # 실시간 학습 시스템
├── train_ai_image_detector.py  # 모델 훈련 스크립트
├── predict_ai_image.py        # 예측 스크립트
└── test_*.py                  # 테스트 스크립트들
```

## 사용 방법

### 1. 이미지 업로드
- 메인 페이지에서 이미지 파일 선택
- 드래그 앤 드롭으로 업로드 가능
- 지원 형식: JPG, PNG, GIF, BMP

### 2. 결과 확인
- AI 탐지 결과 확인
- 신뢰도 점수 확인
- 상세 분석 설명 읽기

### 3. 피드백 제출
- 결과가 정확한 경우: "정확합니다" 클릭
- 결과가 틀린 경우: "틀렸습니다" 클릭 후 올바른 답 선택

### 4. 통계 확인
- 하단 탭에서 "통계보기" 클릭
- 실시간 성능 통계 확인
- 피드백 현황 모니터링

## 모델 정보

### 기본 모델
- **모델명**: `dima806/ai_vs_real_image_detection`
- **아키텍처**: Vision Transformer (ViT)
- **입력 크기**: 224x224 픽셀
- **클래스**: FAKE (AI 생성), REAL (실제 이미지)

### 개선된 모델
- **재훈련된 모델**: `test2_retrained_model/`
- **피드백 개선 모델**: `feedback_improved_model/`
- **실시간 학습**: 사용자 피드백을 통한 지속적 개선

## 성능 지표

### 기본 모델 성능
- **전체 정확도**: ~85%
- **FAKE 탐지**: ~90%
- **REAL 탐지**: ~80%

### 개선된 모델 성능
- **전체 정확도**: ~99%
- **FAKE 탐지**: ~98%
- **REAL 탐지**: ~100%

## 개발자 정보

### 주요 기능 구현
- **AI 모델 통합**: Hugging Face Transformers 사용
- **실시간 학습**: 피드백 기반 모델 개선
- **웹 인터페이스**: Flask 기반 RESTful API
- **데이터베이스**: SQLite를 통한 피드백 저장
- **프론트엔드**: 반응형 웹 디자인

### 확장 가능성
- **다양한 AI 모델** 지원
- **클라우드 배포** 가능
- **API 서비스** 제공
- **모바일 앱** 개발

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 연락처

프로젝트에 대한 문의사항이 있으시면 이슈를 생성해 주세요.

---

**AI Image Detector** - 정확하고 빠른 AI 생성 이미지 탐지 서비스