# AI Image Detector - 피드백 기반 모델 개선 가이드

## 🎯 개요

이 시스템은 사용자 피드백을 수집하여 AI 모델을 지속적으로 개선하는 기능을 제공합니다. 다양한 비율의 이미지에 대한 성능을 향상시키기 위해 설계되었습니다.

## 🔄 피드백 수집 프로세스

### 1. 웹 인터페이스에서 피드백 제공

1. **이미지 업로드**: 웹사이트에서 이미지를 업로드
2. **AI 예측 확인**: 모델의 예측 결과 확인
3. **피드백 제공**: 
   - ✅ **정확합니다**: 예측이 맞는 경우
   - ❌ **틀렸습니다**: 예측이 틀린 경우
     - 올바른 답 선택: 📷 실제 이미지 또는 🎨 AI 생성 이미지

### 2. 피드백 데이터 저장

- 모든 피드백은 SQLite 데이터베이스에 저장
- 이미지 해시를 사용하여 중복 방지
- 타임스탬프와 함께 저장

## 🛠️ 시스템 구성 요소

### 1. 피드백 수집 시스템 (`feedback_system.py`)

```python
from feedback_system import FeedbackSystem

# 피드백 시스템 초기화
feedback_system = FeedbackSystem()

# 피드백 추가
feedback_system.add_feedback(
    image_path="path/to/image.jpg",
    predicted_label="FAKE",
    predicted_confidence=0.85,
    user_feedback="REAL",  # 사용자가 제공한 올바른 답
    is_correct=False
)

# 통계 조회
stats = feedback_system.get_feedback_stats()
print(f"정확도: {stats['accuracy']:.1f}%")
```

### 2. 적응형 전처리 시스템 (`adaptive_preprocessing.py`)

```python
from adaptive_preprocessing import AdaptivePreprocessor

# 전처리기 초기화
preprocessor = AdaptivePreprocessor(target_size=32)

# 이미지 전처리
processed_image = preprocessor.adaptive_preprocessing(image)

# 다양한 전처리 방법
center_cropped = preprocessor.center_crop_resize(image)
padded = preprocessor.padding_resize(image)
multi_scaled = preprocessor.multi_scale_resize(image)
```

### 3. 재훈련 시스템 (`retrain_with_feedback.py`)

```bash
# 기본 재훈련
python retrain_with_feedback.py

# 고급 옵션
python retrain_with_feedback.py \
    --model_name "dima806/ai_vs_real_image_detection" \
    --output_dir "improved_model" \
    --epochs 2 \
    --feedback_ratio 0.3
```

## 📊 피드백 데이터 관리

### 1. 피드백 통계 확인

```python
# 웹 API를 통한 통계 조회
GET /feedback/stats

# 응답 예시
{
    "total_feedback": 150,
    "correct_predictions": 120,
    "incorrect_predictions": 30,
    "unprocessed_feedback": 25,
    "accuracy": 80.0
}
```

### 2. 피드백 데이터 내보내기

```python
# 훈련용 데이터로 내보내기
POST /feedback/export

# 응답 예시
{
    "message": "25개의 피드백이 훈련 데이터로 내보내졌습니다.",
    "exported_count": 25
}
```

## 🔧 모델 개선 워크플로우

### 1단계: 피드백 수집
- 사용자들이 웹 인터페이스를 통해 피드백 제공
- 최소 50-100개의 피드백 수집 권장

### 2단계: 데이터 검증
```python
# 피드백 품질 확인
stats = feedback_system.get_feedback_stats()
if stats['total_feedback'] < 50:
    print("더 많은 피드백이 필요합니다.")
```

### 3단계: 모델 재훈련
```bash
# 피드백 데이터를 활용한 재훈련
python retrain_with_feedback.py --epochs 2 --feedback_ratio 0.3
```

### 4단계: 성능 평가
- 재훈련된 모델의 성능 테스트
- 기존 모델과 비교 분석

### 5단계: 모델 배포
- 성능이 개선된 경우 새 모델로 교체
- 웹 애플리케이션에서 새 모델 사용

## 📈 성능 모니터링

### 1. 정확도 추적
- 피드백 정확도 모니터링
- 시간에 따른 성능 변화 추적

### 2. 오분류 패턴 분석
```python
# 오분류된 피드백 분석
unprocessed = feedback_system.get_unprocessed_feedback()
for feedback in unprocessed:
    print(f"이미지: {feedback[1]}")
    print(f"예측: {feedback[2]}")
    print(f"사용자 피드백: {feedback[3]}")
```

### 3. 이미지 비율별 성능 분석
- 다양한 비율의 이미지에 대한 성능 측정
- 비율별 오분류 패턴 파악

## 🎯 최적화 팁

### 1. 피드백 품질 향상
- 명확한 피드백 가이드라인 제공
- 사용자 교육 및 예시 제공

### 2. 데이터 균형 유지
- REAL/FAKE 피드백 비율 모니터링
- 편향된 피드백 방지

### 3. 전처리 방법 최적화
- 이미지 비율에 따른 적응형 전처리
- 다양한 전처리 방법 실험

## 🚀 고급 기능

### 1. 앙상블 예측
```python
# 여러 전처리 방법의 결과를 결합
ensemble_results = preprocessor.ensemble_preprocessing(image)
```

### 2. 자동 피드백 검증
- 신뢰도가 낮은 예측에 대한 자동 피드백 요청
- 일관성 있는 피드백 패턴 감지

### 3. A/B 테스트
- 기존 모델과 개선된 모델의 성능 비교
- 점진적 모델 배포

## 📝 모범 사례

### 1. 피드백 수집
- **최소 피드백 수**: 50개 이상
- **다양성 확보**: 다양한 비율의 이미지
- **품질 관리**: 명확하고 일관된 피드백

### 2. 재훈련 주기
- **초기**: 매주 재훈련
- **안정화 후**: 월 1-2회 재훈련
- **성능 모니터링**: 지속적인 성능 추적

### 3. 모델 배포
- **단계적 배포**: 소규모 테스트 후 전체 배포
- **롤백 계획**: 문제 발생 시 이전 모델로 복구
- **성능 모니터링**: 배포 후 성능 지속 추적

## 🔍 문제 해결

### 1. 피드백 부족
- 사용자 참여 유도 캠페인
- 인센티브 프로그램 도입
- 피드백 UI/UX 개선

### 2. 성능 저하
- 피드백 품질 검토
- 전처리 방법 재검토
- 모델 파라미터 조정

### 3. 데이터 불균형
- 피드백 균형 모니터링
- 가중치 조정
- 추가 데이터 수집

---

**참고**: 이 시스템은 지속적인 개선을 위해 설계되었습니다. 정기적인 모니터링과 조정을 통해 최적의 성능을 유지하세요.

