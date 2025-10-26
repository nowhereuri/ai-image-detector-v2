#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
과신(Overconfidence) 문제 분석 및 해결
"""

import os
import numpy as np
from PIL import Image
from transformers import pipeline
import matplotlib.pyplot as plt
from feedback_system import FeedbackSystem

def analyze_overconfidence_cases():
    """과신 사례 분석"""
    
    print("=== 과신(Overconfidence) 문제 분석 ===")
    
    # 피드백 시스템에서 과신 사례 찾기
    feedback_system = FeedbackSystem()
    
    # 모든 피드백 조회
    conn = feedback_system.db_path
    import sqlite3
    
    conn = sqlite3.connect(feedback_system.db_path)
    cursor = conn.cursor()
    
    # 높은 신뢰도이지만 틀린 예측들 조회
    cursor.execute('''
        SELECT image_path, predicted_label, predicted_confidence, user_feedback, is_correct
        FROM feedback 
        WHERE predicted_confidence > 0.8 AND is_correct = 0
        ORDER BY predicted_confidence DESC
    ''')
    
    overconfidence_cases = cursor.fetchall()
    conn.close()
    
    print(f"과신 사례 수: {len(overconfidence_cases)}")
    
    if overconfidence_cases:
        print("\n과신 사례들:")
        print("=" * 80)
        
        for i, (image_path, predicted_label, confidence, user_feedback, is_correct) in enumerate(overconfidence_cases[:5]):
            print(f"{i+1}. 이미지: {os.path.basename(image_path)}")
            print(f"   예측: {predicted_label} (신뢰도: {confidence:.4f} = {confidence*100:.1f}%)")
            print(f"   실제: {user_feedback}")
            print(f"   결과: {'정확' if is_correct else '틀림'}")
            print()
    else:
        print("과신 사례가 데이터베이스에 없습니다.")
        print("테스트를 위해 가상의 과신 사례를 분석합니다.")
        
        # 가상의 과신 사례 분석
        analyze_hypothetical_overconfidence()

def analyze_hypothetical_overconfidence():
    """가상의 과신 사례 분석"""
    
    print("\n=== 가상의 과신 사례 분석 ===")
    
    # 과신 사례 시나리오들
    scenarios = [
        {
            "name": "REAL 이미지를 FAKE로 높은 신뢰도로 오분류",
            "predicted": "FAKE",
            "actual": "REAL", 
            "confidence": 0.92,
            "description": "실제 사진을 AI 생성으로 92% 확신하며 오분류"
        },
        {
            "name": "FAKE 이미지를 REAL로 높은 신뢰도로 오분류", 
            "predicted": "REAL",
            "actual": "FAKE",
            "confidence": 0.89,
            "description": "AI 생성 이미지를 실제 사진으로 89% 확신하며 오분류"
        },
        {
            "name": "매우 높은 신뢰도로 완전히 틀린 예측",
            "predicted": "FAKE", 
            "actual": "REAL",
            "confidence": 0.95,
            "description": "95% 확신하며 완전히 틀린 예측"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  예측: {scenario['predicted']} (신뢰도: {scenario['confidence']*100:.0f}%)")
        print(f"  실제: {scenario['actual']}")
        print(f"  설명: {scenario['description']}")
        
        # 과신 정도 계산
        overconfidence_level = scenario['confidence'] - 0.5  # 50% 기준
        print(f"  과신 정도: {overconfidence_level:.2f} ({overconfidence_level*100:.0f}%)")

def identify_overconfidence_causes():
    """과신 원인 분석"""
    
    print("\n=== 과신 원인 분석 ===")
    
    causes = [
        {
            "cause": "훈련 데이터 편향",
            "description": "특정 유형의 이미지에 과도하게 노출되어 편향된 학습",
            "example": "REAL 이미지가 특정 스타일(예: 인물 사진)에 편중되어 다른 스타일을 FAKE로 오인",
            "solution": "다양한 스타일의 훈련 데이터 추가"
        },
        {
            "cause": "모델 복잡도 부족",
            "description": "모델이 복잡한 패턴을 제대로 학습하지 못함",
            "example": "단순한 특징만 보고 판단하여 복잡한 실제 이미지를 FAKE로 오분류",
            "solution": "더 복잡한 모델 아키텍처 사용 또는 앙상블 모델"
        },
        {
            "cause": "전처리 문제",
            "description": "이미지 전처리 과정에서 중요한 정보 손실",
            "example": "중앙 크롭으로 인해 중요한 특징이 잘려나가 잘못된 판단",
            "solution": "다양한 전처리 방법 시도 및 앙상블"
        },
        {
            "cause": "클래스 불균형",
            "description": "훈련 데이터의 클래스 불균형으로 인한 편향",
            "example": "FAKE 이미지가 많아서 모든 이미지를 FAKE로 예측하는 경향",
            "solution": "데이터 균형 조정 및 가중치 조정"
        },
        {
            "cause": "도메인 차이",
            "description": "훈련 데이터와 실제 사용 데이터 간의 차이",
            "example": "훈련 데이터는 32x32 크기인데 실제 사용은 다양한 크기",
            "solution": "실제 사용 환경과 유사한 데이터로 재훈련"
        }
    ]
    
    for i, cause in enumerate(causes, 1):
        print(f"{i}. {cause['cause']}")
        print(f"   설명: {cause['description']}")
        print(f"   예시: {cause['example']}")
        print(f"   해결책: {cause['solution']}")
        print()

def overconfidence_detection_strategy():
    """과신 탐지 전략"""
    
    print("\n=== 과신 탐지 전략 ===")
    
    print("1. 신뢰도 임계값 조정:")
    print("   ```python")
    print("   # 기존: 90% 이상이면 높은 신뢰도")
    print("   if confidence >= 0.9:")
    print("       return 'high_confidence'")
    print("   ")
    print("   # 개선: 95% 이상만 높은 신뢰도로 간주")
    print("   if confidence >= 0.95:")
    print("       return 'high_confidence'")
    print("   elif confidence >= 0.8:")
    print("       return 'medium_confidence'")
    print("   else:")
    print("       return 'low_confidence'")
    print("   ```")
    print()
    
    print("2. 앙상블 예측:")
    print("   ```python")
    print("   def ensemble_prediction(image):")
    print("       # 여러 전처리 방법으로 예측")
    print("       predictions = []")
    print("       for preprocess_method in methods:")
    print("           processed = preprocess_method(image)")
    print("           pred = model(processed)")
    print("           predictions.append(pred)")
    print("       ")
    print("       # 예측 결과 통합")
    print("       final_prediction = combine_predictions(predictions)")
    print("       return final_prediction")
    print("   ```")
    print()
    
    print("3. 불확실성 정량화:")
    print("   ```python")
    print("   def quantify_uncertainty(image):")
    print("       # 여러 번 예측하여 불확실성 측정")
    print("       predictions = []")
    print("       for _ in range(10):  # 10번 예측")
    print("           pred = model(image)")
    print("           predictions.append(pred)")
    print("       ")
    print("       # 예측 분산 계산")
    print("       uncertainty = np.var(predictions)")
    print("       return uncertainty")
    print("   ```")

def overconfidence_mitigation():
    """과신 완화 방법"""
    
    print("\n=== 과신 완화 방법 ===")
    
    print("1. 온도 스케일링 (Temperature Scaling):")
    print("   ```python")
    print("   def temperature_scaling(logits, temperature=2.0):")
    print("       # 온도 파라미터로 신뢰도 조정")
    print("       scaled_logits = logits / temperature")
    print("       probabilities = softmax(scaled_logits)")
    print("       return probabilities")
    print("   ```")
    print()
    
    print("2. 라벨 스무딩 (Label Smoothing):")
    print("   ```python")
    print("   def label_smoothing(labels, smoothing=0.1):")
    print("       # 하드 라벨을 소프트 라벨로 변환")
    print("       num_classes = len(labels)")
    print("       smoothed = (1 - smoothing) * labels + smoothing / num_classes")
    print("       return smoothed")
    print("   ```")
    print()
    
    print("3. 앙상블 불확실성:")
    print("   ```python")
    print("   def ensemble_uncertainty(image):")
    print("       # 여러 모델의 예측 결합")
    print("       model_predictions = []")
    print("       for model in models:")
    print("           pred = model(image)")
    print("           model_predictions.append(pred)")
    print("       ")
    print("       # 예측 평균과 분산 계산")
    print("       mean_pred = np.mean(model_predictions, axis=0)")
    print("       uncertainty = np.var(model_predictions, axis=0)")
    print("       return mean_pred, uncertainty")
    print("   ```")

def practical_overconfidence_handling():
    """실제 과신 처리 방법"""
    
    print("\n=== 실제 과신 처리 방법 ===")
    
    print("1. 웹 애플리케이션에서의 처리:")
    print("   ```python")
    print("   def handle_overconfidence(prediction, confidence):")
    print("       if confidence > 0.9:")
    print("           # 매우 높은 신뢰도는 경고 표시")
    print("           return {")
    print("               'prediction': prediction,")
    print("               'confidence': confidence,")
    print("               'warning': '매우 높은 신뢰도입니다. 결과를 신중히 검토해주세요.',")
    print("               'recommendation': '피드백을 제공해주시면 모델 개선에 도움이 됩니다.'")
    print("           }")
    print("   ```")
    print()
    
    print("2. 피드백 시스템 강화:")
    print("   ```python")
    print("   def enhanced_feedback_collection(prediction, confidence):")
    print("       if confidence > 0.85:")
    print("           # 높은 신뢰도 예측에 대해 적극적인 피드백 요청")
    print("           return {")
    print("               'request_feedback': True,")
    print("               'priority': 'high',")
    print("               'message': '높은 신뢰도 예측입니다. 정확성을 확인해주세요.'")
    print("           }")
    print("   ```")
    print()
    
    print("3. 모델 개선을 위한 데이터 수집:")
    print("   ```python")
    print("   def collect_overconfidence_data():")
    print("       # 과신 사례들을 별도로 수집")
    print("       overconfidence_cases = get_high_confidence_wrong_predictions()")
    print("       ")
    print("       # 추가 훈련 데이터로 활용")
    print("       for case in overconfidence_cases:")
    print("           add_to_training_data(case)")
    print("   ```")

def create_overconfidence_detector():
    """과신 탐지기 생성"""
    
    print("\n=== 과신 탐지기 구현 ===")
    
    detector_code = '''
class OverconfidenceDetector:
    """과신 탐지기"""
    
    def __init__(self, confidence_threshold=0.9, uncertainty_threshold=0.1):
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
    
    def detect_overconfidence(self, prediction, confidence, uncertainty=None):
        """과신 탐지"""
        is_overconfident = False
        reasons = []
        
        # 높은 신뢰도 체크
        if confidence > self.confidence_threshold:
            is_overconfident = True
            reasons.append(f"신뢰도가 {confidence*100:.1f}%로 매우 높음")
        
        # 불확실성 체크 (제공된 경우)
        if uncertainty is not None and uncertainty < self.uncertainty_threshold:
            is_overconfident = True
            reasons.append(f"불확실성이 {uncertainty:.3f}로 매우 낮음")
        
        return {
            'is_overconfident': is_overconfident,
            'reasons': reasons,
            'confidence': confidence,
            'uncertainty': uncertainty
        }
    
    def get_recommendation(self, detection_result):
        """권장사항 제공"""
        if detection_result['is_overconfident']:
            return {
                'action': 'manual_review',
                'message': '과신 가능성이 있습니다. 수동 검토를 권장합니다.',
                'reasons': detection_result['reasons']
            }
        else:
            return {
                'action': 'auto_process',
                'message': '정상적인 신뢰도입니다.',
                'reasons': []
            }
'''
    
    print(detector_code)

if __name__ == "__main__":
    analyze_overconfidence_cases()
    identify_overconfidence_causes()
    overconfidence_detection_strategy()
    overconfidence_mitigation()
    practical_overconfidence_handling()
    create_overconfidence_detector()

