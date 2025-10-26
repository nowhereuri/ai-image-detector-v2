#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
신뢰도 분석 및 해석 가이드
"""

import os
import numpy as np
from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt

def analyze_confidence_scores():
    """신뢰도 점수 분석"""
    
    print("=== 신뢰도 점수 상세 분석 ===")
    
    # 모델 로드
    model_name = "dima806/ai_vs_real_image_detection"
    pipe = pipeline('image-classification', model=model_name, device=-1)
    
    # 테스트 이미지
    test_image_path = "dataSet/test2/real/r (1).jpeg"
    if not os.path.exists(test_image_path):
        print("테스트 이미지를 찾을 수 없습니다.")
        return
    
    image = Image.open(test_image_path).convert('RGB')
    
    # 예측 수행
    results = pipe(image)
    
    print(f"원본 이미지: {test_image_path}")
    print(f"이미지 크기: {image.size}")
    print("\n모델 예측 결과:")
    print("=" * 50)
    
    # 각 클래스별 점수 출력
    for i, result in enumerate(results):
        label = result['label']
        score = result['score']
        percentage = score * 100
        
        print(f"{i+1}. {label}: {score:.6f} ({percentage:.2f}%)")
        
        # 신뢰도 해석
        if percentage >= 90:
            confidence_level = "매우 높음"
            interpretation = "모델이 매우 확신하고 있음"
        elif percentage >= 80:
            confidence_level = "높음"
            interpretation = "모델이 높은 확신을 가지고 있음"
        elif percentage >= 70:
            confidence_level = "보통"
            interpretation = "모델이 어느 정도 확신하고 있음"
        elif percentage >= 60:
            confidence_level = "낮음"
            interpretation = "모델이 불확실함"
        else:
            confidence_level = "매우 낮음"
            interpretation = "모델이 매우 불확실함"
        
        print(f"   신뢰도 수준: {confidence_level}")
        print(f"   해석: {interpretation}")
        print()
    
    # 예측 결과 분석
    predicted_label = results[0]['label']
    predicted_confidence = results[0]['score'] * 100
    second_confidence = results[1]['score'] * 100
    
    print("예측 결과 분석:")
    print("=" * 50)
    print(f"예측된 라벨: {predicted_label}")
    print(f"예측 신뢰도: {predicted_confidence:.2f}%")
    print(f"두 번째 선택지 신뢰도: {second_confidence:.2f}%")
    print(f"신뢰도 차이: {predicted_confidence - second_confidence:.2f}%")
    
    # 신뢰도 차이 해석
    confidence_diff = predicted_confidence - second_confidence
    if confidence_diff >= 50:
        diff_interpretation = "매우 명확한 예측 (모델이 확신함)"
    elif confidence_diff >= 30:
        diff_interpretation = "명확한 예측 (모델이 어느 정도 확신함)"
    elif confidence_diff >= 10:
        diff_interpretation = "애매한 예측 (모델이 불확실함)"
    else:
        diff_interpretation = "매우 애매한 예측 (모델이 매우 불확실함)"
    
    print(f"예측 명확도: {diff_interpretation}")

def explain_confidence_mechanism():
    """신뢰도 메커니즘 설명"""
    
    print("\n=== 신뢰도 메커니즘 상세 설명 ===")
    
    print("1. 소프트맥스 함수 (Softmax Function):")
    print("   - 모델의 원시 출력값(raw logits)을 확률로 변환")
    print("   - 모든 클래스의 확률 합이 1.0이 되도록 정규화")
    print("   - 공식: P(class_i) = exp(logit_i) / sum(exp(logit_j))")
    print()
    
    print("2. 신뢰도 계산 과정:")
    print("   Step 1: 모델이 각 클래스에 대한 원시 점수 생성")
    print("   Step 2: 소프트맥스 함수로 확률 변환")
    print("   Step 3: 가장 높은 확률을 가진 클래스 선택")
    print("   Step 4: 해당 확률값이 신뢰도가 됨")
    print()
    
    print("3. 신뢰도 값의 의미:")
    print("   - 0.93 (93%): 모델이 93% 확률로 이 클래스라고 예측")
    print("   - 0.48 (48%): 모델이 48% 확률로 이 클래스라고 예측")
    print("   - 0.50 (50%): 모델이 완전히 불확실 (랜덤 추측)")
    print()
    
    print("4. 실제 코드에서의 처리:")
    print("   ```python")
    print("   # Hugging Face pipeline 내부 동작")
    print("   logits = model(image)  # 원시 점수")
    print("   probabilities = softmax(logits)  # 확률 변환")
    print("   predicted_class = argmax(probabilities)  # 최고 확률 클래스")
    print("   confidence = max(probabilities)  # 최고 확률값")
    print("   ```")

def demonstrate_confidence_scenarios():
    """다양한 신뢰도 시나리오 시연"""
    
    print("\n=== 신뢰도 시나리오 시연 ===")
    
    # 시뮬레이션된 신뢰도 값들
    scenarios = [
        {
            "name": "매우 확신하는 경우",
            "real_score": 0.93,
            "fake_score": 0.07,
            "description": "모델이 REAL 이미지라고 93% 확신"
        },
        {
            "name": "불확실한 경우",
            "real_score": 0.48,
            "fake_score": 0.52,
            "description": "모델이 FAKE라고 52% 확신 (거의 랜덤)"
        },
        {
            "name": "애매한 경우",
            "real_score": 0.65,
            "fake_score": 0.35,
            "description": "모델이 REAL이라고 65% 확신 (애매함)"
        },
        {
            "name": "매우 불확실한 경우",
            "real_score": 0.51,
            "fake_score": 0.49,
            "description": "모델이 REAL이라고 51% 확신 (거의 랜덤)"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  REAL: {scenario['real_score']:.2f} ({scenario['real_score']*100:.0f}%)")
        print(f"  FAKE: {scenario['fake_score']:.2f} ({scenario['fake_score']*100:.0f}%)")
        print(f"  해석: {scenario['description']}")
        
        # 신뢰도 차이 계산
        diff = abs(scenario['real_score'] - scenario['fake_score'])
        print(f"  신뢰도 차이: {diff:.2f} ({diff*100:.0f}%)")
        
        if diff >= 0.5:
            print("  → 매우 명확한 예측")
        elif diff >= 0.3:
            print("  → 명확한 예측")
        elif diff >= 0.1:
            print("  → 애매한 예측")
        else:
            print("  → 매우 애매한 예측")

def confidence_interpretation_guide():
    """신뢰도 해석 가이드"""
    
    print("\n=== 신뢰도 해석 가이드 ===")
    
    print("신뢰도 값별 해석:")
    print("=" * 30)
    
    confidence_ranges = [
        (0.95, 1.00, "거의 확실함", "모델이 매우 확신하고 있음. 오분류 가능성 매우 낮음."),
        (0.85, 0.94, "높은 확신", "모델이 높은 확신을 가지고 있음. 일반적으로 정확함."),
        (0.75, 0.84, "보통 확신", "모델이 어느 정도 확신하고 있음. 대부분 정확함."),
        (0.65, 0.74, "낮은 확신", "모델이 불확실함. 오분류 가능성 있음."),
        (0.55, 0.64, "매우 낮은 확신", "모델이 매우 불확실함. 오분류 가능성 높음."),
        (0.45, 0.54, "랜덤 수준", "모델이 거의 랜덤하게 예측함. 신뢰하기 어려움."),
        (0.00, 0.44, "반대 확신", "모델이 반대 클래스를 더 확신함. 예측이 잘못되었을 가능성 높음.")
    ]
    
    for min_conf, max_conf, level, interpretation in confidence_ranges:
        print(f"{min_conf:.2f} - {max_conf:.2f} ({min_conf*100:.0f}% - {max_conf*100:.0f}%): {level}")
        print(f"  → {interpretation}")
        print()

def practical_confidence_usage():
    """실제 사용에서의 신뢰도 활용"""
    
    print("\n=== 실제 사용에서의 신뢰도 활용 ===")
    
    print("1. 신뢰도 기반 필터링:")
    print("   ```python")
    print("   if confidence > 0.8:")
    print("       # 높은 신뢰도: 결과를 신뢰")
    print("       return prediction")
    print("   elif confidence > 0.6:")
    print("       # 중간 신뢰도: 추가 검증 필요")
    print("       return 'uncertain'")
    print("   else:")
    print("       # 낮은 신뢰도: 수동 검토 필요")
    print("       return 'manual_review'")
    print("   ```")
    print()
    
    print("2. 신뢰도 기반 사용자 알림:")
    print("   ```python")
    print("   if confidence < 0.7:")
    print("       message = f'예측 신뢰도가 낮습니다 ({confidence*100:.0f}%)'")
    print("       message += '. 결과를 신중히 검토해주세요.'")
    print("   ```")
    print()
    
    print("3. 신뢰도 기반 모델 개선:")
    print("   - 낮은 신뢰도 예측들을 수집")
    print("   - 해당 이미지들을 추가 훈련 데이터로 활용")
    print("   - 모델 성능 개선")

if __name__ == "__main__":
    analyze_confidence_scores()
    explain_confidence_mechanism()
    demonstrate_confidence_scenarios()
    confidence_interpretation_guide()
    practical_confidence_usage()
