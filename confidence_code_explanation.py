#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
신뢰도 계산 과정의 실제 코드 구현
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import pipeline, ViTForImageClassification, ViTImageProcessor
from PIL import Image

def explain_confidence_calculation():
    """신뢰도 계산 과정 상세 설명"""
    
    print("=== 신뢰도 계산 과정 상세 설명 ===")
    
    # 1. 모델 로드
    model_name = "dima806/ai_vs_real_image_detection"
    model = ViTForImageClassification.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    
    # 2. 테스트 이미지
    test_image_path = "dataSet/test2/real/r (1).jpeg"
    image = Image.open(test_image_path).convert('RGB')
    
    print(f"원본 이미지: {test_image_path}")
    print(f"이미지 크기: {image.size}")
    
    # 3. 이미지 전처리
    inputs = processor(image, return_tensors="pt")
    print(f"\n전처리된 입력 크기: {inputs['pixel_values'].shape}")
    
    # 4. 모델 추론 (원시 점수)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # 원시 점수 (raw logits)
    
    print(f"\n원시 점수 (logits): {logits}")
    print(f"REAL 클래스 점수: {logits[0][0].item():.6f}")
    print(f"FAKE 클래스 점수: {logits[0][1].item():.6f}")
    
    # 5. 소프트맥스 함수 적용
    probabilities = F.softmax(logits, dim=-1)
    print(f"\n소프트맥스 적용 후 확률:")
    print(f"REAL 확률: {probabilities[0][0].item():.6f} ({probabilities[0][0].item()*100:.2f}%)")
    print(f"FAKE 확률: {probabilities[0][1].item():.6f} ({probabilities[0][1].item()*100:.2f}%)")
    
    # 6. 예측 및 신뢰도
    predicted_class_id = torch.argmax(probabilities, dim=-1).item()
    confidence = torch.max(probabilities, dim=-1)[0].item()
    predicted_class = model.config.id2label[predicted_class_id]
    
    print(f"\n최종 결과:")
    print(f"예측된 클래스: {predicted_class}")
    print(f"신뢰도: {confidence:.6f} ({confidence*100:.2f}%)")
    
    # 7. 신뢰도 차이 계산
    sorted_probs = torch.sort(probabilities, dim=-1, descending=True)[0]
    confidence_diff = sorted_probs[0][0].item() - sorted_probs[0][1].item()
    print(f"신뢰도 차이: {confidence_diff:.6f} ({confidence_diff*100:.2f}%)")
    
    return {
        'logits': logits,
        'probabilities': probabilities,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'confidence_diff': confidence_diff
    }

def demonstrate_softmax_function():
    """소프트맥스 함수 시연"""
    
    print("\n=== 소프트맥스 함수 시연 ===")
    
    # 예시 원시 점수들
    example_logits = [
        [2.0, 0.5],    # 높은 신뢰도
        [0.1, 0.2],    # 낮은 신뢰도
        [1.0, 1.1],    # 애매한 신뢰도
        [0.0, 0.0],    # 동일한 점수
    ]
    
    for i, logits in enumerate(example_logits):
        print(f"\n예시 {i+1}: 원시 점수 = {logits}")
        
        # 소프트맥스 계산
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        probabilities = F.softmax(logits_tensor, dim=0)
        
        real_prob = probabilities[0].item()
        fake_prob = probabilities[1].item()
        
        print(f"  REAL 확률: {real_prob:.4f} ({real_prob*100:.1f}%)")
        print(f"  FAKE 확률: {fake_prob:.4f} ({fake_prob*100:.1f}%)")
        
        # 신뢰도 해석
        max_prob = max(real_prob, fake_prob)
        predicted_class = "REAL" if real_prob > fake_prob else "FAKE"
        
        if max_prob >= 0.9:
            interpretation = "매우 높은 신뢰도"
        elif max_prob >= 0.8:
            interpretation = "높은 신뢰도"
        elif max_prob >= 0.7:
            interpretation = "보통 신뢰도"
        elif max_prob >= 0.6:
            interpretation = "낮은 신뢰도"
        else:
            interpretation = "매우 낮은 신뢰도"
        
        print(f"  예측: {predicted_class} (신뢰도: {max_prob*100:.1f}%)")
        print(f"  해석: {interpretation}")

def confidence_threshold_analysis():
    """신뢰도 임계값 분석"""
    
    print("\n=== 신뢰도 임계값 분석 ===")
    
    # 다양한 신뢰도 시나리오
    scenarios = [
        {"confidence": 0.95, "description": "거의 확실한 예측"},
        {"confidence": 0.85, "description": "높은 신뢰도 예측"},
        {"confidence": 0.75, "description": "보통 신뢰도 예측"},
        {"confidence": 0.65, "description": "낮은 신뢰도 예측"},
        {"confidence": 0.55, "description": "매우 낮은 신뢰도 예측"},
        {"confidence": 0.48, "description": "거의 랜덤한 예측"},
    ]
    
    print("신뢰도별 권장 행동:")
    print("=" * 50)
    
    for scenario in scenarios:
        conf = scenario["confidence"]
        desc = scenario["description"]
        
        if conf >= 0.9:
            action = "결과를 신뢰하고 사용"
            color = "🟢"
        elif conf >= 0.8:
            action = "결과를 신뢰하되 주의 깊게 확인"
            color = "🟡"
        elif conf >= 0.7:
            action = "추가 검증 권장"
            color = "🟠"
        elif conf >= 0.6:
            action = "수동 검토 필요"
            color = "🔴"
        else:
            action = "결과를 신뢰하지 말고 재검토"
            color = "⚫"
        
        print(f"{color} {conf*100:.0f}% ({desc})")
        print(f"   → {action}")
        print()

def practical_confidence_implementation():
    """실제 구현에서의 신뢰도 활용"""
    
    print("\n=== 실제 구현에서의 신뢰도 활용 ===")
    
    print("1. 웹 애플리케이션에서의 신뢰도 처리:")
    print("```python")
    print("def predict_with_confidence(image_path):")
    print("    # 모델 예측")
    print("    results = pipe(image)")
    print("    ")
    print("    predicted_label = results[0]['label']")
    print("    confidence = results[0]['score']")
    print("    ")
    print("    # 신뢰도 기반 응답")
    print("    if confidence >= 0.8:")
    print("        return {")
    print("            'prediction': predicted_label,")
    print("            'confidence': confidence,")
    print("            'status': 'high_confidence',")
    print("            'message': '높은 신뢰도로 예측되었습니다.'")
    print("        }")
    print("    elif confidence >= 0.6:")
    print("        return {")
    print("            'prediction': predicted_label,")
    print("            'confidence': confidence,")
    print("            'status': 'medium_confidence',")
    print("            'message': '중간 신뢰도입니다. 추가 확인을 권장합니다.'")
    print("        }")
    print("    else:")
    print("        return {")
    print("            'prediction': predicted_label,")
    print("            'confidence': confidence,")
    print("            'status': 'low_confidence',")
    print("            'message': '낮은 신뢰도입니다. 수동 검토가 필요합니다.'")
    print("        }")
    print("```")
    print()
    
    print("2. 피드백 시스템에서의 신뢰도 활용:")
    print("```python")
    print("def collect_feedback_based_on_confidence(prediction, confidence):")
    print("    if confidence < 0.7:")
    print("        # 낮은 신뢰도 예측에 대해 피드백 요청")
    print("        return {")
    print("            'request_feedback': True,")
    print("            'reason': 'low_confidence',")
    print("            'message': '예측 신뢰도가 낮습니다. 피드백을 주시면 모델 개선에 도움이 됩니다.'")
    print("        }")
    print("    else:")
    print("        # 높은 신뢰도 예측은 피드백 요청하지 않음")
    print("        return {")
    print("            'request_feedback': False,")
    print("            'reason': 'high_confidence'")
    print("        }")
    print("```")

if __name__ == "__main__":
    try:
        result = explain_confidence_calculation()
        demonstrate_softmax_function()
        confidence_threshold_analysis()
        practical_confidence_implementation()
    except Exception as e:
        print(f"오류 발생: {e}")
        print("테스트 이미지가 없거나 모델 로딩에 실패했습니다.")

