#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
과신 탐지기 구현
"""

import numpy as np
from transformers import pipeline
from PIL import Image
from adaptive_preprocessing import AdaptivePreprocessor

class OverconfidenceDetector:
    """과신 탐지기"""
    
    def __init__(self, confidence_threshold=0.9, uncertainty_threshold=0.1):
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.preprocessor = AdaptivePreprocessor(target_size=32)
    
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
                'reasons': detection_result['reasons'],
                'priority': 'high'
            }
        else:
            return {
                'action': 'auto_process',
                'message': '정상적인 신뢰도입니다.',
                'reasons': [],
                'priority': 'normal'
            }
    
    def ensemble_prediction(self, image, model_pipe, num_predictions=5):
        """앙상블 예측으로 불확실성 측정"""
        predictions = []
        confidences = []
        
        # 다양한 전처리 방법으로 예측
        preprocessing_methods = [
            self.preprocessor.center_crop_resize,
            self.preprocessor.padding_resize,
            self.preprocessor.aspect_ratio_preserve_resize,
            self.preprocessor.multi_scale_resize,
            self.preprocessor.adaptive_preprocessing
        ]
        
        for method in preprocessing_methods[:num_predictions]:
            try:
                processed_image = method(image)
                result = model_pipe(processed_image)
                
                predicted_label = result[0]['label']
                confidence = result[0]['score']
                
                predictions.append(predicted_label)
                confidences.append(confidence)
            except Exception as e:
                print(f"앙상블 예측 중 오류: {e}")
                continue
        
        if not predictions:
            return None, None, None
        
        # 예측 결과 통합
        final_prediction = max(set(predictions), key=predictions.count)  # 최빈값
        avg_confidence = np.mean(confidences)
        
        # 불확실성 계산 (예측 분산)
        prediction_variance = np.var(confidences)
        uncertainty = prediction_variance
        
        return final_prediction, avg_confidence, uncertainty
    
    def analyze_prediction_reliability(self, image, model_pipe):
        """예측 신뢰성 분석"""
        # 기본 예측
        result = model_pipe(image)
        basic_prediction = result[0]['label']
        basic_confidence = result[0]['score']
        
        # 앙상블 예측
        ensemble_pred, ensemble_conf, uncertainty = self.ensemble_prediction(image, model_pipe)
        
        # 과신 탐지
        overconfidence_result = self.detect_overconfidence(
            basic_prediction, basic_confidence, uncertainty
        )
        
        # 권장사항
        recommendation = self.get_recommendation(overconfidence_result)
        
        return {
            'basic_prediction': basic_prediction,
            'basic_confidence': basic_confidence,
            'ensemble_prediction': ensemble_pred,
            'ensemble_confidence': ensemble_conf,
            'uncertainty': uncertainty,
            'overconfidence_detected': overconfidence_result['is_overconfident'],
            'overconfidence_reasons': overconfidence_result['reasons'],
            'recommendation': recommendation,
            'prediction_agreement': basic_prediction == ensemble_pred if ensemble_pred else True
        }

def test_overconfidence_detector():
    """과신 탐지기 테스트"""
    
    print("=== 과신 탐지기 테스트 ===")
    
    # 모델 로드
    model_name = "dima806/ai_vs_real_image_detection"
    model_pipe = pipeline('image-classification', model=model_name, device=-1)
    
    # 탐지기 초기화
    detector = OverconfidenceDetector(confidence_threshold=0.85)
    
    # 테스트 이미지
    test_image_path = "dataSet/test2/real/r (1).jpeg"
    if not os.path.exists(test_image_path):
        print("테스트 이미지를 찾을 수 없습니다.")
        return
    
    image = Image.open(test_image_path).convert('RGB')
    
    # 신뢰성 분석
    analysis = detector.analyze_prediction_reliability(image, model_pipe)
    
    print(f"테스트 이미지: {test_image_path}")
    print(f"이미지 크기: {image.size}")
    print()
    
    print("기본 예측:")
    print(f"  예측: {analysis['basic_prediction']}")
    print(f"  신뢰도: {analysis['basic_confidence']:.4f} ({analysis['basic_confidence']*100:.1f}%)")
    print()
    
    print("앙상블 예측:")
    print(f"  예측: {analysis['ensemble_prediction']}")
    print(f"  신뢰도: {analysis['ensemble_confidence']:.4f} ({analysis['ensemble_confidence']*100:.1f}%)")
    print(f"  불확실성: {analysis['uncertainty']:.4f}")
    print()
    
    print("과신 탐지:")
    print(f"  과신 탐지됨: {'예' if analysis['overconfidence_detected'] else '아니오'}")
    if analysis['overconfidence_reasons']:
        print("  과신 이유:")
        for reason in analysis['overconfidence_reasons']:
            print(f"    - {reason}")
    print()
    
    print("권장사항:")
    print(f"  행동: {analysis['recommendation']['action']}")
    print(f"  메시지: {analysis['recommendation']['message']}")
    print(f"  우선순위: {analysis['recommendation']['priority']}")
    print()
    
    print("예측 일치성:")
    print(f"  기본 예측과 앙상블 예측 일치: {'예' if analysis['prediction_agreement'] else '아니오'}")

if __name__ == "__main__":
    import os
    test_overconfidence_detector()

