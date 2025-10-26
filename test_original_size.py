#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
import numpy as np
from pathlib import Path

def test_original_size_prediction():
    """원본 크기 이미지로 예측 테스트"""
    print("원본 크기 이미지 예측 테스트 시작...")
    
    # 모델 로드
    print("모델 로딩 중...")
    try:
        # 피드백 개선된 모델 우선 사용
        if os.path.exists("./feedback_improved_model"):
            model = ViTForImageClassification.from_pretrained("./feedback_improved_model")
            processor = ViTImageProcessor.from_pretrained("./feedback_improved_model")
            print("피드백 개선된 모델 로딩 완료!")
        elif os.path.exists("./test2_retrained_model"):
            model = ViTForImageClassification.from_pretrained("./test2_retrained_model")
            processor = ViTImageProcessor.from_pretrained("dima806/ai_vs_real_image_detection")
            print("test2 재훈련된 모델 로딩 완료!")
        else:
            model = ViTForImageClassification.from_pretrained("dima806/ai_vs_real_image_detection")
            processor = ViTImageProcessor.from_pretrained("dima806/ai_vs_real_image_detection")
            print("기본 모델 로딩 완료!")
    except Exception as e:
        print(f"모델 로딩 실패: {e}")
        return
    
    # 테스트 이미지 경로
    test_paths = [
        "./dataSet/test2/FAKE",
        "./dataSet/test2/REAL"
    ]
    
    results = []
    
    for test_path in test_paths:
        if not os.path.exists(test_path):
            print(f"테스트 경로가 존재하지 않습니다: {test_path}")
            continue
            
        label = "FAKE" if "FAKE" in test_path else "REAL"
        print(f"\n{label} 이미지 테스트 중...")
        
        # 이미지 파일들 가져오기
        image_files = list(Path(test_path).glob("*.jpg")) + list(Path(test_path).glob("*.png"))
        
        for i, image_file in enumerate(image_files[:5]):  # 처음 5개만 테스트
            try:
                # 원본 이미지 로드
                image = Image.open(image_file).convert('RGB')
                original_size = image.size
                
                print(f"  이미지 {i+1}: {image_file.name} ({original_size[0]}x{original_size[1]})")
                
                # 원본 크기 그대로 예측 (ViT는 224x224로 자동 리사이즈됨)
                inputs = processor(images=image, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=-1)
                    predicted_class_id = logits.argmax(-1).item()
                    confidence = probabilities[0][predicted_class_id].item()
                
                # 라벨 매핑
                id_to_label = {0: "FAKE", 1: "REAL"}
                predicted_label = id_to_label[predicted_class_id]
                
                # 결과 저장
                result = {
                    'file': image_file.name,
                    'original_size': original_size,
                    'actual_label': label,
                    'predicted_label': predicted_label,
                    'confidence': confidence,
                    'is_correct': predicted_label == label
                }
                results.append(result)
                
                status = "정확" if result['is_correct'] else "부정확"
                print(f"    예측: {predicted_label} (신뢰도: {confidence:.4f}) - {status}")
                
            except Exception as e:
                print(f"    오류: {e}")
                continue
    
    # 결과 분석
    print("\n" + "="*60)
    print("원본 크기 이미지 예측 결과 분석")
    print("="*60)
    
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"총 테스트 이미지: {total}개")
    print(f"정확한 예측: {correct}개")
    print(f"부정확한 예측: {total - correct}개")
    print(f"정확도: {accuracy:.2f}%")
    
    # 클래스별 분석
    fake_results = [r for r in results if r['actual_label'] == 'FAKE']
    real_results = [r for r in results if r['actual_label'] == 'REAL']
    
    if fake_results:
        fake_correct = sum(1 for r in fake_results if r['is_correct'])
        fake_accuracy = (fake_correct / len(fake_results) * 100)
        print(f"\nFAKE 이미지 정확도: {fake_accuracy:.2f}% ({fake_correct}/{len(fake_results)})")
    
    if real_results:
        real_correct = sum(1 for r in real_results if r['is_correct'])
        real_accuracy = (real_correct / len(real_results) * 100)
        print(f"REAL 이미지 정확도: {real_accuracy:.2f}% ({real_correct}/{len(real_results)})")
    
    # 이미지 크기별 분석
    print(f"\n이미지 크기 분석:")
    for result in results:
        size = result['original_size']
        status = "정확" if result['is_correct'] else "부정확"
        print(f"  {size[0]}x{size[1]}: {result['predicted_label']} (신뢰도: {result['confidence']:.4f}) - {status}")
    
    # 신뢰도 분석
    confidences = [r['confidence'] for r in results]
    avg_confidence = np.mean(confidences)
    print(f"\n평균 신뢰도: {avg_confidence:.4f}")
    print(f"최고 신뢰도: {max(confidences):.4f}")
    print(f"최저 신뢰도: {min(confidences):.4f}")
    
    return results

if __name__ == "__main__":
    test_original_size_prediction()

