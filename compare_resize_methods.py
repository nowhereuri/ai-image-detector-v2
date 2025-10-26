#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
import numpy as np
from pathlib import Path

def predict_with_method(image, model, processor, method="original"):
    """다양한 전처리 방법으로 예측"""
    if method == "original":
        # 원본 크기 그대로 (ViT가 224x224로 자동 리사이즈)
        inputs = processor(images=image, return_tensors="pt")
    elif method == "32x32":
        # 32x32로 리사이즈
        resized_image = image.resize((32, 32), Image.Resampling.LANCZOS)
        inputs = processor(images=resized_image, return_tensors="pt")
    elif method == "center_crop_32":
        # 중앙 크롭 후 32x32
        width, height = image.size
        min_size = min(width, height)
        left = (width - min_size) // 2
        top = (height - min_size) // 2
        right = left + min_size
        bottom = top + min_size
        cropped_image = image.crop((left, top, right, bottom))
        resized_image = cropped_image.resize((32, 32), Image.Resampling.LANCZOS)
        inputs = processor(images=resized_image, return_tensors="pt")
    elif method == "center_crop_224":
        # 중앙 크롭 후 224x224
        width, height = image.size
        min_size = min(width, height)
        left = (width - min_size) // 2
        top = (height - min_size) // 2
        right = left + min_size
        bottom = top + min_size
        cropped_image = image.crop((left, top, right, bottom))
        resized_image = cropped_image.resize((224, 224), Image.Resampling.LANCZOS)
        inputs = processor(images=resized_image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class_id = logits.argmax(-1).item()
        confidence = probabilities[0][predicted_class_id].item()
    
    id_to_label = {0: "FAKE", 1: "REAL"}
    predicted_label = id_to_label[predicted_class_id]
    
    return predicted_label, confidence

def compare_resize_methods():
    """다양한 리사이즈 방법 비교"""
    print("리사이즈 방법 비교 테스트 시작...")
    
    # 모델 로드
    print("모델 로딩 중...")
    try:
        if os.path.exists("./feedback_improved_model"):
            model = ViTForImageClassification.from_pretrained("./feedback_improved_model")
            processor = ViTImageProcessor.from_pretrained("./feedback_improved_model")
            print("피드백 개선된 모델 로딩 완료!")
        else:
            model = ViTForImageClassification.from_pretrained("dima806/ai_vs_real_image_detection")
            processor = ViTImageProcessor.from_pretrained("dima806/ai_vs_real_image_detection")
            print("기본 모델 로딩 완료!")
    except Exception as e:
        print(f"모델 로딩 실패: {e}")
        return
    
    # 테스트 방법들
    methods = {
        "original": "원본 크기 (ViT 자동 224x224)",
        "32x32": "32x32 직접 리사이즈",
        "center_crop_32": "중앙 크롭 후 32x32",
        "center_crop_224": "중앙 크롭 후 224x224"
    }
    
    # 테스트 이미지 경로
    test_paths = [
        "./dataSet/test2/FAKE",
        "./dataSet/test2/REAL"
    ]
    
    results = {}
    
    for method_name in methods.keys():
        results[method_name] = {
            'total': 0,
            'correct': 0,
            'fake_correct': 0,
            'fake_total': 0,
            'real_correct': 0,
            'real_total': 0,
            'confidences': []
        }
    
    for test_path in test_paths:
        if not os.path.exists(test_path):
            continue
            
        actual_label = "FAKE" if "FAKE" in test_path else "REAL"
        print(f"\n{actual_label} 이미지 테스트 중...")
        
        # 이미지 파일들 가져오기
        image_files = list(Path(test_path).glob("*.jpg")) + list(Path(test_path).glob("*.png"))
        
        for i, image_file in enumerate(image_files[:3]):  # 처음 3개만 테스트
            try:
                # 원본 이미지 로드
                image = Image.open(image_file).convert('RGB')
                original_size = image.size
                
                print(f"  이미지 {i+1}: {image_file.name} ({original_size[0]}x{original_size[1]})")
                
                # 각 방법으로 예측
                for method_name, method_desc in methods.items():
                    predicted_label, confidence = predict_with_method(
                        image, model, processor, method_name
                    )
                    
                    is_correct = predicted_label == actual_label
                    results[method_name]['total'] += 1
                    results[method_name]['confidences'].append(confidence)
                    
                    if is_correct:
                        results[method_name]['correct'] += 1
                    
                    if actual_label == "FAKE":
                        results[method_name]['fake_total'] += 1
                        if is_correct:
                            results[method_name]['fake_correct'] += 1
                    else:
                        results[method_name]['real_total'] += 1
                        if is_correct:
                            results[method_name]['real_correct'] += 1
                    
                    status = "정확" if is_correct else "부정확"
                    print(f"    {method_desc}: {predicted_label} (신뢰도: {confidence:.4f}) - {status}")
                
            except Exception as e:
                print(f"    오류: {e}")
                continue
    
    # 결과 분석
    print("\n" + "="*80)
    print("리사이즈 방법별 성능 비교")
    print("="*80)
    
    for method_name, method_desc in methods.items():
        result = results[method_name]
        if result['total'] == 0:
            continue
            
        accuracy = (result['correct'] / result['total'] * 100)
        fake_accuracy = (result['fake_correct'] / result['fake_total'] * 100) if result['fake_total'] > 0 else 0
        real_accuracy = (result['real_correct'] / result['real_total'] * 100) if result['real_total'] > 0 else 0
        avg_confidence = np.mean(result['confidences'])
        
        print(f"\n{method_desc}:")
        print(f"  전체 정확도: {accuracy:.2f}% ({result['correct']}/{result['total']})")
        print(f"  FAKE 정확도: {fake_accuracy:.2f}% ({result['fake_correct']}/{result['fake_total']})")
        print(f"  REAL 정확도: {real_accuracy:.2f}% ({result['real_correct']}/{result['real_total']})")
        print(f"  평균 신뢰도: {avg_confidence:.4f}")
    
    # 최고 성능 방법 찾기
    best_method = max(methods.keys(), key=lambda x: results[x]['correct'] / results[x]['total'] if results[x]['total'] > 0 else 0)
    best_accuracy = (results[best_method]['correct'] / results[best_method]['total'] * 100) if results[best_method]['total'] > 0 else 0
    
    print(f"\n최고 성능 방법: {methods[best_method]}")
    print(f"최고 정확도: {best_accuracy:.2f}%")

if __name__ == "__main__":
    compare_resize_methods()

