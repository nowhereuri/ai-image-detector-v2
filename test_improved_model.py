#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
개선된 모델 성능 테스트
"""

import os
import json
import pandas as pd
import numpy as np
from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from datetime import datetime

def test_improved_model():
    """개선된 모델 테스트"""
    print("=== 개선된 모델 성능 테스트 ===")
    
    try:
        # 재훈련된 모델 로드
        print("재훈련된 모델 로딩 중...")
        model = ViTForImageClassification.from_pretrained("./test2_retrained_model")
        # 원본 모델에서 프로세서 로드
        processor = ViTImageProcessor.from_pretrained("dima806/ai_vs_real_image_detection")
        print("모델 로딩 완료!")
        
        # 테스트 데이터 준비
        print("테스트 데이터 준비 중...")
        test_paths = []
        test_labels = []
        
        # FAKE 이미지
        fake_path = "./dataSet/test2/fake"
        if os.path.exists(fake_path):
            for file in os.listdir(fake_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    test_paths.append(os.path.join(fake_path, file))
                    test_labels.append("FAKE")
        
        # REAL 이미지
        real_path = "./dataSet/test2/real"
        if os.path.exists(real_path):
            for file in os.listdir(real_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    test_paths.append(os.path.join(real_path, file))
                    test_labels.append("REAL")
        
        print(f"총 {len(test_paths)}개 이미지 테스트")
        print(f"FAKE: {test_labels.count('FAKE')}개, REAL: {test_labels.count('REAL')}개")
        
        # 예측 수행
        correct = 0
        total = len(test_paths)
        predictions = []
        confidences = []
        results = []
        
        print(f"\n예측 수행 중...")
        
        for i, (path, true_label) in enumerate(zip(test_paths, test_labels)):
            try:
                # 이미지 로드 및 전처리
                image = Image.open(path).convert('RGB')
                
                # 중앙 크롭 후 224x224로 리사이즈
                width, height = image.size
                min_size = min(width, height)
                left = (width - min_size) // 2
                top = (height - min_size) // 2
                right = left + min_size
                bottom = top + min_size
                
                image = image.crop((left, top, right, bottom))
                image = image.resize((224, 224), Image.Resampling.LANCZOS)
                
                # 예측
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
                
                predictions.append(predicted_label)
                confidences.append(confidence)
                
                is_correct = predicted_label == true_label
                if is_correct:
                    correct += 1
                
                results.append({
                    'image_path': path,
                    'image_name': os.path.basename(path),
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'confidence': confidence,
                    'is_correct': is_correct,
                    'original_size': (width, height)
                })
                
                if (i + 1) % 20 == 0:
                    print(f"진행률: {i+1}/{total} ({((i+1)/total)*100:.1f}%)")
                    
            except Exception as e:
                print(f"예측 오류 {path}: {e}")
                predictions.append("FAKE")  # 기본값
                confidences.append(0.5)
                results.append({
                    'image_path': path,
                    'image_name': os.path.basename(path),
                    'true_label': true_label,
                    'predicted_label': "FAKE",
                    'confidence': 0.5,
                    'is_correct': False,
                    'original_size': (0, 0)
                })
        
        # 결과 분석
        accuracy = correct / total
        print(f"\n=== 개선된 모델 테스트 결과 ===")
        print(f"전체 정확도: {accuracy:.4f} ({correct}/{total})")
        
        # 클래스별 정확도
        fake_correct = sum(1 for r in results if r['is_correct'] and r['true_label'] == 'FAKE')
        real_correct = sum(1 for r in results if r['is_correct'] and r['true_label'] == 'REAL')
        
        fake_total = test_labels.count("FAKE")
        real_total = test_labels.count("REAL")
        
        fake_accuracy = fake_correct / fake_total if fake_total > 0 else 0
        real_accuracy = real_correct / real_total if real_total > 0 else 0
        
        print(f"FAKE 정확도: {fake_accuracy:.4f} ({fake_correct}/{fake_total})")
        print(f"REAL 정확도: {real_accuracy:.4f} ({real_correct}/{real_total})")
        
        # 신뢰도 분석
        avg_confidence = np.mean(confidences)
        print(f"평균 신뢰도: {avg_confidence:.4f}")
        print(f"최고 신뢰도: {max(confidences):.4f}")
        print(f"최저 신뢰도: {min(confidences):.4f}")
        
        # 혼동 행렬
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(test_labels, predictions, labels=["FAKE", "REAL"])
        print(f"\n혼동 행렬:")
        print(f"실제\\예측  FAKE  REAL")
        print(f"FAKE       {cm[0,0]:>3}   {cm[0,1]:>3}")
        print(f"REAL       {cm[1,0]:>3}   {cm[1,1]:>3}")
        
        # 상세 리포트
        report = classification_report(test_labels, predictions, labels=["FAKE", "REAL"])
        print(f"\n상세 분류 리포트:")
        print(report)
        
        # 오분류 사례 분석
        misclassified = [r for r in results if not r['is_correct']]
        if misclassified:
            print(f"\n오분류 사례: {len(misclassified)}개")
            for i, case in enumerate(misclassified[:5]):  # 상위 5개만 표시
                print(f"  {i+1}. {case['image_name']}: {case['true_label']} → {case['predicted_label']} (신뢰도: {case['confidence']:.4f})")
        
        # 결과 저장
        test_results = {
            "model_path": "./test2_retrained_model",
            "test_date": datetime.now().isoformat(),
            "total_images": total,
            "accuracy": accuracy,
            "fake_accuracy": fake_accuracy,
            "real_accuracy": real_accuracy,
            "avg_confidence": avg_confidence,
            "max_confidence": max(confidences),
            "min_confidence": min(confidences),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "misclassified_count": len(misclassified),
            "detailed_results": results
        }
        
        with open("improved_model_test_results.json", "w", encoding="utf-8") as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n테스트 결과가 'improved_model_test_results.json'에 저장되었습니다.")
        
        return test_results
        
    except Exception as e:
        print(f"모델 테스트 중 오류 발생: {e}")
        return None

def compare_with_original():
    """기존 모델과 성능 비교"""
    print("\n=== 기존 모델과 성능 비교 ===")
    
    # 기존 모델 결과 로드
    try:
        with open('comprehensive_test_results.json', 'r', encoding='utf-8') as f:
            original_results = json.load(f)
        
        # 개선된 모델 결과 로드
        with open('improved_model_test_results.json', 'r', encoding='utf-8') as f:
            improved_results = json.load(f)
        
        print("성능 비교표:")
        print("-" * 60)
        print(f"{'지표':<20} {'기존 모델':<15} {'개선된 모델':<15} {'개선도':<10}")
        print("-" * 60)
        
        # 전체 정확도 비교
        original_acc = float(original_results['analysis_table']['전체 통계']['전체 정확도'].replace('%', '')) / 100
        improved_acc = improved_results['accuracy']
        improvement = (improved_acc - original_acc) * 100
        
        print(f"{'전체 정확도':<20} {original_acc:.4f} ({original_acc*100:.2f}%){'':<3} {improved_acc:.4f} ({improved_acc*100:.2f}%){'':<3} {improvement:+.2f}%")
        
        # FAKE 정확도 비교
        original_fake = float(original_results['analysis_table']['FAKE 이미지 분석']['FAKE 정확도'].replace('%', '')) / 100
        improved_fake = improved_results['fake_accuracy']
        fake_improvement = (improved_fake - original_fake) * 100
        
        print(f"{'FAKE 정확도':<20} {original_fake:.4f} ({original_fake*100:.2f}%){'':<3} {improved_fake:.4f} ({improved_fake*100:.2f}%){'':<3} {fake_improvement:+.2f}%")
        
        # REAL 정확도 비교
        original_real = float(original_results['analysis_table']['REAL 이미지 분석']['REAL 정확도'].replace('%', '')) / 100
        improved_real = improved_results['real_accuracy']
        real_improvement = (improved_real - original_real) * 100
        
        print(f"{'REAL 정확도':<20} {original_real:.4f} ({original_real*100:.2f}%){'':<3} {improved_real:.4f} ({improved_real*100:.2f}%){'':<3} {real_improvement:+.2f}%")
        
        print("-" * 60)
        
        # 결론
        if improvement > 0:
            print(f"✅ 모델 성능이 {improvement:.2f}% 향상되었습니다!")
        else:
            print(f"❌ 모델 성능이 {abs(improvement):.2f}% 하락했습니다.")
        
        if real_improvement > 0:
            print(f"✅ REAL 이미지 탐지 성능이 {real_improvement:.2f}% 크게 향상되었습니다!")
        
        # 비교 결과 저장
        comparison_results = {
            "comparison_date": datetime.now().isoformat(),
            "original_model": {
                "accuracy": original_acc,
                "fake_accuracy": original_fake,
                "real_accuracy": original_real
            },
            "improved_model": {
                "accuracy": improved_acc,
                "fake_accuracy": improved_fake,
                "real_accuracy": improved_real
            },
            "improvements": {
                "overall_improvement": improvement,
                "fake_improvement": fake_improvement,
                "real_improvement": real_improvement
            }
        }
        
        with open("model_comparison_results.json", "w", encoding="utf-8") as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=2)
        
        print("비교 결과가 'model_comparison_results.json'에 저장되었습니다.")
        
    except Exception as e:
        print(f"성능 비교 중 오류 발생: {e}")

def main():
    """메인 함수"""
    print("개선된 모델 성능 테스트 및 비교")
    print("="*60)
    
    # 1. 개선된 모델 테스트
    test_results = test_improved_model()
    
    if test_results is not None:
        # 2. 기존 모델과 성능 비교
        compare_with_original()
        
        print("\n" + "="*60)
        print("모델 성능 테스트 및 비교 완료!")
    else:
        print("모델 테스트에 실패했습니다.")

if __name__ == "__main__":
    main()
