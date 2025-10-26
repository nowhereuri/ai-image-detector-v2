#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REAL 이미지 오분류 원인 분석 스크립트
"""

import os
from PIL import Image
from transformers import pipeline
import warnings
import random

warnings.filterwarnings("ignore")

def analyze_real_images():
    """REAL 이미지 오분류 원인 분석"""
    
    print("=== REAL 이미지 오분류 원인 분석 ===")
    
    # 모델 로드
    model_name = "dima806/ai_vs_real_image_detection"
    pipe = pipeline('image-classification', model=model_name, device=-1)
    
    # REAL 이미지들 분석
    real_dir = "dataSet/test2/real"
    if not os.path.exists(real_dir):
        print("REAL 이미지 디렉토리를 찾을 수 없습니다.")
        return
    
    real_files = [f for f in os.listdir(real_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not real_files:
        print("REAL 이미지 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(real_files)}개의 REAL 이미지 분석 중...")
    print("=" * 80)
    
    correct_count = 0
    total_count = 0
    
    # 모든 REAL 이미지 분석
    for i, filename in enumerate(real_files[:20]):  # 처음 20개만 분석
        try:
            image_path = os.path.join(real_dir, filename)
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            
            # 가로 32픽셀로 리사이즈 (비율 유지)
            new_width = 32
            new_height = int((original_size[1] * new_width) / original_size[0])
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 예측 수행
            results = pipe(resized_image)
            predicted_label = results[0]['label']
            confidence = results[0]['score']
            
            is_correct = predicted_label == 'REAL'
            status = "O" if is_correct else "X"
            
            if is_correct:
                correct_count += 1
            total_count += 1
            
            print(f"{i+1:2d}. {filename}")
            print(f"    원본 크기: {original_size[0]}x{original_size[1]}")
            print(f"    리사이즈: {new_width}x{new_height}")
            print(f"    예측: {predicted_label} (신뢰도: {confidence:.4f}) {status}")
            
            # 오분류된 경우 상세 분석
            if not is_correct:
                print(f"    오분류! 실제: REAL, 예측: {predicted_label}")
                print(f"    신뢰도: {confidence:.4f}")
                
                # 모든 클래스 점수 출력
                print("    전체 점수:")
                for result in results:
                    print(f"      {result['label']}: {result['score']:.4f}")
            
            print()
            
        except Exception as e:
            print(f"오류: {filename} - {e}")
    
    # 결과 요약
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print("=" * 80)
    print(f"분석 결과:")
    print(f"정확도: {correct_count}/{total_count} ({accuracy:.1f}%)")
    
    if accuracy < 50:
        print("\n심각한 문제 발견!")
        print("가능한 원인들:")
        print("1. 훈련 데이터의 라벨이 잘못되었을 수 있음")
        print("2. 모델이 제대로 훈련되지 않았을 수 있음")
        print("3. 테스트 데이터의 품질 문제")
        print("4. 이미지 전처리 방식의 문제")
    elif accuracy < 80:
        print("\n성능 개선 필요")
        print("가능한 원인들:")
        print("1. 이미지 전처리 방식 개선 필요")
        print("2. 모델 파라미터 조정 필요")
        print("3. 더 많은 훈련 데이터 필요")
    else:
        print("\n모델이 잘 작동하고 있습니다!")

def check_training_data_labels():
    """훈련 데이터 라벨 확인"""
    
    print("\n=== 훈련 데이터 라벨 확인 ===")
    
    train_dir = "dataSet/train"
    if not os.path.exists(train_dir):
        print("훈련 데이터 디렉토리를 찾을 수 없습니다.")
        return
    
    real_dir = os.path.join(train_dir, "REAL")
    fake_dir = os.path.join(train_dir, "FAKE")
    
    if os.path.exists(real_dir):
        real_files = [f for f in os.listdir(real_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"훈련 데이터 REAL 이미지 수: {len(real_files)}")
        
        # 샘플 이미지 확인
        if real_files:
            sample_file = random.choice(real_files)
            sample_path = os.path.join(real_dir, sample_file)
            try:
                img = Image.open(sample_path)
                print(f"REAL 샘플: {sample_file} - 크기: {img.size}")
            except Exception as e:
                print(f"REAL 샘플 오류: {e}")
    
    if os.path.exists(fake_dir):
        fake_files = [f for f in os.listdir(fake_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"훈련 데이터 FAKE 이미지 수: {len(fake_files)}")
        
        # 샘플 이미지 확인
        if fake_files:
            sample_file = random.choice(fake_files)
            sample_path = os.path.join(fake_dir, sample_file)
            try:
                img = Image.open(sample_path)
                print(f"FAKE 샘플: {sample_file} - 크기: {img.size}")
            except Exception as e:
                print(f"FAKE 샘플 오류: {e}")

if __name__ == "__main__":
    analyze_real_images()
    check_training_data_labels()
