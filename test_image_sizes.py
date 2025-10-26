#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
이미지 크기 테스트 스크립트
"""

import os
from PIL import Image
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")

def test_different_sizes():
    """다양한 크기의 이미지로 테스트"""
    
    print("=== 이미지 크기별 테스트 ===")
    
    # 모델 로드
    model_name = "dima806/ai_vs_real_image_detection"
    pipe = pipeline('image-classification', model=model_name, device=-1)
    
    # 테스트할 이미지들
    test_images = []
    
    # dataSet/test2에서 이미지 찾기
    test_dir = "dataSet/test2"
    if os.path.exists(test_dir):
        # fake 폴더에서 이미지 찾기
        fake_dir = os.path.join(test_dir, "fake")
        if os.path.exists(fake_dir):
            for file in os.listdir(fake_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_images.append((os.path.join(fake_dir, file), "FAKE"))
                    if len(test_images) >= 2:
                        break
        
        # real 폴더에서 이미지 찾기
        real_dir = os.path.join(test_dir, "real")
        if os.path.exists(real_dir):
            for file in os.listdir(real_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_images.append((os.path.join(real_dir, file), "REAL"))
                    if len(test_images) >= 4:
                        break
    
    if not test_images:
        print("테스트할 이미지를 찾을 수 없습니다.")
        return
    
    print(f"{len(test_images)}개의 이미지로 테스트 시작...")
    print("=" * 60)
    
    for i, (image_path, expected_label) in enumerate(test_images, 1):
        try:
            # 원본 이미지 로드
            original_image = Image.open(image_path).convert('RGB')
            original_size = original_image.size
            
            print(f"\n{i}. {os.path.basename(image_path)}")
            print(f"   실제 라벨: {expected_label}")
            print(f"   원본 크기: {original_size[0]}x{original_size[1]}")
            
            # 다양한 크기로 테스트 (중앙 크롭 후 정사각형으로 리사이즈)
            test_sizes = [32, 224, 512]
            
            for size in test_sizes:
                # 중앙 크롭 후 정사각형으로 리사이즈
                width, height = original_size
                min_size = min(width, height)
                left = (width - min_size) // 2
                top = (height - min_size) // 2
                right = left + min_size
                bottom = top + min_size
                
                # 중앙 크롭
                cropped_image = original_image.crop((left, top, right, bottom))
                
                # 정사각형으로 리사이즈
                resized_image = cropped_image.resize((size, size), Image.Resampling.LANCZOS)
                
                # 예측 수행
                results = pipe(resized_image)
                predicted_label = results[0]['label']
                confidence = results[0]['score']
                
                is_correct = predicted_label == expected_label
                status = "O" if is_correct else "X"
                
                print(f"   {size}x{size} (중앙크롭): {predicted_label} ({confidence:.4f}) {status}")
            
        except Exception as e:
            print(f"   오류: {e}")
    
    print("\n" + "=" * 60)
    print("테스트 완료!")

if __name__ == "__main__":
    test_different_sizes()
