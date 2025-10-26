#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
개선된 이미지 전처리 방법들
"""

from PIL import Image
import numpy as np

def preprocess_method1(image, target_width=32):
    """방법 1: 가로 고정, 세로 비율 유지"""
    width, height = image.size
    new_height = int((height * target_width) / width)
    return image.resize((target_width, new_height), Image.Resampling.LANCZOS)

def preprocess_method2(image, target_size=32):
    """방법 2: 정사각형으로 패딩 추가"""
    # 비율을 유지하면서 리사이즈
    image.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    
    # 정사각형 캔버스 생성
    new_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    
    # 중앙에 이미지 붙이기
    x = (target_size - image.width) // 2
    y = (target_size - image.height) // 2
    new_image.paste(image, (x, y))
    
    return new_image

def preprocess_method3(image, target_size=32):
    """방법 3: 중앙 크롭 후 정사각형"""
    width, height = image.size
    
    # 정사각형으로 크롭
    min_size = min(width, height)
    left = (width - min_size) // 2
    top = (height - min_size) // 2
    right = left + min_size
    bottom = top + min_size
    
    cropped = image.crop((left, top, right, bottom))
    return cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)

def preprocess_method4(image, target_size=32):
    """방법 4: 다중 스케일 + 평균"""
    # 여러 크기로 리사이즈 후 평균
    sizes = [(target_size, target_size), (target_size//2, target_size//2), (target_size*2, target_size*2)]
    
    resized_images = []
    for size in sizes:
        resized = image.resize(size, Image.Resampling.LANCZOS)
        # target_size로 다시 리사이즈
        resized = resized.resize((target_size, target_size), Image.Resampling.LANCZOS)
        resized_images.append(np.array(resized))
    
    # 평균 계산
    avg_image = np.mean(resized_images, axis=0).astype(np.uint8)
    return Image.fromarray(avg_image)

# 테스트 함수
def test_preprocessing_methods():
    """전처리 방법들 테스트"""
    from transformers import pipeline
    
    # 모델 로드
    model_name = "dima806/ai_vs_real_image_detection"
    pipe = pipeline('image-classification', model=model_name, device=-1)
    
    # 테스트 이미지
    test_image_path = "dataSet/test2/real/r (1).jpeg"
    if not os.path.exists(test_image_path):
        print("테스트 이미지를 찾을 수 없습니다.")
        return
    
    image = Image.open(test_image_path).convert('RGB')
    print(f"원본 이미지 크기: {image.size}")
    
    methods = [
        ("가로 고정", preprocess_method1),
        ("패딩 추가", preprocess_method2),
        ("중앙 크롭", preprocess_method3),
        ("다중 스케일", preprocess_method4)
    ]
    
    for name, method in methods:
        try:
            processed = method(image)
            results = pipe(processed)
            predicted = results[0]['label']
            confidence = results[0]['score']
            
            print(f"{name}: {predicted} (신뢰도: {confidence:.4f}) - 크기: {processed.size}")
        except Exception as e:
            print(f"{name}: 오류 - {e}")

if __name__ == "__main__":
    import os
    test_preprocessing_methods()

