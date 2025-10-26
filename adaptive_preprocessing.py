#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
적응형 이미지 전처리 시스템
다양한 비율의 이미지에 대한 전처리 방법들
"""

from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import random

class AdaptivePreprocessor:
    """적응형 이미지 전처리 클래스"""
    
    def __init__(self, target_size=32):
        self.target_size = target_size
        self.preprocessing_methods = [
            self.center_crop_resize,
            self.padding_resize,
            self.multi_scale_resize,
            self.aspect_ratio_preserve_resize
        ]
    
    def center_crop_resize(self, image):
        """중앙 크롭 후 리사이즈"""
        width, height = image.size
        
        # 정사각형으로 크롭
        min_size = min(width, height)
        left = (width - min_size) // 2
        top = (height - min_size) // 2
        right = left + min_size
        bottom = top + min_size
        
        cropped = image.crop((left, top, right, bottom))
        return cropped.resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)
    
    def padding_resize(self, image):
        """패딩 추가 후 리사이즈"""
        # 비율을 유지하면서 리사이즈
        image.thumbnail((self.target_size, self.target_size), Image.Resampling.LANCZOS)
        
        # 정사각형 캔버스 생성
        new_image = Image.new('RGB', (self.target_size, self.target_size), (0, 0, 0))
        
        # 중앙에 이미지 붙이기
        x = (self.target_size - image.width) // 2
        y = (self.target_size - image.height) // 2
        new_image.paste(image, (x, y))
        
        return new_image
    
    def multi_scale_resize(self, image):
        """다중 스케일 리사이즈"""
        # 여러 크기로 리사이즈 후 평균
        sizes = [(self.target_size, self.target_size), 
                (self.target_size//2, self.target_size//2), 
                (self.target_size*2, self.target_size*2)]
        
        resized_images = []
        for size in sizes:
            resized = image.resize(size, Image.Resampling.LANCZOS)
            # target_size로 다시 리사이즈
            resized = resized.resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)
            resized_images.append(np.array(resized))
        
        # 평균 계산
        avg_image = np.mean(resized_images, axis=0).astype(np.uint8)
        return Image.fromarray(avg_image)
    
    def aspect_ratio_preserve_resize(self, image):
        """비율 유지 리사이즈 (가로 고정)"""
        width, height = image.size
        
        # 가로를 target_size로 고정하고 세로는 비율에 맞춰 계산
        new_height = int((height * self.target_size) / width)
        
        # 리사이즈
        resized = image.resize((self.target_size, new_height), Image.Resampling.LANCZOS)
        
        # 정사각형으로 만들기 위해 패딩 추가
        new_image = Image.new('RGB', (self.target_size, self.target_size), (0, 0, 0))
        
        # 중앙에 이미지 붙이기
        y = (self.target_size - new_height) // 2
        new_image.paste(resized, (0, y))
        
        return new_image
    
    def random_preprocessing(self, image):
        """랜덤 전처리 방법 선택"""
        method = random.choice(self.preprocessing_methods)
        return method(image)
    
    def ensemble_preprocessing(self, image):
        """앙상블 전처리 (모든 방법 적용)"""
        processed_images = []
        for method in self.preprocessing_methods:
            processed_images.append(method(image))
        return processed_images
    
    def adaptive_preprocessing(self, image, aspect_ratio_threshold=1.5):
        """이미지 비율에 따른 적응형 전처리"""
        width, height = image.size
        aspect_ratio = max(width, height) / min(width, height)
        
        if aspect_ratio > aspect_ratio_threshold:
            # 비율이 큰 경우 (가로 또는 세로가 매우 긴 경우)
            return self.aspect_ratio_preserve_resize(image)
        else:
            # 비율이 작은 경우 (정사각형에 가까운 경우)
            return self.center_crop_resize(image)

def test_adaptive_preprocessing():
    """적응형 전처리 테스트"""
    from transformers import pipeline
    
    # 모델 로드
    model_name = "dima806/ai_vs_real_image_detection"
    pipe = pipeline('image-classification', model=model_name, device=-1)
    
    # 전처리기 초기화
    preprocessor = AdaptivePreprocessor(target_size=32)
    
    # 테스트 이미지
    test_image_path = "dataSet/test2/real/r (1).jpeg"
    if not os.path.exists(test_image_path):
        print("테스트 이미지를 찾을 수 없습니다.")
        return
    
    image = Image.open(test_image_path).convert('RGB')
    print(f"원본 이미지 크기: {image.size}")
    
    # 다양한 전처리 방법 테스트
    methods = [
        ("중앙 크롭", preprocessor.center_crop_resize),
        ("패딩 추가", preprocessor.padding_resize),
        ("다중 스케일", preprocessor.multi_scale_resize),
        ("비율 유지", preprocessor.aspect_ratio_preserve_resize),
        ("적응형", preprocessor.adaptive_preprocessing),
        ("랜덤", preprocessor.random_preprocessing)
    ]
    
    print("\n전처리 방법별 결과:")
    print("=" * 50)
    
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
    test_adaptive_preprocessing()

