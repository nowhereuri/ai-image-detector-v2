#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Image Detector - 빠른 테스트 스크립트
사전 훈련된 모델을 사용하여 이미지를 분류합니다.
"""

import os
from PIL import Image
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")

def test_ai_image_detector():
    """
    AI 이미지 탐지기 테스트 함수
    """
    print("=== AI Image Detector 빠른 테스트 ===")
    
    # 사전 훈련된 모델 사용
    model_name = "dima806/ai_vs_real_image_detection"
    
    print(f"모델 로딩 중: {model_name}")
    
    try:
        # 파이프라인 생성 (CPU 사용)
        pipe = pipeline('image-classification', model=model_name, device=-1)
        print("✅ 모델 로딩 완료!")
        
        # 테스트할 이미지 경로들
        test_images = []
        
        # dataSet/test2에서 몇 개 이미지 테스트
        test_dir = "dataSet/test2"
        if os.path.exists(test_dir):
            # fake 폴더에서 이미지 찾기
            fake_dir = os.path.join(test_dir, "fake")
            if os.path.exists(fake_dir):
                for file in os.listdir(fake_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        test_images.append((os.path.join(fake_dir, file), "FAKE"))
                        if len(test_images) >= 3:  # 최대 3개
                            break
            
            # real 폴더에서 이미지 찾기
            real_dir = os.path.join(test_dir, "real")
            if os.path.exists(real_dir):
                for file in os.listdir(real_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        test_images.append((os.path.join(real_dir, file), "REAL"))
                        if len(test_images) >= 6:  # 총 6개
                            break
        
        if not test_images:
            print("❌ 테스트할 이미지를 찾을 수 없습니다.")
            return
        
        print(f"\n{len(test_images)}개의 이미지로 테스트 시작...")
        print("=" * 50)
        
        correct_predictions = 0
        total_predictions = len(test_images)
        
        for i, (image_path, expected_label) in enumerate(test_images, 1):
            try:
                # 이미지 로드
                image = Image.open(image_path).convert('RGB')
                
                # 예측 수행
                results = pipe(image)
                predicted_label = results[0]['label']
                confidence = results[0]['score']
                
                # 결과 출력
                image_name = os.path.basename(image_path)
                is_correct = predicted_label == expected_label
                status = "✅" if is_correct else "❌"
                
                print(f"{i}. {image_name}")
                print(f"   실제: {expected_label}")
                print(f"   예측: {predicted_label} (신뢰도: {confidence:.4f})")
                print(f"   결과: {status}")
                
                if is_correct:
                    correct_predictions += 1
                
                print()
                
            except Exception as e:
                print(f"❌ {image_path} 처리 중 오류: {e}")
        
        # 전체 결과
        accuracy = correct_predictions / total_predictions * 100
        print("=" * 50)
        print(f"테스트 완료!")
        print(f"정확도: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)")
        
        if accuracy >= 80:
            print("🎉 모델이 잘 작동하고 있습니다!")
        else:
            print("⚠️  모델 성능이 예상보다 낮습니다.")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("\n해결 방법:")
        print("1. 인터넷 연결 확인")
        print("2. transformers 패키지 업데이트: pip install --upgrade transformers")
        print("3. 필요한 패키지 설치: pip install -r requirements.txt")

if __name__ == "__main__":
    test_ai_image_detector()

