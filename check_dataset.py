#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터셋 검증 스크립트
"""

import os
from PIL import Image
import random

def check_dataset():
    """데이터셋 검증"""
    
    print("=== 데이터셋 검증 ===")
    
    # 훈련 데이터 확인
    train_dir = "dataSet/train"
    if os.path.exists(train_dir):
        print(f"\n훈련 데이터: {train_dir}")
        
        real_dir = os.path.join(train_dir, "REAL")
        fake_dir = os.path.join(train_dir, "FAKE")
        
        if os.path.exists(real_dir):
            real_files = [f for f in os.listdir(real_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"REAL 이미지 수: {len(real_files)}")
            
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
            print(f"FAKE 이미지 수: {len(fake_files)}")
            
            # 샘플 이미지 확인
            if fake_files:
                sample_file = random.choice(fake_files)
                sample_path = os.path.join(fake_dir, sample_file)
                try:
                    img = Image.open(sample_path)
                    print(f"FAKE 샘플: {sample_file} - 크기: {img.size}")
                except Exception as e:
                    print(f"FAKE 샘플 오류: {e}")
    
    # 테스트 데이터 확인
    test_dir = "dataSet/test2"
    if os.path.exists(test_dir):
        print(f"\n테스트 데이터: {test_dir}")
        
        real_dir = os.path.join(test_dir, "real")
        fake_dir = os.path.join(test_dir, "fake")
        
        if os.path.exists(real_dir):
            real_files = [f for f in os.listdir(real_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"real 이미지 수: {len(real_files)}")
            
            # 샘플 이미지 확인
            if real_files:
                sample_file = random.choice(real_files)
                sample_path = os.path.join(real_dir, sample_file)
                try:
                    img = Image.open(sample_path)
                    print(f"real 샘플: {sample_file} - 크기: {img.size}")
                except Exception as e:
                    print(f"real 샘플 오류: {e}")
        
        if os.path.exists(fake_dir):
            fake_files = [f for f in os.listdir(fake_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"fake 이미지 수: {len(fake_files)}")
            
            # 샘플 이미지 확인
            if fake_files:
                sample_file = random.choice(fake_files)
                sample_path = os.path.join(fake_dir, sample_file)
                try:
                    img = Image.open(sample_path)
                    print(f"fake 샘플: {sample_file} - 크기: {img.size}")
                except Exception as e:
                    print(f"fake 샘플 오류: {e}")

if __name__ == "__main__":
    check_dataset()

