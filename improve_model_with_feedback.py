#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from datasets import Dataset
from PIL import Image
import numpy as np
from feedback_system import FeedbackSystem
from pathlib import Path

def main():
    print("피드백 데이터로 모델 개선 시작...")
    
    # 피드백 시스템에서 데이터 가져오기
    fs = FeedbackSystem()
    stats = fs.get_feedback_stats()
    
    print(f"총 피드백: {stats['total_feedback']}")
    print(f"정확한 피드백: {stats['correct_predictions']}")
    print(f"부정확한 피드백: {stats['incorrect_predictions']}")
    print(f"현재 정확도: {stats['accuracy']:.2f}%")
    
    if stats['total_feedback'] < 10:
        print("피드백이 부족합니다. 최소 10개 이상의 피드백이 필요합니다.")
        return
    
    # 부정확한 피드백만 학습에 사용
    print("\n부정확한 피드백으로 모델 개선 중...")
    
    # 기존 모델 로드
    print("기존 모델 로딩...")
    model = ViTForImageClassification.from_pretrained("dima806/ai_vs_real_image_detection")
    processor = ViTImageProcessor.from_pretrained("dima806/ai_vs_real_image_detection")
    
    # 부정확한 피드백 데이터 준비
    conn = fs.conn if hasattr(fs, 'conn') else None
    if not conn:
        import sqlite3
        conn = sqlite3.connect(fs.db_path)
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT image_path, user_feedback, predicted_confidence 
        FROM feedback 
        WHERE is_correct = 0 AND image_path IS NOT NULL
        LIMIT 50
    """)
    
    feedback_data = cursor.fetchall()
    print(f"학습에 사용할 부정확한 피드백: {len(feedback_data)}개")
    
    if len(feedback_data) == 0:
        print("학습할 부정확한 피드백이 없습니다.")
        return
    
    # 데이터 전처리
    images = []
    labels = []
    
    for image_path, correct_label, confidence in feedback_data:
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
                
                # 이미지 전처리 (224x224 중앙 크롭)
                width, height = image.size
                min_size = min(width, height)
                left = (width - min_size) // 2
                top = (height - min_size) // 2
                right = left + min_size
                bottom = top + min_size
                
                image = image.crop((left, top, right, bottom))
                image = image.resize((224, 224), Image.Resampling.LANCZOS)
                
                images.append(image)
                
                # 라벨 매핑
                label = 1 if correct_label == "REAL" else 0
                labels.append(label)
                
            except Exception as e:
                print(f"이미지 처리 오류: {image_path} - {e}")
                continue
    
    if len(images) == 0:
        print("처리 가능한 이미지가 없습니다.")
        return
    
    print(f"학습 데이터 준비 완료: {len(images)}개 이미지")
    
    # 데이터셋 생성
    def preprocess_function(examples):
        return processor(examples["image"], return_tensors="pt")
    
    dataset = Dataset.from_dict({
        "image": images,
        "label": labels
    })
    
    # 데이터셋 전처리
    dataset = dataset.map(preprocess_function, batched=True)
    
    # 훈련 인수 설정
    training_args = TrainingArguments(
        output_dir="./feedback_improved_model",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        warmup_steps=10,
        learning_rate=1e-5,
        logging_steps=1,
        save_steps=10,
        evaluation_strategy="no",
        save_total_limit=1,
    )
    
    # 트레이너 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor,
    )
    
    # 모델 훈련
    print("모델 훈련 시작...")
    trainer.train()
    
    # 모델 저장
    print("개선된 모델 저장 중...")
    model.save_pretrained("./feedback_improved_model")
    processor.save_pretrained("./feedback_improved_model")
    
    print("모델 개선 완료!")
    print("개선된 모델이 './feedback_improved_model' 폴더에 저장되었습니다.")
    
    # 피드백 처리 완료 표시
    cursor.execute("UPDATE feedback SET processed = 1 WHERE is_correct = 0")
    conn.commit()
    conn.close()
    
    print("피드백 처리 완료 표시 완료.")

if __name__ == "__main__":
    main()
