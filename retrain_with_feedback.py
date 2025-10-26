#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
피드백 데이터를 활용한 모델 재훈련 스크립트
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TrainingArguments,
    Trainer,
    ViTImageProcessor,
    ViTForImageClassification,
    DefaultDataCollator
)
from datasets import Dataset as HFDataset, Image as HFImage, ClassLabel
from adaptive_preprocessing import AdaptivePreprocessor
from feedback_system import FeedbackSystem
import warnings

warnings.filterwarnings("ignore")

class FeedbackDataset(Dataset):
    """피드백 데이터셋 클래스"""
    
    def __init__(self, image_paths, labels, processor, preprocessor):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.preprocessor = preprocessor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        
        # 적응형 전처리 적용
        processed_image = self.preprocessor.adaptive_preprocessing(image)
        
        # 프로세서 적용
        inputs = self.processor(processed_image, return_tensors="pt")
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_feedback_data(feedback_system):
    """피드백 데이터 로드"""
    print("피드백 데이터 로드 중...")
    
    # 피드백 데이터 내보내기
    exported_count = feedback_system.export_feedback_for_training()
    print(f"{exported_count}개의 피드백이 훈련 데이터로 내보내졌습니다.")
    
    # 훈련 데이터 조회
    training_data = feedback_system.get_training_data()
    
    if not training_data:
        print("훈련할 피드백 데이터가 없습니다.")
        return None, None
    
    # 데이터 분리
    image_paths = []
    labels = []
    
    for image_path, label, image_size, source in training_data:
        if os.path.exists(image_path):
            image_paths.append(image_path)
            labels.append(0 if label == "REAL" else 1)  # REAL=0, FAKE=1
    
    print(f"총 {len(image_paths)}개의 피드백 데이터를 로드했습니다.")
    print(f"REAL: {labels.count(0)}개, FAKE: {labels.count(1)}개")
    
    return image_paths, labels

def load_original_training_data():
    """기존 훈련 데이터 로드"""
    print("기존 훈련 데이터 로드 중...")
    
    train_dir = "dataSet/train"
    if not os.path.exists(train_dir):
        print("기존 훈련 데이터를 찾을 수 없습니다.")
        return None, None
    
    image_paths = []
    labels = []
    
    # REAL 이미지
    real_dir = os.path.join(train_dir, "REAL")
    if os.path.exists(real_dir):
        for file in os.listdir(real_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(real_dir, file))
                labels.append(0)  # REAL = 0
    
    # FAKE 이미지
    fake_dir = os.path.join(train_dir, "FAKE")
    if os.path.exists(fake_dir):
        for file in os.listdir(fake_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(fake_dir, file))
                labels.append(1)  # FAKE = 1
    
    print(f"기존 훈련 데이터: {len(image_paths)}개")
    print(f"REAL: {labels.count(0)}개, FAKE: {labels.count(1)}개")
    
    return image_paths, labels

def combine_datasets(original_paths, original_labels, feedback_paths, feedback_labels, feedback_ratio=0.3):
    """기존 데이터와 피드백 데이터 결합"""
    print("데이터셋 결합 중...")
    
    # 피드백 데이터 샘플링 (비율에 따라)
    if feedback_paths:
        total_original = len(original_paths)
        max_feedback = int(total_original * feedback_ratio)
        
        if len(feedback_paths) > max_feedback:
            # 랜덤 샘플링
            indices = np.random.choice(len(feedback_paths), max_feedback, replace=False)
            feedback_paths = [feedback_paths[i] for i in indices]
            feedback_labels = [feedback_labels[i] for i in indices]
        
        print(f"피드백 데이터 {len(feedback_paths)}개를 기존 데이터에 추가합니다.")
    
    # 데이터 결합
    combined_paths = original_paths + feedback_paths
    combined_labels = original_labels + feedback_labels
    
    print(f"결합된 데이터셋: {len(combined_paths)}개")
    print(f"REAL: {combined_labels.count(0)}개, FAKE: {combined_labels.count(1)}개")
    
    return combined_paths, combined_labels

def train_with_feedback(model_name="dima806/ai_vs_real_image_detection", 
                       output_dir="retrained_model", 
                       epochs=1,
                       feedback_ratio=0.3):
    """피드백을 활용한 모델 재훈련"""
    
    print("=== 피드백 기반 모델 재훈련 시작 ===")
    
    # 피드백 시스템 초기화
    feedback_system = FeedbackSystem()
    
    # 피드백 데이터 로드
    feedback_paths, feedback_labels = load_feedback_data(feedback_system)
    
    # 기존 훈련 데이터 로드
    original_paths, original_labels = load_original_training_data()
    
    if not original_paths:
        print("기존 훈련 데이터가 없어 재훈련을 진행할 수 없습니다.")
        return
    
    # 데이터셋 결합
    if feedback_paths:
        combined_paths, combined_labels = combine_datasets(
            original_paths, original_labels, feedback_paths, feedback_labels, feedback_ratio
        )
    else:
        print("피드백 데이터가 없어 기존 데이터만 사용합니다.")
        combined_paths, combined_labels = original_paths, original_labels
    
    # 데이터셋 분할 (80% 훈련, 20% 검증)
    total_size = len(combined_paths)
    train_size = int(total_size * 0.8)
    
    indices = np.random.permutation(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_paths = [combined_paths[i] for i in train_indices]
    train_labels = [combined_labels[i] for i in train_indices]
    val_paths = [combined_paths[i] for i in val_indices]
    val_labels = [combined_labels[i] for i in val_indices]
    
    print(f"훈련 데이터: {len(train_paths)}개")
    print(f"검증 데이터: {len(val_paths)}개")
    
    # 프로세서 및 전처리기 초기화
    processor = ViTImageProcessor.from_pretrained(model_name)
    preprocessor = AdaptivePreprocessor(target_size=32)
    
    # 데이터셋 생성
    train_dataset = FeedbackDataset(train_paths, train_labels, processor, preprocessor)
    val_dataset = FeedbackDataset(val_paths, val_labels, processor, preprocessor)
    
    # 모델 로드
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=2)
    model.config.id2label = {0: "REAL", 1: "FAKE"}
    model.config.label2id = {"REAL": 0, "FAKE": 1}
    
    # 훈련 인수 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir='./logs',
        eval_strategy="epoch",
        learning_rate=1e-6,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.02,
        warmup_steps=50,
        remove_unused_columns=False,
        save_strategy='epoch',
        load_best_model_at_end=True,
        save_total_limit=1,
        report_to="none"
    )
    
    # 데이터 콜레이터
    data_collator = DefaultDataCollator()
    
    # 트레이너 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=processor,
    )
    
    # 훈련 전 평가
    print("훈련 전 모델 평가...")
    trainer.evaluate()
    
    # 훈련 시작
    print("모델 재훈련 시작...")
    trainer.train()
    
    # 훈련 후 평가
    print("훈련 후 모델 평가...")
    trainer.evaluate()
    
    # 모델 저장
    print(f"재훈련된 모델을 {output_dir}에 저장 중...")
    trainer.save_model()
    
    print("=== 재훈련 완료 ===")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='피드백 기반 모델 재훈련')
    parser.add_argument('--model_name', type=str, default='dima806/ai_vs_real_image_detection',
                       help='기본 모델 이름')
    parser.add_argument('--output_dir', type=str, default='retrained_model',
                       help='재훈련된 모델 저장 디렉토리')
    parser.add_argument('--epochs', type=int, default=1,
                       help='훈련 에포크 수')
    parser.add_argument('--feedback_ratio', type=float, default=0.3,
                       help='피드백 데이터 비율 (0.0-1.0)')
    
    args = parser.parse_args()
    
    train_with_feedback(
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        feedback_ratio=args.feedback_ratio
    )

if __name__ == "__main__":
    main()

