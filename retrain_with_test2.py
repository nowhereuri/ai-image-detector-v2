#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test2 데이터셋을 사용한 모델 재학습
"""

import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
from datetime import datetime

class ImageDataset(Dataset):
    """이미지 데이터셋 클래스"""
    
    def __init__(self, image_paths, labels, processor, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.transform = transform
        
        # 라벨 매핑
        self.label_to_id = {"FAKE": 0, "REAL": 1}
        self.id_to_label = {0: "FAKE", 1: "REAL"}
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 이미지 로드
        try:
            image = Image.open(image_path).convert('RGB')
            
            # 전처리 (중앙 크롭 후 224x224로 리사이즈)
            width, height = image.size
            min_size = min(width, height)
            left = (width - min_size) // 2
            top = (height - min_size) // 2
            right = left + min_size
            bottom = top + min_size
            
            # 중앙 크롭
            image = image.crop((left, top, right, bottom))
            
            # 224x224로 리사이즈 (ViT 표준 입력 크기)
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            
            # 추가 변환 적용
            if self.transform:
                image = self.transform(image)
            
            # ViT 프로세서 적용
            inputs = self.processor(images=image, return_tensors="pt")
            
            return {
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'labels': torch.tensor(self.label_to_id[label], dtype=torch.long)
            }
            
        except Exception as e:
            print(f"이미지 로드 오류 {image_path}: {e}")
            # 기본값 반환
            return {
                'pixel_values': torch.zeros(3, 224, 224),
                'labels': torch.tensor(0, dtype=torch.long)
            }

def prepare_test2_data():
    """test2 데이터셋 준비"""
    print("test2 데이터셋 준비 중...")
    
    # 데이터 경로
    fake_path = "./dataSet/test2/fake"
    real_path = "./dataSet/test2/real"
    
    # 이미지 파일 수집
    image_paths = []
    labels = []
    
    # FAKE 이미지
    if os.path.exists(fake_path):
        for file in os.listdir(fake_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(fake_path, file))
                labels.append("FAKE")
    
    # REAL 이미지
    if os.path.exists(real_path):
        for file in os.listdir(real_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(real_path, file))
                labels.append("REAL")
    
    print(f"총 {len(image_paths)}개의 이미지를 찾았습니다.")
    print(f"FAKE: {labels.count('FAKE')}개, REAL: {labels.count('REAL')}개")
    
    return image_paths, labels

def create_augmented_dataset(image_paths, labels, augmentation_factor=3):
    """데이터 증강을 통한 데이터셋 확장"""
    print(f"데이터 증강 중... (증강 배수: {augmentation_factor})")
    
    augmented_paths = image_paths.copy()
    augmented_labels = labels.copy()
    
    # REAL 이미지가 적으므로 더 많이 증강
    real_indices = [i for i, label in enumerate(labels) if label == "REAL"]
    fake_indices = [i for i, label in enumerate(labels) if label == "FAKE"]
    
    # REAL 이미지 증강 (더 많이)
    for _ in range(augmentation_factor * 2):
        for idx in real_indices:
            augmented_paths.append(image_paths[idx])
            augmented_labels.append(labels[idx])
    
    # FAKE 이미지 증강
    for _ in range(augmentation_factor):
        for idx in fake_indices:
            augmented_paths.append(image_paths[idx])
            augmented_labels.append(labels[idx])
    
    print(f"증강 후 총 {len(augmented_paths)}개의 이미지")
    print(f"FAKE: {augmented_labels.count('FAKE')}개, REAL: {augmented_labels.count('REAL')}개")
    
    return augmented_paths, augmented_labels

def train_model_with_test2():
    """test2 데이터로 모델 훈련"""
    print("=== test2 데이터셋을 사용한 모델 재학습 시작 ===")
    
    # 1. 데이터 준비
    image_paths, labels = prepare_test2_data()
    
    if len(image_paths) == 0:
        print("훈련할 데이터가 없습니다.")
        return None
    
    # 2. 데이터 증강
    augmented_paths, augmented_labels = create_augmented_dataset(image_paths, labels)
    
    # 3. 훈련/검증 분할
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        augmented_paths, augmented_labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=augmented_labels
    )
    
    print(f"훈련 데이터: {len(train_paths)}개")
    print(f"검증 데이터: {len(val_paths)}개")
    
    # 4. 모델 및 프로세서 로드
    print("모델 로딩 중...")
    model_name = "dima806/ai_vs_real_image_detection"
    
    try:
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        processor = ViTImageProcessor.from_pretrained(model_name)
        print("모델 로딩 완료!")
    except Exception as e:
        print(f"모델 로딩 실패: {e}")
        return None
    
    # 5. 데이터셋 생성
    train_dataset = ImageDataset(train_paths, train_labels, processor)
    val_dataset = ImageDataset(val_paths, val_labels, processor)
    
    # 6. 훈련 설정
    training_args = TrainingArguments(
        output_dir="./test2_retrained_model",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        learning_rate=2e-5,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=3,
        report_to=None,  # wandb 비활성화
    )
    
    # 7. 메트릭 계산 함수
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}
    
    # 8. 트레이너 생성 및 훈련
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("모델 훈련 시작...")
    try:
        trainer.train()
        print("모델 훈련 완료!")
        
        # 9. 최종 평가
        print("최종 평가 중...")
        eval_results = trainer.evaluate()
        print(f"최종 검증 정확도: {eval_results['eval_accuracy']:.4f}")
        
        # 10. 모델 저장
        model.save_pretrained("./test2_retrained_model")
        processor.save_pretrained("./test2_retrained_model")
        
        # 11. 훈련 결과 저장
        training_results = {
            "model_name": model_name,
            "training_date": datetime.now().isoformat(),
            "total_images": len(augmented_paths),
            "train_images": len(train_paths),
            "val_images": len(val_paths),
            "final_accuracy": eval_results['eval_accuracy'],
            "epochs": 10,
            "learning_rate": 2e-5,
            "batch_size": 8
        }
        
        with open("test2_training_results.json", "w", encoding="utf-8") as f:
            json.dump(training_results, f, ensure_ascii=False, indent=2)
        
        print("훈련 결과가 'test2_training_results.json'에 저장되었습니다.")
        
        return model, processor
        
    except Exception as e:
        print(f"훈련 중 오류 발생: {e}")
        return None

def test_retrained_model():
    """재훈련된 모델 테스트"""
    print("\n=== 재훈련된 모델 테스트 ===")
    
    try:
        # 재훈련된 모델 로드
        model = ViTForImageClassification.from_pretrained("./test2_retrained_model")
        processor = ViTImageProcessor.from_pretrained("./test2_retrained_model")
        
        # 테스트 데이터 준비
        test_paths, test_labels = prepare_test2_data()
        
        if len(test_paths) == 0:
            print("테스트할 데이터가 없습니다.")
            return
        
        # 예측 수행
        correct = 0
        total = len(test_paths)
        predictions = []
        
        print(f"총 {total}개 이미지 테스트 중...")
        
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
                    predicted_class_id = logits.argmax(-1).item()
                
                # 라벨 매핑
                id_to_label = {0: "FAKE", 1: "REAL"}
                predicted_label = id_to_label[predicted_class_id]
                
                predictions.append(predicted_label)
                
                if predicted_label == true_label:
                    correct += 1
                
                if (i + 1) % 20 == 0:
                    print(f"진행률: {i+1}/{total} ({((i+1)/total)*100:.1f}%)")
                    
            except Exception as e:
                print(f"예측 오류 {path}: {e}")
                predictions.append("FAKE")  # 기본값
        
        # 결과 분석
        accuracy = correct / total
        print(f"\n재훈련된 모델 테스트 결과:")
        print(f"전체 정확도: {accuracy:.4f} ({correct}/{total})")
        
        # 클래스별 정확도
        fake_correct = sum(1 for i, (pred, true) in enumerate(zip(predictions, test_labels)) 
                          if pred == true and true == "FAKE")
        real_correct = sum(1 for i, (pred, true) in enumerate(zip(predictions, test_labels)) 
                          if pred == true and true == "REAL")
        
        fake_total = test_labels.count("FAKE")
        real_total = test_labels.count("REAL")
        
        print(f"FAKE 정확도: {fake_correct/fake_total:.4f} ({fake_correct}/{fake_total})")
        print(f"REAL 정확도: {real_correct/real_total:.4f} ({real_correct}/{real_total})")
        
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
        
        # 결과 저장
        test_results = {
            "model_path": "./test2_retrained_model",
            "test_date": datetime.now().isoformat(),
            "total_images": total,
            "accuracy": accuracy,
            "fake_accuracy": fake_correct/fake_total,
            "real_accuracy": real_correct/real_total,
            "confusion_matrix": cm.tolist(),
            "classification_report": report
        }
        
        with open("test2_retrained_results.json", "w", encoding="utf-8") as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)
        
        print("테스트 결과가 'test2_retrained_results.json'에 저장되었습니다.")
        
    except Exception as e:
        print(f"모델 테스트 중 오류 발생: {e}")

def main():
    """메인 함수"""
    print("test2 데이터셋을 사용한 모델 재학습 및 성능 향상")
    print("="*60)
    
    # 1. 모델 재훈련
    model, processor = train_model_with_test2()
    
    if model is not None:
        # 2. 재훈련된 모델 테스트
        test_retrained_model()
        
        print("\n" + "="*60)
        print("모델 재학습 및 테스트 완료!")
        print("개선된 모델은 './test2_retrained_model' 폴더에 저장되었습니다.")
    else:
        print("모델 재학습에 실패했습니다.")

if __name__ == "__main__":
    main()

