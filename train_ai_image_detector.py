#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Image Detector Model VIT - 훈련 스크립트
실제 이미지와 AI 생성 이미지를 구분하는 Vision Transformer 모델을 훈련합니다.
"""

import warnings
warnings.filterwarnings("ignore")

import gc
import numpy as np
import pandas as pd
import itertools
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    f1_score
)

# 이미지 처리 관련
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 데이터 불균형 처리
from imblearn.over_sampling import RandomOverSampler

# Hugging Face 관련
import accelerate
import evaluate
from datasets import Dataset, Image, ClassLabel
from transformers import (
    TrainingArguments,
    Trainer,
    ViTImageProcessor,
    ViTForImageClassification,
    DefaultDataCollator
)

# PyTorch 관련
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomRotation,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomAdjustSharpness,
    Resize,
    ToTensor
)

# 기타 유틸리티
from pathlib import Path
from tqdm import tqdm
import os
import argparse


def load_and_prepare_data(data_path):
    """
    데이터를 로드하고 전처리하는 함수
    
    Args:
        data_path (str): 데이터가 있는 경로
        
    Returns:
        tuple: (train_data, test_data, labels_list, label2id, id2label)
    """
    print("데이터 로딩 중...")
    
    # 이미지 파일 경로와 라벨 수집
    file_names = []
    labels = []
    
    for file in sorted(Path(data_path).glob('*/*.*')):
        label = str(file).split(os.sep)[-2]  # 라벨 추출
        labels.append(label)
        file_names.append(str(file))
    
    print(f"총 {len(file_names)}개의 이미지 파일을 찾았습니다.")
    
    # DataFrame 생성
    df = pd.DataFrame.from_dict({"image": file_names, "label": labels})
    print(f"데이터셋 크기: {df.shape}")
    
    # 데이터 불균형 처리 (Random Oversampling)
    print("데이터 불균형 처리 중...")
    y = df[['label']]
    df = df.drop(['label'], axis=1)
    
    ros = RandomOverSampler(random_state=83)
    df, y_resampled = ros.fit_resample(df, y)
    
    del y
    df['label'] = y_resampled
    del y_resampled
    gc.collect()
    
    print(f"리샘플링 후 데이터셋 크기: {df.shape}")
    
    # Dataset 객체 생성
    dataset = Dataset.from_pandas(df).cast_column("image", Image())
    
    # 라벨 매핑 설정
    labels_list = ['REAL', 'FAKE']
    label2id, id2label = dict(), dict()
    
    for i, label in enumerate(labels_list):
        label2id[label] = i
        id2label[i] = label
    
    print(f"라벨 매핑: {id2label}")
    
    # ClassLabel 생성 및 매핑
    ClassLabels = ClassLabel(num_classes=len(labels_list), names=labels_list)
    
    def map_label2id(example):
        example['label'] = ClassLabels.str2int(example['label'])
        return example
    
    dataset = dataset.map(map_label2id, batched=True)
    dataset = dataset.cast_column('label', ClassLabels)
    
    # 훈련/테스트 데이터 분할 (60-40)
    dataset = dataset.train_test_split(test_size=0.4, shuffle=True, stratify_by_column="label")
    train_data = dataset['train']
    test_data = dataset['test']
    
    print(f"훈련 데이터 크기: {len(train_data)}")
    print(f"테스트 데이터 크기: {len(test_data)}")
    
    return train_data, test_data, labels_list, label2id, id2label


def setup_transforms(model_str):
    """
    이미지 변환 설정 함수
    
    Args:
        model_str (str): 사용할 모델 이름
        
    Returns:
        tuple: (processor, train_transforms, val_transforms)
    """
    print("이미지 변환 설정 중...")
    
    # ViT 프로세서 로드
    processor = ViTImageProcessor.from_pretrained(model_str)
    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]
    
    print(f"이미지 크기: {size}")
    
    # 정규화 설정
    normalize = Normalize(mean=image_mean, std=image_std)
    
    # 훈련용 변환
    _train_transforms = Compose([
        Resize((size, size)),
        RandomRotation(90),
        RandomAdjustSharpness(2),
        ToTensor(),
        normalize
    ])
    
    # 검증용 변환
    _val_transforms = Compose([
        Resize((size, size)),
        ToTensor(),
        normalize
    ])
    
    def train_transforms(examples):
        examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
        return examples
    
    def val_transforms(examples):
        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
        return examples
    
    return processor, train_transforms, val_transforms


def collate_fn(examples):
    """
    배치 데이터를 준비하는 함수
    
    Args:
        examples: 배치 예제들
        
    Returns:
        dict: 배치된 픽셀 값과 라벨
    """
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example['label'] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(eval_pred):
    """
    평가 메트릭을 계산하는 함수
    
    Args:
        eval_pred: 평가 예측 결과
        
    Returns:
        dict: 계산된 메트릭들
    """
    predictions = eval_pred.predictions
    label_ids = eval_pred.label_ids
    
    accuracy = evaluate.load("accuracy")
    predicted_labels = predictions.argmax(axis=1)
    acc_score = accuracy.compute(predictions=predicted_labels, references=label_ids)['accuracy']
    
    return {"accuracy": acc_score}


def train_model(train_data, test_data, model_str, labels_list, label2id, id2label, 
                num_epochs=2, output_dir="ai_vs_real_image_detection"):
    """
    모델을 훈련하는 함수
    
    Args:
        train_data: 훈련 데이터
        test_data: 테스트 데이터
        model_str: 모델 이름
        labels_list: 라벨 리스트
        label2id: 라벨 to ID 매핑
        id2label: ID to 라벨 매핑
        num_epochs: 훈련 에포크 수
        output_dir: 출력 디렉토리
        
    Returns:
        Trainer: 훈련된 트레이너 객체
    """
    print("모델 설정 중...")
    
    # 프로세서 및 변환 설정
    processor, train_transforms, val_transforms = setup_transforms(model_str)
    
    # 데이터에 변환 적용
    train_data.set_transform(train_transforms)
    test_data.set_transform(val_transforms)
    
    # 모델 로드
    model = ViTForImageClassification.from_pretrained(model_str, num_labels=len(labels_list))
    model.config.id2label = id2label
    model.config.label2id = label2id
    
    print(f"훈련 가능한 파라미터 수: {model.num_parameters(only_trainable=True) / 1e6:.2f}M")
    
    # 훈련 인수 설정
    args = TrainingArguments(
        output_dir=output_dir,
        logging_dir='./logs',
        eval_strategy="epoch",
        learning_rate=1e-6,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=32,
        num_train_epochs=num_epochs,
        weight_decay=0.02,
        warmup_steps=50,
        remove_unused_columns=False,
        save_strategy='epoch',
        load_best_model_at_end=True,
        save_total_limit=1,
        report_to="none"
    )
    
    # 트레이너 생성
    trainer = Trainer(
        model,
        args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )
    
    print("훈련 전 모델 평가...")
    trainer.evaluate()
    
    print("모델 훈련 시작...")
    trainer.train()
    
    print("훈련 후 모델 평가...")
    trainer.evaluate()
    
    # 모델 저장
    print(f"모델을 {output_dir}에 저장 중...")
    trainer.save_model()
    
    return trainer


def evaluate_model(trainer, test_data, labels_list):
    """
    모델을 평가하고 결과를 시각화하는 함수
    
    Args:
        trainer: 훈련된 트레이너
        test_data: 테스트 데이터
        labels_list: 라벨 리스트
    """
    print("모델 평가 중...")
    
    # 예측 수행
    outputs = trainer.predict(test_data)
    print("예측 메트릭:", outputs.metrics)
    
    # 실제 라벨과 예측 라벨 추출
    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)
    
    # 정확도 및 F1 점수 계산
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"정확도: {accuracy:.4f}")
    print(f"F1 점수: {f1:.4f}")
    
    # 혼동 행렬 시각화
    def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, figsize=(10, 8)):
        plt.figure(figsize=figsize)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)
        
        fmt = '.0f'
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), 
                    horizontalalignment="center", 
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('실제 라벨')
        plt.xlabel('예측 라벨')
        plt.tight_layout()
        plt.show()
    
    # 혼동 행렬 계산 및 시각화
    if len(labels_list) <= 150:
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, labels_list, figsize=(8, 6))
    
    # 분류 리포트 출력
    print("\n분류 리포트:")
    print(classification_report(y_true, y_pred, target_names=labels_list, digits=4))


def main():
    """
    메인 함수
    """
    parser = argparse.ArgumentParser(description='AI Image Detector Model VIT 훈련')
    parser.add_argument('--data_path', type=str, required=True,
                       help='데이터가 있는 경로')
    parser.add_argument('--model_name', type=str, default='dima806/ai_vs_real_image_detection',
                       help='사용할 모델 이름')
    parser.add_argument('--epochs', type=int, default=2,
                       help='훈련 에포크 수')
    parser.add_argument('--output_dir', type=str, default='ai_vs_real_image_detection',
                       help='모델 저장 디렉토리')
    
    args = parser.parse_args()
    
    print("=== AI Image Detector Model VIT 훈련 시작 ===")
    print(f"데이터 경로: {args.data_path}")
    print(f"모델 이름: {args.model_name}")
    print(f"에포크 수: {args.epochs}")
    print(f"출력 디렉토리: {args.output_dir}")
    
    try:
        # 데이터 로드 및 전처리
        train_data, test_data, labels_list, label2id, id2label = load_and_prepare_data(args.data_path)
        
        # 모델 훈련
        trainer = train_model(
            train_data, test_data, args.model_name, labels_list, 
            label2id, id2label, args.epochs, args.output_dir
        )
        
        # 모델 평가
        evaluate_model(trainer, test_data, labels_list)
        
        print("=== 훈련 완료 ===")
        print(f"훈련된 모델이 {args.output_dir}에 저장되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        raise


if __name__ == "__main__":
    main()
