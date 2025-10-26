"""
실시간 학습 시스템
업로드된 이미지를 즉시 학습에 반영하여 다음 분류부터 개선된 성능을 제공
"""

import os
import json
import sqlite3
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
import threading
import time
from datetime import datetime
import shutil

class RealtimeLearningSystem:
    def __init__(self, model_path="./test2_retrained_model", feedback_db_path="feedback.db"):
        self.model_path = model_path
        self.feedback_db_path = feedback_db_path
        self.learning_data_path = "./realtime_learning_data"
        self.backup_model_path = "./backup_models"
        
        # 디렉토리 생성
        os.makedirs(self.learning_data_path, exist_ok=True)
        os.makedirs(self.backup_model_path, exist_ok=True)
        
        # 모델과 프로세서 로드
        self.load_model()
        
        # 학습 상태
        self.is_learning = False
        self.learning_queue = []
        self.learning_lock = threading.Lock()
        
        # 성능 추적
        self.performance_history = []
        
        print("실시간 학습 시스템 초기화 완료")
    
    def load_model(self):
        """모델과 프로세서 로드"""
        try:
            if os.path.exists(self.model_path):
                self.model = ViTForImageClassification.from_pretrained(self.model_path)
                self.processor = ViTImageProcessor.from_pretrained("dima806/ai_vs_real_image_detection")
                print("개선된 모델 로드 완료")
            else:
                # 기본 모델로 폴백
                self.model = ViTForImageClassification.from_pretrained("dima806/ai_vs_real_image_detection")
                self.processor = ViTImageProcessor.from_pretrained("dima806/ai_vs_real_image_detection")
                print("기본 모델 로드 완료")
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            # 기본 모델로 폴백
            try:
                self.model = ViTForImageClassification.from_pretrained("dima806/ai_vs_real_image_detection")
                self.processor = ViTImageProcessor.from_pretrained("dima806/ai_vs_real_image_detection")
                print("기본 모델 폴백 로드 완료")
            except Exception as e2:
                print(f"기본 모델 로드도 실패: {e2}")
                self.model = None
                self.processor = None
    
    def add_feedback_to_queue(self, image_path, predicted_label, user_feedback, confidence):
        """피드백을 학습 큐에 추가"""
        with self.learning_lock:
            # 중복 피드백 방지
            for existing_feedback in self.learning_queue:
                if (existing_feedback['image_path'] == image_path and 
                    existing_feedback['user_feedback'] == user_feedback and
                    not existing_feedback['processed']):
                    print(f"중복 피드백 방지: {image_path}")
                    return
            
            feedback_data = {
                'image_path': image_path,
                'predicted_label': predicted_label,
                'user_feedback': user_feedback,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'processed': False
            }
            
            self.learning_queue.append(feedback_data)
            print(f"피드백 추가됨: {user_feedback} (신뢰도: {confidence:.3f})")
            
            # 백그라운드에서 학습 시작 (중복 실행 방지)
            if not self.is_learning and len(self.learning_queue) >= 1:
                print(f"실시간 학습 시작: 큐에 {len(self.learning_queue)}개 피드백")
                threading.Thread(target=self.process_learning_queue, daemon=True).start()
    
    def process_learning_queue(self):
        """학습 큐 처리"""
        if self.is_learning:
            return
        
        self.is_learning = True
        
        try:
            with self.learning_lock:
                if not self.learning_queue:
                    self.is_learning = False
                    return
                
                # 처리되지 않은 피드백만 필터링
                unprocessed_feedback = [f for f in self.learning_queue if not f['processed']]
                
                if len(unprocessed_feedback) >= 2:  # 2개 이상의 피드백이 쌓이면 즉시 학습
                    print(f"실시간 학습 시작: {len(unprocessed_feedback)}개 피드백 처리")
                    self.perform_realtime_learning(unprocessed_feedback)
                    
                    # 처리된 피드백 마킹
                    for feedback in unprocessed_feedback:
                        feedback['processed'] = True
                
        except Exception as e:
            print(f"학습 큐 처리 중 오류: {e}")
        finally:
            self.is_learning = False
    
    def perform_realtime_learning(self, feedback_data):
        """실시간 학습 수행"""
        try:
            # 모델 백업
            self.backup_current_model()
            
            # 학습 데이터 준비
            learning_dataset = self.prepare_learning_dataset(feedback_data)
            
            if len(learning_dataset) == 0:
                print("학습할 데이터가 없습니다.")
                return
            
            # 학습 설정 (더 빠르고 효과적인 학습)
            training_args = TrainingArguments(
                output_dir="./realtime_training_output",
                num_train_epochs=2,  # 2 에포크로 증가하여 더 효과적인 학습
                per_device_train_batch_size=2,  # 배치 크기 감소로 안정성 향상
                per_device_eval_batch_size=2,
                warmup_steps=5,  # 워밍업 단계 감소
                weight_decay=0.01,
                logging_dir="./logs",
                logging_steps=2,
                save_steps=20,
                evaluation_strategy="no",
                learning_rate=2e-5,  # 학습률 증가로 더 빠른 적응
                save_total_limit=1,
                remove_unused_columns=False,
                dataloader_drop_last=False,
            )
            
            # 트레이너 설정
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=learning_dataset,
                tokenizer=self.processor,
            )
            
            # 학습 실행
            print("실시간 학습 실행 중...")
            train_result = trainer.train()
            
            # 모델 저장
            self.model.save_pretrained(self.model_path)
            print("실시간 학습 완료 - 모델 업데이트됨")
            
            # 성능 기록
            self.record_performance_improvement(len(feedback_data))
            
        except Exception as e:
            print(f"실시간 학습 중 오류: {e}")
            # 오류 발생 시 백업 모델로 복원
            self.restore_backup_model()
    
    def prepare_learning_dataset(self, feedback_data):
        """학습 데이터셋 준비"""
        class RealtimeDataset(Dataset):
            def __init__(self, images, labels, processor):
                self.images = images
                self.labels = labels
                self.processor = processor
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                image = self.images[idx]
                label = self.labels[idx]
                
                # 이미지 전처리
                inputs = self.processor(images=image, return_tensors="pt")
                inputs['labels'] = torch.tensor(label, dtype=torch.long)
                
                return {key: val.squeeze() for key, val in inputs.items()}
        
        images = []
        labels = []
        
        for feedback in feedback_data:
            try:
                # 이미지 로드
                image = Image.open(feedback['image_path']).convert('RGB')
                
                # 중앙 크롭 및 리사이즈
                width, height = image.size
                min_size = min(width, height)
                left = (width - min_size) // 2
                top = (height - min_size) // 2
                right = left + min_size
                bottom = top + min_size
                
                image = image.crop((left, top, right, bottom))
                image = image.resize((224, 224), Image.Resampling.LANCZOS)
                
                # 라벨 매핑
                label = 1 if feedback['user_feedback'] == 'REAL' else 0
                
                images.append(image)
                labels.append(label)
                
            except Exception as e:
                print(f"이미지 처리 중 오류: {e}")
                continue
        
        if len(images) == 0:
            return None
        
        return RealtimeDataset(images, labels, self.processor)
    
    def backup_current_model(self):
        """현재 모델 백업"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.backup_model_path, f"model_backup_{timestamp}")
            
            if os.path.exists(self.model_path):
                shutil.copytree(self.model_path, backup_path)
                print(f"모델 백업 완료: {backup_path}")
        except Exception as e:
            print(f"모델 백업 중 오류: {e}")
    
    def restore_backup_model(self):
        """백업 모델 복원"""
        try:
            # 가장 최근 백업 찾기
            backup_dirs = [d for d in os.listdir(self.backup_model_path) if d.startswith("model_backup_")]
            if backup_dirs:
                latest_backup = sorted(backup_dirs)[-1]
                backup_path = os.path.join(self.backup_model_path, latest_backup)
                
                if os.path.exists(self.model_path):
                    shutil.rmtree(self.model_path)
                shutil.copytree(backup_path, self.model_path)
                
                # 모델 재로드
                self.load_model()
                print(f"백업 모델 복원 완료: {latest_backup}")
        except Exception as e:
            print(f"백업 모델 복원 중 오류: {e}")
    
    def record_performance_improvement(self, feedback_count):
        """성능 개선 기록"""
        performance_record = {
            'timestamp': datetime.now().isoformat(),
            'feedback_count': feedback_count,
            'total_feedback': len(self.learning_queue),
            'model_version': f"realtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        self.performance_history.append(performance_record)
        
        # 성능 히스토리 저장
        history_path = os.path.join(self.learning_data_path, "performance_history.json")
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.performance_history, f, ensure_ascii=False, indent=2)
    
    def get_learning_stats(self):
        """학습 통계 반환"""
        with self.learning_lock:
            total_feedback = len(self.learning_queue)
            processed_feedback = len([f for f in self.learning_queue if f['processed']])
            pending_feedback = total_feedback - processed_feedback
            
            return {
                'total_feedback': total_feedback,
                'processed_feedback': processed_feedback,
                'pending_feedback': pending_feedback,
                'is_learning': self.is_learning,
                'performance_history_count': len(self.performance_history),
                'last_learning_time': self.performance_history[-1]['timestamp'] if self.performance_history else None
            }
    
    def predict_with_learning(self, image_path):
        """학습된 모델로 예측"""
        try:
            # 모델이 로드되지 않은 경우 None 반환
            if self.model is None or self.processor is None:
                print("실시간 학습 모델이 로드되지 않음")
                return None
            
            # 이미지 로드 및 전처리
            image = Image.open(image_path).convert('RGB')
            width, height = image.size
            
            # 중앙 크롭 후 224x224로 리사이즈
            min_size = min(width, height)
            left = (width - min_size) // 2
            top = (height - min_size) // 2
            right = left + min_size
            bottom = top + min_size
            
            image = image.crop((left, top, right, bottom))
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            
            # 예측 수행
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class_id = logits.argmax(-1).item()
                confidence = probabilities[0][predicted_class_id].item()
            
            # 라벨 매핑
            id_to_label = {0: "FAKE", 1: "REAL"}
            predicted_label = id_to_label[predicted_class_id]
            
            return {
                'predicted_label': predicted_label,
                'confidence': confidence,
                'model_type': '실시간 학습 모델',
                'image_size': f"{width}x{height} -> 224x224 (중앙크롭)"
            }
            
        except Exception as e:
            print(f"실시간 학습 모델 예측 중 오류: {e}")
            return None

# 전역 실시간 학습 시스템 인스턴스
realtime_learning = RealtimeLearningSystem()
