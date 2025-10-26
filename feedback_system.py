#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
피드백 수집 및 관리 시스템
"""

import sqlite3
import json
import os
from datetime import datetime
from PIL import Image
import hashlib

class FeedbackSystem:
    """피드백 수집 및 관리 클래스"""
    
    def __init__(self, db_path="feedback.db"):
        """피드백 시스템 초기화"""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 피드백 테이블 생성
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_hash TEXT UNIQUE,
                image_path TEXT,
                original_size TEXT,
                predicted_label TEXT,
                predicted_confidence REAL,
                user_feedback TEXT,
                is_correct BOOLEAN,
                feedback_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # 훈련 데이터 테이블 생성
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT,
                label TEXT,
                image_size TEXT,
                source TEXT,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def calculate_image_hash(self, image_path):
        """이미지 해시 계산"""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def add_feedback(self, image_path, predicted_label, predicted_confidence, user_feedback, is_correct):
        """피드백 추가"""
        try:
            # 이미지 해시 계산
            image_hash = self.calculate_image_hash(image_path)
            
            # 이미지 크기 정보
            with Image.open(image_path) as img:
                original_size = f"{img.size[0]}x{img.size[1]}"
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 피드백 저장
            cursor.execute('''
                INSERT OR REPLACE INTO feedback 
                (image_hash, image_path, original_size, predicted_label, predicted_confidence, 
                 user_feedback, is_correct, feedback_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (image_hash, image_path, original_size, predicted_label, predicted_confidence, 
                  user_feedback, is_correct, datetime.now()))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"피드백 저장 오류: {e}")
            return False
    
    def get_feedback_stats(self):
        """피드백 통계 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 전체 피드백 수
        cursor.execute("SELECT COUNT(*) FROM feedback")
        total_feedback = cursor.fetchone()[0]
        
        # 정확한 예측 수
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE is_correct = 1")
        correct_predictions = cursor.fetchone()[0]
        
        # 부정확한 예측 수
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE is_correct = 0")
        incorrect_predictions = cursor.fetchone()[0]
        
        # 처리되지 않은 피드백 수
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE processed = 0")
        unprocessed_feedback = cursor.fetchone()[0]
        
        conn.close()
        
        accuracy = (correct_predictions / total_feedback * 100) if total_feedback > 0 else 0
        
        # 최근 피드백 조회
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT predicted_label, user_feedback, predicted_confidence, is_correct, feedback_timestamp
            FROM feedback 
            ORDER BY feedback_timestamp DESC 
            LIMIT 10
        ''')
        recent_feedback = []
        for row in cursor.fetchall():
            recent_feedback.append({
                'predicted_label': row[0],
                'user_feedback': row[1],
                'predicted_confidence': row[2],
                'is_correct': bool(row[3]),
                'timestamp': row[4]
            })
        conn.close()
        
        return {
            "total_feedback": total_feedback,
            "correct_predictions": correct_predictions,
            "incorrect_predictions": incorrect_predictions,
            "unprocessed_feedback": unprocessed_feedback,
            "accuracy": accuracy,
            "recent_feedback": recent_feedback
        }
    
    def get_unprocessed_feedback(self):
        """처리되지 않은 피드백 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, image_path, predicted_label, user_feedback, is_correct, original_size
            FROM feedback 
            WHERE processed = 0
            ORDER BY feedback_timestamp DESC
        ''')
        
        feedback_data = cursor.fetchall()
        conn.close()
        
        return feedback_data
    
    def mark_feedback_processed(self, feedback_id):
        """피드백을 처리됨으로 표시"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("UPDATE feedback SET processed = 1 WHERE id = ?", (feedback_id,))
        conn.commit()
        conn.close()
    
    def add_training_data(self, image_path, label, image_size, source="user_feedback"):
        """훈련 데이터 추가"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO training_data (image_path, label, image_size, source)
            VALUES (?, ?, ?, ?)
        ''', (image_path, label, image_size, source))
        
        conn.commit()
        conn.close()
    
    def get_training_data(self):
        """훈련 데이터 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT image_path, label, image_size, source
            FROM training_data
            ORDER BY created_timestamp DESC
        ''')
        
        training_data = cursor.fetchall()
        conn.close()
        
        return training_data
    
    def export_feedback_for_training(self, output_dir="feedback_training_data"):
        """피드백 데이터를 훈련용으로 내보내기"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 라벨별 디렉토리 생성
        real_dir = os.path.join(output_dir, "REAL")
        fake_dir = os.path.join(output_dir, "FAKE")
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)
        
        feedback_data = self.get_unprocessed_feedback()
        exported_count = 0
        
        for feedback_id, image_path, predicted_label, user_feedback, is_correct, original_size in feedback_data:
            try:
                # 사용자 피드백이 있는 경우 해당 라벨 사용
                if user_feedback in ["REAL", "FAKE"]:
                    correct_label = user_feedback
                else:
                    # 피드백이 없으면 예측이 맞는지 확인
                    correct_label = predicted_label if is_correct else ("FAKE" if predicted_label == "REAL" else "REAL")
                
                # 이미지 복사
                if os.path.exists(image_path):
                    filename = os.path.basename(image_path)
                    # 파일명에 타임스탬프 추가하여 중복 방지
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_filename = f"{timestamp}_{filename}"
                    
                    target_dir = real_dir if correct_label == "REAL" else fake_dir
                    target_path = os.path.join(target_dir, new_filename)
                    
                    # 이미지 복사
                    with Image.open(image_path) as img:
                        img.save(target_path)
                    
                    # 훈련 데이터로 추가
                    self.add_training_data(target_path, correct_label, original_size, "user_feedback")
                    
                    # 피드백을 처리됨으로 표시
                    self.mark_feedback_processed(feedback_id)
                    
                    exported_count += 1
                    
            except Exception as e:
                print(f"피드백 내보내기 오류: {e}")
        
        return exported_count

# 테스트 함수
def test_feedback_system():
    """피드백 시스템 테스트"""
    feedback_system = FeedbackSystem()
    
    # 테스트 피드백 추가
    test_image = "dataSet/test2/real/r (1).jpeg"
    if os.path.exists(test_image):
        feedback_system.add_feedback(
            image_path=test_image,
            predicted_label="FAKE",
            predicted_confidence=0.8213,
            user_feedback="REAL",
            is_correct=False
        )
        
        # 통계 조회
        stats = feedback_system.get_feedback_stats()
        print("피드백 통계:", stats)
        
        # 미처리 피드백 조회
        unprocessed = feedback_system.get_unprocessed_feedback()
        print(f"미처리 피드백 수: {len(unprocessed)}")

if __name__ == "__main__":
    test_feedback_system()
