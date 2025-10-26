#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
종합적인 딥페이크 탐지 테스트 및 분석
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from transformers import pipeline
import time
from datetime import datetime
import json

class ComprehensiveTester:
    """종합 테스트 클래스"""
    
    def __init__(self):
        self.model_name = "dima806/ai_vs_real_image_detection"
        self.ai_pipe = None
        self.results = []
        self.load_model()
    
    def load_model(self):
        """모델 로드"""
        print("AI 모델 로딩 중...")
        try:
            self.ai_pipe = pipeline('image-classification', model=self.model_name, device=-1)
            print("AI 모델 로딩 완료!")
        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            self.ai_pipe = None
    
    def preprocess_image(self, image_path):
        """이미지 전처리 (중앙 크롭 방식)"""
        try:
            image = Image.open(image_path).convert('RGB')
            width, height = image.size
            
            # 중앙 크롭 후 32x32로 리사이즈
            min_size = min(width, height)
            left = (width - min_size) // 2
            top = (height - min_size) // 2
            right = left + min_size
            bottom = top + min_size
            
            # 중앙 크롭
            image = image.crop((left, top, right, bottom))
            
            # 32x32로 리사이즈
            image = image.resize((32, 32), Image.Resampling.LANCZOS)
            
            return image, (width, height)
        except Exception as e:
            print(f"이미지 전처리 오류 {image_path}: {e}")
            return None, None
    
    def predict_image(self, image_path):
        """이미지 예측"""
        if not self.ai_pipe:
            return None
        
        try:
            # 이미지 전처리
            processed_image, original_size = self.preprocess_image(image_path)
            if processed_image is None:
                return None
            
            # 예측 수행
            start_time = time.time()
            results = self.ai_pipe(processed_image)
            inference_time = time.time() - start_time
            
            # 결과 처리
            predicted_label = results[0]['label']
            confidence = results[0]['score']
            
            # 모든 클래스 점수
            all_scores = {}
            for result in results:
                all_scores[result['label']] = result['score']
            
            return {
                'predicted_label': predicted_label,
                'confidence': confidence,
                'all_scores': all_scores,
                'inference_time': inference_time,
                'original_size': original_size,
                'processed_size': (32, 32)
            }
        except Exception as e:
            print(f"예측 오류 {image_path}: {e}")
            return None
    
    def test_dataset(self, dataset_path, true_label):
        """데이터셋 테스트"""
        print(f"\n=== {true_label} 데이터셋 테스트 ===")
        
        if not os.path.exists(dataset_path):
            print(f"데이터셋 경로가 존재하지 않습니다: {dataset_path}")
            return []
        
        # 이미지 파일 목록
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']:
            image_files.extend(os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.lower().endswith(ext.replace('*', '')))
        
        print(f"총 {len(image_files)}개의 이미지 파일을 찾았습니다.")
        
        results = []
        for i, image_path in enumerate(image_files):
            print(f"처리 중: {i+1}/{len(image_files)} - {os.path.basename(image_path)}")
            
            # 예측 수행
            prediction = self.predict_image(image_path)
            if prediction is None:
                continue
            
            # 결과 저장
            result = {
                'image_path': image_path,
                'image_name': os.path.basename(image_path),
                'true_label': true_label,
                'predicted_label': prediction['predicted_label'],
                'confidence': prediction['confidence'],
                'is_correct': prediction['predicted_label'] == true_label,
                'inference_time': prediction['inference_time'],
                'original_size': prediction['original_size'],
                'real_score': prediction['all_scores'].get('REAL', 0),
                'fake_score': prediction['all_scores'].get('FAKE', 0)
            }
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def run_comprehensive_test(self):
        """종합 테스트 실행"""
        print("=== 종합 딥페이크 탐지 테스트 시작 ===")
        
        # 테스트 데이터셋 경로
        fake_dataset = "./dataSet/test2/FAKE"
        real_dataset = "./dataSet/test2/REAL"
        
        # FAKE 데이터셋 테스트
        fake_results = self.test_dataset(fake_dataset, "FAKE")
        
        # REAL 데이터셋 테스트
        real_results = self.test_dataset(real_dataset, "REAL")
        
        print(f"\n=== 테스트 완료 ===")
        print(f"FAKE 이미지 테스트: {len(fake_results)}개")
        print(f"REAL 이미지 테스트: {len(real_results)}개")
        print(f"총 테스트 이미지: {len(self.results)}개")
        
        return self.results
    
    def analyze_results(self):
        """결과 분석"""
        if not self.results:
            print("분석할 결과가 없습니다.")
            return
        
        print("\n=== 결과 분석 ===")
        
        # 전체 통계
        total_images = len(self.results)
        correct_predictions = sum(1 for r in self.results if r['is_correct'])
        accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
        
        print(f"전체 정확도: {accuracy:.2f}% ({correct_predictions}/{total_images})")
        
        # 클래스별 분석
        fake_results = [r for r in self.results if r['true_label'] == 'FAKE']
        real_results = [r for r in self.results if r['true_label'] == 'REAL']
        
        if fake_results:
            fake_correct = sum(1 for r in fake_results if r['is_correct'])
            fake_accuracy = (fake_correct / len(fake_results)) * 100
            print(f"FAKE 이미지 정확도: {fake_accuracy:.2f}% ({fake_correct}/{len(fake_results)})")
        
        if real_results:
            real_correct = sum(1 for r in real_results if r['is_correct'])
            real_accuracy = (real_correct / len(real_results)) * 100
            print(f"REAL 이미지 정확도: {real_accuracy:.2f}% ({real_correct}/{len(real_results)})")
        
        # 신뢰도 분석
        confidences = [r['confidence'] for r in self.results]
        avg_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)
        
        print(f"평균 신뢰도: {avg_confidence:.4f} ± {std_confidence:.4f}")
        print(f"최고 신뢰도: {max(confidences):.4f}")
        print(f"최저 신뢰도: {min(confidences):.4f}")
        
        # 추론 시간 분석
        inference_times = [r['inference_time'] for r in self.results]
        avg_inference_time = np.mean(inference_times)
        
        print(f"평균 추론 시간: {avg_inference_time:.4f}초")
        
        # 오분류 분석
        misclassified = [r for r in self.results if not r['is_correct']]
        if misclassified:
            print(f"\n오분류 사례: {len(misclassified)}개")
            for i, case in enumerate(misclassified[:5]):  # 상위 5개만 표시
                print(f"  {i+1}. {case['image_name']}: {case['true_label']} → {case['predicted_label']} (신뢰도: {case['confidence']:.4f})")
    
    def create_analysis_table(self):
        """분석표 생성"""
        if not self.results:
            print("분석할 결과가 없습니다.")
            return None
        
        # DataFrame 생성
        df = pd.DataFrame(self.results)
        
        # 분석표 생성
        analysis_table = {
            '전체 통계': {
                '총 이미지 수': len(self.results),
                '정확한 예측': sum(1 for r in self.results if r['is_correct']),
                '틀린 예측': sum(1 for r in self.results if not r['is_correct']),
                '전체 정확도': f"{(sum(1 for r in self.results if r['is_correct']) / len(self.results)) * 100:.2f}%"
            },
            'FAKE 이미지 분석': {
                '총 FAKE 이미지': len([r for r in self.results if r['true_label'] == 'FAKE']),
                '올바르게 FAKE로 분류': len([r for r in self.results if r['true_label'] == 'FAKE' and r['predicted_label'] == 'FAKE']),
                'REAL로 오분류': len([r for r in self.results if r['true_label'] == 'FAKE' and r['predicted_label'] == 'REAL']),
                'FAKE 정확도': f"{(len([r for r in self.results if r['true_label'] == 'FAKE' and r['predicted_label'] == 'FAKE']) / len([r for r in self.results if r['true_label'] == 'FAKE'])) * 100:.2f}%" if len([r for r in self.results if r['true_label'] == 'FAKE']) > 0 else "0%"
            },
            'REAL 이미지 분석': {
                '총 REAL 이미지': len([r for r in self.results if r['true_label'] == 'REAL']),
                '올바르게 REAL로 분류': len([r for r in self.results if r['true_label'] == 'REAL' and r['predicted_label'] == 'REAL']),
                'FAKE로 오분류': len([r for r in self.results if r['true_label'] == 'REAL' and r['predicted_label'] == 'FAKE']),
                'REAL 정확도': f"{(len([r for r in self.results if r['true_label'] == 'REAL' and r['predicted_label'] == 'REAL']) / len([r for r in self.results if r['true_label'] == 'REAL'])) * 100:.2f}%" if len([r for r in self.results if r['true_label'] == 'REAL']) > 0 else "0%"
            },
            '성능 지표': {
                '평균 신뢰도': f"{np.mean([r['confidence'] for r in self.results]):.4f}",
                '신뢰도 표준편차': f"{np.std([r['confidence'] for r in self.results]):.4f}",
                '평균 추론 시간': f"{np.mean([r['inference_time'] for r in self.results]):.4f}초",
                '최고 신뢰도': f"{max([r['confidence'] for r in self.results]):.4f}",
                '최저 신뢰도': f"{min([r['confidence'] for r in self.results]):.4f}"
            }
        }
        
        return analysis_table
    
    def save_results(self, filename="test_results.json"):
        """결과 저장"""
        if not self.results:
            print("저장할 결과가 없습니다.")
            return
        
        # 결과를 JSON으로 저장
        results_data = {
            'test_info': {
                'model_name': self.model_name,
                'test_date': datetime.now().isoformat(),
                'total_images': len(self.results)
            },
            'analysis_table': self.create_analysis_table(),
            'detailed_results': self.results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        print(f"결과가 {filename}에 저장되었습니다.")
    
    def print_analysis_table(self):
        """분석표 출력"""
        analysis_table = self.create_analysis_table()
        if not analysis_table:
            return
        
        print("\n" + "="*60)
        print("딥페이크 탐지 모델 성능 분석표")
        print("="*60)
        
        for category, metrics in analysis_table.items():
            print(f"\n{category}")
            print("-" * 40)
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
        
        print("\n" + "="*60)
    
    def create_confusion_matrix(self):
        """혼동 행렬 생성"""
        if not self.results:
            return None
        
        # 혼동 행렬 계산
        true_fake_pred_fake = len([r for r in self.results if r['true_label'] == 'FAKE' and r['predicted_label'] == 'FAKE'])
        true_fake_pred_real = len([r for r in self.results if r['true_label'] == 'FAKE' and r['predicted_label'] == 'REAL'])
        true_real_pred_fake = len([r for r in self.results if r['true_label'] == 'REAL' and r['predicted_label'] == 'FAKE'])
        true_real_pred_real = len([r for r in self.results if r['true_label'] == 'REAL' and r['predicted_label'] == 'REAL'])
        
        confusion_matrix = {
            '실제\\예측': ['FAKE', 'REAL'],
            'FAKE': [true_fake_pred_fake, true_fake_pred_real],
            'REAL': [true_real_pred_fake, true_real_pred_real]
        }
        
        return confusion_matrix
    
    def print_confusion_matrix(self):
        """혼동 행렬 출력"""
        confusion_matrix = self.create_confusion_matrix()
        if not confusion_matrix:
            return
        
        print("\n혼동 행렬 (Confusion Matrix)")
        print("-" * 30)
        actual_pred = '실제\\예측'
        print(f"{actual_pred:<10} {'FAKE':<8} {'REAL':<8}")
        print("-" * 30)
        print(f"{'FAKE':<10} {confusion_matrix['FAKE'][0]:<8} {confusion_matrix['FAKE'][1]:<8}")
        print(f"{'REAL':<10} {confusion_matrix['REAL'][0]:<8} {confusion_matrix['REAL'][1]:<8}")
        print("-" * 30)

def main():
    """메인 함수"""
    tester = ComprehensiveTester()
    
    # 종합 테스트 실행
    results = tester.run_comprehensive_test()
    
    # 결과 분석
    tester.analyze_results()
    
    # 분석표 출력
    tester.print_analysis_table()
    
    # 혼동 행렬 출력
    tester.print_confusion_matrix()
    
    # 결과 저장
    tester.save_results("comprehensive_test_results.json")
    
    print("\n종합 테스트 완료!")

if __name__ == "__main__":
    main()
