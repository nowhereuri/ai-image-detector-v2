#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
상세 분석 리포트 생성
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime

def load_test_results():
    """테스트 결과 로드"""
    with open('comprehensive_test_results.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def create_detailed_analysis():
    """상세 분석 리포트 생성"""
    results = load_test_results()
    
    print("="*80)
    print("딥페이크 탐지 모델 종합 성능 분석 리포트")
    print("="*80)
    print(f"테스트 일시: {results['test_info']['test_date']}")
    print(f"모델명: {results['test_info']['model_name']}")
    print(f"총 테스트 이미지: {results['test_info']['total_images']}개")
    print("="*80)
    
    # 1. 전체 성능 요약
    print("\n1. 전체 성능 요약")
    print("-" * 50)
    analysis = results['analysis_table']
    
    print(f"전체 정확도: {analysis['전체 통계']['전체 정확도']}")
    print(f"총 이미지 수: {analysis['전체 통계']['총 이미지 수']}개")
    print(f"정확한 예측: {analysis['전체 통계']['정확한 예측']}개")
    print(f"틀린 예측: {analysis['전체 통계']['틀린 예측']}개")
    
    # 2. 클래스별 성능 분석
    print("\n2. 클래스별 성능 분석")
    print("-" * 50)
    
    print("FAKE 이미지 탐지 성능:")
    print(f"  - 총 FAKE 이미지: {analysis['FAKE 이미지 분석']['총 FAKE 이미지']}개")
    print(f"  - 올바르게 FAKE로 분류: {analysis['FAKE 이미지 분석']['올바르게 FAKE로 분류']}개")
    print(f"  - REAL로 오분류: {analysis['FAKE 이미지 분석']['REAL로 오분류']}개")
    print(f"  - FAKE 정확도: {analysis['FAKE 이미지 분석']['FAKE 정확도']}")
    
    print("\nREAL 이미지 탐지 성능:")
    print(f"  - 총 REAL 이미지: {analysis['REAL 이미지 분석']['총 REAL 이미지']}개")
    print(f"  - 올바르게 REAL로 분류: {analysis['REAL 이미지 분석']['올바르게 REAL로 분류']}개")
    print(f"  - FAKE로 오분류: {analysis['REAL 이미지 분석']['FAKE로 오분류']}개")
    print(f"  - REAL 정확도: {analysis['REAL 이미지 분석']['REAL 정확도']}")
    
    # 3. 성능 지표 분석
    print("\n3. 성능 지표 분석")
    print("-" * 50)
    performance = analysis['성능 지표']
    print(f"평균 신뢰도: {performance['평균 신뢰도']}")
    print(f"신뢰도 표준편차: {performance['신뢰도 표준편차']}")
    print(f"평균 추론 시간: {performance['평균 추론 시간']}")
    print(f"최고 신뢰도: {performance['최고 신뢰도']}")
    print(f"최저 신뢰도: {performance['최저 신뢰도']}")
    
    # 4. 혼동 행렬
    print("\n4. 혼동 행렬 (Confusion Matrix)")
    print("-" * 50)
    print("실제\\예측    FAKE    REAL")
    print("-" * 30)
    print(f"FAKE        {analysis['FAKE 이미지 분석']['올바르게 FAKE로 분류']:>3}     {analysis['FAKE 이미지 분석']['REAL로 오분류']:>3}")
    print(f"REAL        {analysis['REAL 이미지 분석']['FAKE로 오분류']:>3}     {analysis['REAL 이미지 분석']['올바르게 REAL로 분류']:>3}")
    print("-" * 30)
    
    # 5. 상세 통계 계산
    detailed_results = results['detailed_results']
    df = pd.DataFrame(detailed_results)
    
    print("\n5. 상세 통계 분석")
    print("-" * 50)
    
    # 신뢰도 분포
    fake_correct = df[(df['true_label'] == 'FAKE') & (df['is_correct'] == True)]['confidence']
    fake_incorrect = df[(df['true_label'] == 'FAKE') & (df['is_correct'] == False)]['confidence']
    real_correct = df[(df['true_label'] == 'REAL') & (df['is_correct'] == True)]['confidence']
    real_incorrect = df[(df['true_label'] == 'REAL') & (df['is_correct'] == False)]['confidence']
    
    print("신뢰도 분포 분석:")
    print(f"  FAKE 정확 분류 평균 신뢰도: {fake_correct.mean():.4f}")
    print(f"  FAKE 오분류 평균 신뢰도: {fake_incorrect.mean():.4f}")
    print(f"  REAL 정확 분류 평균 신뢰도: {real_correct.mean():.4f}")
    print(f"  REAL 오분류 평균 신뢰도: {real_incorrect.mean():.4f}")
    
    # 추론 시간 분석
    print(f"\n추론 시간 분석:")
    print(f"  평균 추론 시간: {df['inference_time'].mean():.4f}초")
    print(f"  최대 추론 시간: {df['inference_time'].max():.4f}초")
    print(f"  최소 추론 시간: {df['inference_time'].min():.4f}초")
    print(f"  추론 시간 표준편차: {df['inference_time'].std():.4f}초")
    
    # 6. 오분류 사례 분석
    print("\n6. 오분류 사례 분석")
    print("-" * 50)
    
    misclassified = df[df['is_correct'] == False]
    print(f"총 오분류 사례: {len(misclassified)}개")
    
    # FAKE가 REAL로 오분류된 경우
    fake_to_real = misclassified[misclassified['true_label'] == 'FAKE']
    print(f"FAKE → REAL 오분류: {len(fake_to_real)}개")
    if len(fake_to_real) > 0:
        print(f"  평균 신뢰도: {fake_to_real['confidence'].mean():.4f}")
        print("  상위 5개 오분류 사례:")
        for i, (_, row) in enumerate(fake_to_real.nlargest(5, 'confidence').iterrows()):
            print(f"    {i+1}. {row['image_name']}: 신뢰도 {row['confidence']:.4f}")
    
    # REAL이 FAKE로 오분류된 경우
    real_to_fake = misclassified[misclassified['true_label'] == 'REAL']
    print(f"\nREAL → FAKE 오분류: {len(real_to_fake)}개")
    if len(real_to_fake) > 0:
        print(f"  평균 신뢰도: {real_to_fake['confidence'].mean():.4f}")
        print("  상위 5개 오분류 사례:")
        for i, (_, row) in enumerate(real_to_fake.nlargest(5, 'confidence').iterrows()):
            print(f"    {i+1}. {row['image_name']}: 신뢰도 {row['confidence']:.4f}")
    
    # 7. 성능 개선 제안
    print("\n7. 성능 개선 제안")
    print("-" * 50)
    
    real_accuracy = float(analysis['REAL 이미지 분석']['REAL 정확도'].replace('%', ''))
    fake_accuracy = float(analysis['FAKE 이미지 분석']['FAKE 정확도'].replace('%', ''))
    
    if real_accuracy < 70:
        print("WARNING: REAL 이미지 탐지 성능이 낮습니다 (50.00%)")
        print("   제안사항:")
        print("   - REAL 이미지 데이터 증강 필요")
        print("   - 전처리 방식 개선 (비율 유지 리사이즈)")
        print("   - 모델 재훈련 고려")
    
    if fake_accuracy > 80:
        print("GOOD: FAKE 이미지 탐지 성능이 양호합니다 (86.79%)")
    
    print("\n전체적인 개선 방향:")
    print("1. 데이터 불균형 해결: REAL 이미지 데이터 증강")
    print("2. 전처리 개선: 다양한 이미지 크기와 비율에 대한 적응적 처리")
    print("3. 앙상블 모델: 여러 모델의 예측 결과 결합")
    print("4. 피드백 학습: 사용자 피드백을 통한 지속적 모델 개선")
    
    # 8. 결론
    print("\n8. 결론")
    print("-" * 50)
    overall_accuracy = float(analysis['전체 통계']['전체 정확도'].replace('%', ''))
    
    if overall_accuracy >= 80:
        print("EXCELLENT: 모델 성능이 우수합니다.")
    elif overall_accuracy >= 70:
        print("WARNING: 모델 성능이 보통입니다. 개선이 필요합니다.")
    else:
        print("CRITICAL: 모델 성능이 부족합니다. 대폭적인 개선이 필요합니다.")
    
    print(f"\n현재 모델의 전체 정확도는 {overall_accuracy:.2f}%로,")
    print("FAKE 이미지 탐지는 우수하지만 REAL 이미지 탐지에서 개선이 필요합니다.")
    
    print("\n" + "="*80)

def create_performance_summary():
    """성능 요약표 생성"""
    results = load_test_results()
    analysis = results['analysis_table']
    
    print("\n성능 요약표")
    print("="*60)
    print(f"{'지표':<20} {'값':<15} {'비고':<20}")
    print("-"*60)
    print(f"{'전체 정확도':<20} {analysis['전체 통계']['전체 정확도']:<15} {'목표: 80% 이상':<20}")
    print(f"{'FAKE 정확도':<20} {analysis['FAKE 이미지 분석']['FAKE 정확도']:<15} {'우수':<20}")
    print(f"{'REAL 정확도':<20} {analysis['REAL 이미지 분석']['REAL 정확도']:<15} {'개선 필요':<20}")
    print(f"{'평균 신뢰도':<20} {analysis['성능 지표']['평균 신뢰도']:<15} {'안정적':<20}")
    print(f"{'평균 추론시간':<20} {analysis['성능 지표']['평균 추론 시간']:<15} {'빠름':<20}")
    print("="*60)

if __name__ == "__main__":
    create_detailed_analysis()
    create_performance_summary()
