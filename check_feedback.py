#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from feedback_system import FeedbackSystem

def main():
    print("피드백 통계 확인 중...")
    
    fs = FeedbackSystem()
    stats = fs.get_feedback_stats()
    
    print(f"총 피드백: {stats['total_feedback']}")
    print(f"정확한 피드백: {stats['correct_predictions']}")
    print(f"부정확한 피드백: {stats['incorrect_predictions']}")
    print(f"정확도: {stats['accuracy']:.2f}%")
    
    print("\n최근 피드백:")
    for feedback in stats['recent_feedback'][:5]:
        status = "정확" if feedback['is_correct'] else "부정확"
        print(f"  {status} {feedback['predicted_label']} -> {feedback['user_feedback']} (신뢰도: {feedback['predicted_confidence']:.2f}%)")
    
    # 피드백 데이터를 학습용으로 내보내기
    print("\n피드백 데이터 내보내기...")
    feedback_data = fs.export_feedback_for_training()
    print(f"학습용 데이터: {len(feedback_data)}개 피드백")
    
    if len(feedback_data) > 0:
        print("피드백 데이터가 충분합니다. 모델 재훈련을 시작할 수 있습니다.")
    else:
        print("피드백 데이터가 부족합니다. 더 많은 피드백을 수집해주세요.")

if __name__ == "__main__":
    main()
