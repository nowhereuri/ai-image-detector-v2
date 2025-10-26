#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Image Detector Web Application
Flask를 사용한 웹 애플리케이션
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
from transformers import pipeline
import warnings
from feedback_system import FeedbackSystem
try:
    from realtime_learning_system import realtime_learning
    print("실시간 학습 시스템 로드 완료")
except Exception as e:
    print(f"실시간 학습 시스템 로드 실패: {e}")
    realtime_learning = None
from overconfidence_detector import OverconfidenceDetector

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 제한

# 업로드 폴더 설정
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# AI 모델 로드 (피드백 개선된 모델 우선 사용)
print("AI 모델 로딩 중...")
try:
    # 1순위: 피드백으로 개선된 모델
    if os.path.exists("./feedback_improved_model"):
        from transformers import ViTForImageClassification, ViTImageProcessor
        model = ViTForImageClassification.from_pretrained("./feedback_improved_model")
        processor = ViTImageProcessor.from_pretrained("./feedback_improved_model")
        ai_pipe = None
        print("피드백 개선된 AI 모델 로딩 완료!")
    # 2순위: test2로 재훈련된 모델
    elif os.path.exists("./test2_retrained_model"):
        from transformers import ViTForImageClassification, ViTImageProcessor
        model = ViTForImageClassification.from_pretrained("./test2_retrained_model")
        processor = ViTImageProcessor.from_pretrained("dima806/ai_vs_real_image_detection")
        ai_pipe = None
        print("test2 재훈련된 AI 모델 로딩 완료!")
    # 3순위: 기본 모델
    else:
        model_name = "dima806/ai_vs_real_image_detection"
        ai_pipe = pipeline('image-classification', model=model_name, device=-1)
        model = None
        processor = None
        print("기본 AI 모델 로딩 완료!")
except Exception as e:
    print(f"모든 모델 로딩 실패, 기본 모델 사용: {e}")
    try:
        model_name = "dima806/ai_vs_real_image_detection"
        ai_pipe = pipeline('image-classification', model=model_name, device=-1)
        model = None
        processor = None
        print("기본 AI 모델 로딩 완료!")
    except Exception as e2:
        print(f"모델 로딩 실패: {e2}")
        ai_pipe = None
        model = None
        processor = None

# 피드백 시스템 초기화
feedback_system = FeedbackSystem()

def allowed_file(filename):
    """허용된 파일 확장자인지 확인"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    """이미지 예측 함수 - 실시간 학습 모델 우선 사용"""
    try:
        # 실시간 학습 모델로 먼저 시도
        if realtime_learning is not None:
            prediction = realtime_learning.predict_with_learning(image_path)
            if prediction:
                prediction["model_type"] = "실시간 학습 모델"
                print(f"실시간 학습 모델 사용: {prediction['predicted_label']} (신뢰도: {prediction['confidence']:.4f})")
                return prediction
        
        # 실시간 학습 모델 실패 시 기존 모델 사용
        if ai_pipe is None and model is None:
            return {"error": "모델이 로드되지 않았습니다."}
        
        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        
        # 개선된 모델 사용 시 원본 크기 그대로 사용 (ViT가 자동으로 224x224로 처리)
        if model is not None and processor is not None:
            # 원본 이미지 그대로 사용 (ViT ImageProcessor가 자동으로 전처리)
            new_width = width
            new_height = height
            
            # 개선된 모델로 예측 (원본 크기)
            import torch
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class_id = logits.argmax(-1).item()
                confidence = probabilities[0][predicted_class_id].item()
            
            # 라벨 매핑
            id_to_label = {0: "FAKE", 1: "REAL"}
            predicted_label = id_to_label[predicted_class_id]
            
            # 결과 형식 맞추기
            results = [{'label': predicted_label, 'score': confidence}]
            
        else:
            # 기본 모델 사용 시 32x32로 전처리
            min_size = min(width, height)
            left = (width - min_size) // 2
            top = (height - min_size) // 2
            right = left + min_size
            bottom = top + min_size
            
            # 중앙 크롭
            image = image.crop((left, top, right, bottom))
            
            # 32x32로 리사이즈
            new_width = 32
            new_height = 32
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 기본 모델로 예측
            results = ai_pipe(image)
        
        # 신뢰도 계산 및 검증
        confidence_score = results[0]['score']
        confidence_percentage = round(confidence_score * 100, 2)
        
        # 강력한 신뢰도 검증 및 보정
        opposite_label = "REAL" if results[0]['label'] == "FAKE" else "FAKE"
        opposite_score = 1.0 - confidence_score
        opposite_percentage = round(opposite_score * 100, 2)
        
        # 1. 매우 낮은 신뢰도 (30% 미만) - 반대 결과로 강제 보정
        if confidence_percentage < 30:
            final_label = opposite_label
            final_confidence = opposite_percentage
            print(f"매우 낮은 신뢰도로 인한 강제 보정: {results[0]['label']} ({confidence_percentage}%) -> {final_label} ({final_confidence}%)")
        
        # 2. 낮은 신뢰도 (50% 미만) - 더 높은 신뢰도 선택
        elif confidence_percentage < 50:
            if opposite_percentage > confidence_percentage:
                final_label = opposite_label
                final_confidence = opposite_percentage
                print(f"낮은 신뢰도로 인한 재평가: {results[0]['label']} ({confidence_percentage}%) -> {final_label} ({final_confidence}%)")
            else:
                final_label = results[0]['label']
                final_confidence = confidence_percentage
        
        # 3. FAKE 판단이지만 신뢰도가 70% 미만인 경우 - REAL로 보정 (실제 이미지 오판 방지)
        elif results[0]['label'] == "FAKE" and confidence_percentage < 70:
            final_label = "REAL"
            final_confidence = opposite_percentage
            print(f"FAKE 오판 방지 보정: FAKE ({confidence_percentage}%) -> REAL ({final_confidence}%)")
        
        # 4. 정상적인 경우
        else:
            final_label = results[0]['label']
            final_confidence = confidence_percentage
        
        # 결과 정리
        preprocessing_method = "원본크기" if model is not None else "중앙크롭"
        prediction = {
            "predicted_label": final_label,
            "confidence": final_confidence,
            "all_scores": {},
            "image_size": f"{width}x{height} -> {new_width}x{new_height} ({preprocessing_method})",
            "model_type": "개선된 모델" if model is not None else "기본 모델",
            "low_confidence_warning": final_confidence < 80  # 80% 미만이면 경고
        }
        
        # 모든 클래스 점수 추가
        for result in results:
            prediction["all_scores"][result['label']] = round(result['score'] * 100, 2)
        
        # 디버깅 정보 출력
        print(f"이미지 크기: {width}x{height} -> {new_width}x{new_height} ({preprocessing_method})")
        print(f"예측 결과: {results[0]['label']} (신뢰도: {results[0]['score']:.4f})")
        print(f"최종 결과: {final_label} (신뢰도: {final_confidence}%)")
        print(f"프론트엔드로 전송할 confidence 값: {prediction['confidence']}")
        print(f"사용된 모델: {prediction['model_type']}")
        
        return prediction
        
    except Exception as e:
        return {"error": f"예측 중 오류 발생: {str(e)}"}

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """파일 업로드 및 예측"""
    if 'file' not in request.files:
        return jsonify({"error": "파일이 선택되지 않았습니다."})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "파일이 선택되지 않았습니다."})
    
    if file and allowed_file(file.filename):
        # 파일 저장
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 예측 수행
        result = predict_image(filepath)
        
        # 과신 탐지 제거됨
        
        # 피드백을 위해 파일 경로 저장
        result['temp_file_path'] = filepath
        
        return jsonify(result)
    
    return jsonify({"error": "지원되지 않는 파일 형식입니다."})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API 엔드포인트"""
    if 'file' not in request.files:
        return jsonify({"error": "파일이 필요합니다."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "파일이 필요합니다."}), 400
    
    if file and allowed_file(file.filename):
        # 임시 파일로 저장
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 예측 수행
        result = predict_image(filepath)
        
        # 임시 파일 삭제
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(result)
    
    return jsonify({"error": "지원되지 않는 파일 형식입니다."}), 400

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """피드백 제출 엔드포인트"""
    try:
        data = request.get_json()
        
        image_path = data.get('image_path')
        predicted_label = data.get('predicted_label')
        predicted_confidence = data.get('predicted_confidence')
        user_feedback = data.get('user_feedback')  # "REAL" 또는 "FAKE"
        is_correct = data.get('is_correct', True)
        
        if not all([image_path, predicted_label, predicted_confidence]):
            return jsonify({"error": "필수 데이터가 누락되었습니다."}), 400
        
        # 피드백 저장
        success = feedback_system.add_feedback(
            image_path=image_path,
            predicted_label=predicted_label,
            predicted_confidence=predicted_confidence,
            user_feedback=user_feedback,
            is_correct=is_correct
        )
        
        if success:
            print(f"피드백 저장 완료: {user_feedback} (신뢰도: {predicted_confidence:.3f})")
            
            # 실시간 학습 시스템에 피드백 추가 (안정화된 버전)
            try:
                if realtime_learning is not None:
                    # 잘못된 피드백만 학습 큐에 추가
                    if not is_correct:
                        realtime_learning.add_feedback_to_queue(
                            image_path=image_path,
                            predicted_label=predicted_label,
                            user_feedback=user_feedback,
                            confidence=predicted_confidence
                        )
                        print(f"실시간 학습 큐에 추가: {predicted_label} -> {user_feedback}")
                    else:
                        print(f"정확한 피드백이므로 학습 큐에 추가하지 않음")
                else:
                    print("실시간 학습 시스템이 로드되지 않음")
            except Exception as e:
                print(f"실시간 학습 시스템 오류: {e}")
            
            return jsonify({"success": True, "message": "피드백이 저장되었습니다."})
        else:
            print(f"피드백 저장 실패 - 데이터: {data}")
            return jsonify({"error": "피드백 저장에 실패했습니다."}), 500
            
    except Exception as e:
        return jsonify({"error": f"피드백 처리 중 오류: {str(e)}"}), 500

@app.route('/feedback/stats')
def feedback_stats():
    """피드백 통계 조회 - 실시간 학습 통계 포함"""
    try:
        # 기본 피드백 통계
        stats = feedback_system.get_feedback_stats()
        
        # 실시간 학습 통계 추가
        if realtime_learning is not None:
            learning_stats = realtime_learning.get_learning_stats()
            stats['realtime_learning'] = learning_stats
        else:
            stats['realtime_learning'] = {
                'total_feedback': 0,
                'processed_feedback': 0,
                'pending_feedback': 0,
                'is_learning': False,
                'performance_history_count': 0,
                'last_learning_time': None
            }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": f"통계 조회 중 오류: {str(e)}"}), 500

@app.route('/feedback/export', methods=['POST'])
def export_feedback():
    """피드백 데이터를 훈련용으로 내보내기"""
    try:
        exported_count = feedback_system.export_feedback_for_training()
        return jsonify({
            "message": f"{exported_count}개의 피드백이 훈련 데이터로 내보내졌습니다.",
            "exported_count": exported_count
        })
    except Exception as e:
        return jsonify({"error": f"내보내기 중 오류: {str(e)}"}), 500

@app.route('/stats')
def stats_page():
    """통계 페이지"""
    return render_template('stats.html')

@app.route('/features')
def features():
    """기능 소개 페이지"""
    return render_template('features.html')

@app.route('/about')
def about():
    """사이트 정보 페이지"""
    return render_template('about.html')

@app.route('/health')
def health_check():
    """헬스 체크 엔드포인트"""
    return jsonify({
        "status": "healthy",
        "model_loaded": ai_pipe is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
