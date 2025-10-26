#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Image Detector Model VIT - 추론 스크립트
훈련된 모델을 사용하여 이미지가 실제 이미지인지 AI 생성 이미지인지 예측합니다.
"""

import argparse
import os
from pathlib import Path
from PIL import Image
import torch
from transformers import pipeline, ViTImageProcessor, ViTForImageClassification
import warnings

warnings.filterwarnings("ignore")


class AIImageDetector:
    """
    AI 이미지 탐지기 클래스
    """
    
    def __init__(self, model_path, device=None):
        """
        AI 이미지 탐지기 초기화
        
        Args:
            model_path (str): 훈련된 모델의 경로
            device (str): 사용할 디바이스 ('cuda', 'cpu', None)
        """
        self.model_path = model_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"디바이스: {self.device}")
        print(f"모델 로딩 중: {model_path}")
        
        # 파이프라인 생성
        self.pipe = pipeline(
            'image-classification', 
            model=model_path, 
            device=0 if self.device == 'cuda' else -1
        )
        
        # 모델 정보 출력
        self._print_model_info()
    
    def _print_model_info(self):
        """
        모델 정보를 출력하는 함수
        """
        try:
            # 모델과 프로세서 로드
            model = ViTForImageClassification.from_pretrained(self.model_path)
            processor = ViTImageProcessor.from_pretrained(self.model_path)
            
            print(f"모델 클래스 수: {model.config.num_labels}")
            print(f"라벨 매핑: {model.config.id2label}")
            print(f"이미지 크기: {processor.size}")
            
        except Exception as e:
            print(f"모델 정보 로드 중 오류: {e}")
    
    def predict_single_image(self, image_path, return_confidence=True):
        """
        단일 이미지에 대한 예측 수행
        
        Args:
            image_path (str): 이미지 파일 경로
            return_confidence (bool): 신뢰도 점수 반환 여부
            
        Returns:
            dict: 예측 결과
        """
        try:
            # 이미지 로드
            image = Image.open(image_path).convert('RGB')
            
            # 예측 수행
            results = self.pipe(image)
            
            # 결과 정리
            prediction = {
                'image_path': image_path,
                'predicted_label': results[0]['label'],
                'confidence': results[0]['score']
            }
            
            if return_confidence:
                # 모든 클래스에 대한 신뢰도 점수 추가
                prediction['all_scores'] = {result['label']: result['score'] for result in results}
            
            return prediction
            
        except Exception as e:
            return {
                'image_path': image_path,
                'error': str(e)
            }
    
    def predict_batch(self, image_paths, return_confidence=True):
        """
        여러 이미지에 대한 배치 예측 수행
        
        Args:
            image_paths (list): 이미지 파일 경로 리스트
            return_confidence (bool): 신뢰도 점수 반환 여부
            
        Returns:
            list: 예측 결과 리스트
        """
        results = []
        
        print(f"{len(image_paths)}개의 이미지 예측 중...")
        
        for i, image_path in enumerate(image_paths):
            print(f"진행률: {i+1}/{len(image_paths)} - {os.path.basename(image_path)}")
            result = self.predict_single_image(image_path, return_confidence)
            results.append(result)
        
        return results
    
    def predict_directory(self, directory_path, extensions=None, return_confidence=True):
        """
        디렉토리 내 모든 이미지에 대한 예측 수행
        
        Args:
            directory_path (str): 이미지가 있는 디렉토리 경로
            extensions (list): 지원할 이미지 확장자 리스트
            return_confidence (bool): 신뢰도 점수 반환 여부
            
        Returns:
            list: 예측 결과 리스트
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        # 디렉토리에서 이미지 파일 찾기
        image_paths = []
        directory = Path(directory_path)
        
        for ext in extensions:
            image_paths.extend(directory.glob(f'*{ext}'))
            image_paths.extend(directory.glob(f'*{ext.upper()}'))
        
        image_paths = [str(path) for path in image_paths]
        
        if not image_paths:
            print(f"디렉토리 {directory_path}에서 이미지 파일을 찾을 수 없습니다.")
            return []
        
        print(f"{len(image_paths)}개의 이미지 파일을 찾았습니다.")
        return self.predict_batch(image_paths, return_confidence)
    
    def print_prediction_result(self, result):
        """
        예측 결과를 보기 좋게 출력하는 함수
        
        Args:
            result (dict): 예측 결과
        """
        if 'error' in result:
            print(f"❌ 오류: {result['image_path']} - {result['error']}")
            return
        
        image_name = os.path.basename(result['image_path'])
        label = result['predicted_label']
        confidence = result['confidence']
        
        # 라벨에 따른 이모지 설정
        emoji = "🎨" if label == "FAKE" else "📷"
        
        print(f"{emoji} {image_name}")
        print(f"   예측: {label}")
        print(f"   신뢰도: {confidence:.4f}")
        
        if 'all_scores' in result:
            print("   전체 점수:")
            for label_name, score in result['all_scores'].items():
                print(f"     {label_name}: {score:.4f}")
        print()


def main():
    """
    메인 함수
    """
    parser = argparse.ArgumentParser(description='AI Image Detector - 이미지 분류')
    parser.add_argument('--model_path', type=str, required=True,
                       help='훈련된 모델의 경로')
    parser.add_argument('--image_path', type=str,
                       help='예측할 단일 이미지 경로')
    parser.add_argument('--directory', type=str,
                       help='예측할 이미지들이 있는 디렉토리 경로')
    parser.add_argument('--output_file', type=str,
                       help='결과를 저장할 파일 경로 (CSV 형식)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='사용할 디바이스 (cuda/cpu)')
    parser.add_argument('--no_confidence', action='store_true',
                       help='신뢰도 점수 출력하지 않기')
    
    args = parser.parse_args()
    
    # 입력 검증
    if not args.image_path and not args.directory:
        print("오류: --image_path 또는 --directory 중 하나를 지정해야 합니다.")
        return
    
    if args.image_path and args.directory:
        print("오류: --image_path와 --directory 중 하나만 지정할 수 있습니다.")
        return
    
    # 모델 경로 검증
    if not os.path.exists(args.model_path):
        print(f"오류: 모델 경로가 존재하지 않습니다: {args.model_path}")
        return
    
    print("=== AI Image Detector ===")
    
    try:
        # AI 이미지 탐지기 초기화
        detector = AIImageDetector(args.model_path, args.device)
        
        results = []
        
        if args.image_path:
            # 단일 이미지 예측
            if not os.path.exists(args.image_path):
                print(f"오류: 이미지 파일이 존재하지 않습니다: {args.image_path}")
                return
            
            print(f"\n단일 이미지 예측: {args.image_path}")
            result = detector.predict_single_image(args.image_path, not args.no_confidence)
            results = [result]
            detector.print_prediction_result(result)
        
        elif args.directory:
            # 디렉토리 내 이미지들 예측
            if not os.path.exists(args.directory):
                print(f"오류: 디렉토리가 존재하지 않습니다: {args.directory}")
                return
            
            print(f"\n디렉토리 내 이미지들 예측: {args.directory}")
            results = detector.predict_directory(args.directory, return_confidence=not args.no_confidence)
            
            # 결과 출력
            for result in results:
                detector.print_prediction_result(result)
        
        # 결과를 파일로 저장
        if args.output_file and results:
            import pandas as pd
            
            # 결과를 DataFrame으로 변환
            df_data = []
            for result in results:
                if 'error' not in result:
                    row = {
                        'image_path': result['image_path'],
                        'predicted_label': result['predicted_label'],
                        'confidence': result['confidence']
                    }
                    if 'all_scores' in result:
                        for label, score in result['all_scores'].items():
                            row[f'score_{label}'] = score
                    df_data.append(row)
            
            if df_data:
                df = pd.DataFrame(df_data)
                df.to_csv(args.output_file, index=False, encoding='utf-8-sig')
                print(f"\n결과가 {args.output_file}에 저장되었습니다.")
        
        # 통계 출력
        if results and len(results) > 1:
            valid_results = [r for r in results if 'error' not in r]
            if valid_results:
                real_count = sum(1 for r in valid_results if r['predicted_label'] == 'REAL')
                fake_count = sum(1 for r in valid_results if r['predicted_label'] == 'FAKE')
                
                print(f"\n=== 예측 통계 ===")
                print(f"총 이미지 수: {len(valid_results)}")
                print(f"실제 이미지 (REAL): {real_count} ({real_count/len(valid_results)*100:.1f}%)")
                print(f"AI 생성 이미지 (FAKE): {fake_count} ({fake_count/len(valid_results)*100:.1f}%)")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        raise


if __name__ == "__main__":
    main()

