#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTML 분석 리포트 생성
"""

import json
import pandas as pd
from datetime import datetime

def create_html_report():
    """HTML 분석 리포트 생성"""
    with open('comprehensive_test_results.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    analysis = results['analysis_table']
    detailed_results = results['detailed_results']
    
    html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>딥페이크 탐지 모델 성능 분석 리포트</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 30px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            font-size: 1.2em;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .confusion-matrix {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
        }}
        .confusion-matrix table {{
            margin: 0 auto;
            border-collapse: collapse;
        }}
        .confusion-matrix th, .confusion-matrix td {{
            border: 1px solid #dee2e6;
            padding: 10px 20px;
            text-align: center;
        }}
        .confusion-matrix th {{
            background-color: #e9ecef;
            font-weight: bold;
        }}
        .performance-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #fff;
            border: 1px solid #e1e8ed;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}
        .metric-card .label {{
            font-size: 0.9em;
            color: #657786;
            margin-bottom: 5px;
        }}
        .metric-card .value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #1da1f2;
        }}
        .warning {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 15px;
            margin: 20px 0;
        }}
        .success {{
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 8px;
            padding: 15px;
            margin: 20px 0;
        }}
        .recommendations {{
            background: #e7f3ff;
            border-left: 4px solid #007bff;
            padding: 20px;
            margin: 20px 0;
        }}
        .recommendations ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e1e8ed;
            color: #657786;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>딥페이크 탐지 모델 성능 분석 리포트</h1>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>전체 정확도</h3>
                <div class="value">{analysis['전체 통계']['전체 정확도']}</div>
                <p>목표: 80% 이상</p>
            </div>
            <div class="summary-card">
                <h3>총 테스트 이미지</h3>
                <div class="value">{analysis['전체 통계']['총 이미지 수']}개</div>
                <p>FAKE: 106개, REAL: 66개</p>
            </div>
            <div class="summary-card">
                <h3>정확한 예측</h3>
                <div class="value">{analysis['전체 통계']['정확한 예측']}개</div>
                <p>성공률: 72.67%</p>
            </div>
            <div class="summary-card">
                <h3>틀린 예측</h3>
                <div class="value">{analysis['전체 통계']['틀린 예측']}개</div>
                <p>개선 필요</p>
            </div>
        </div>

        <h2>클래스별 성능 분석</h2>
        
        <div class="performance-metrics">
            <div class="metric-card">
                <div class="label">FAKE 이미지 정확도</div>
                <div class="value">{analysis['FAKE 이미지 분석']['FAKE 정확도']}</div>
            </div>
            <div class="metric-card">
                <div class="label">REAL 이미지 정확도</div>
                <div class="value">{analysis['REAL 이미지 분석']['REAL 정확도']}</div>
            </div>
            <div class="metric-card">
                <div class="label">FAKE 올바른 분류</div>
                <div class="value">{analysis['FAKE 이미지 분석']['올바르게 FAKE로 분류']}개</div>
            </div>
            <div class="metric-card">
                <div class="label">REAL 올바른 분류</div>
                <div class="value">{analysis['REAL 이미지 분석']['올바르게 REAL로 분류']}개</div>
            </div>
        </div>

        <h2>혼동 행렬 (Confusion Matrix)</h2>
        <div class="confusion-matrix">
            <table>
                <tr>
                    <th>실제\\예측</th>
                    <th>FAKE</th>
                    <th>REAL</th>
                </tr>
                <tr>
                    <th>FAKE</th>
                    <td>{analysis['FAKE 이미지 분석']['올바르게 FAKE로 분류']}</td>
                    <td>{analysis['FAKE 이미지 분석']['REAL로 오분류']}</td>
                </tr>
                <tr>
                    <th>REAL</th>
                    <td>{analysis['REAL 이미지 분석']['FAKE로 오분류']}</td>
                    <td>{analysis['REAL 이미지 분석']['올바르게 REAL로 분류']}</td>
                </tr>
            </table>
        </div>

        <h2>성능 지표</h2>
        <div class="performance-metrics">
            <div class="metric-card">
                <div class="label">평균 신뢰도</div>
                <div class="value">{analysis['성능 지표']['평균 신뢰도']}</div>
            </div>
            <div class="metric-card">
                <div class="label">신뢰도 표준편차</div>
                <div class="value">{analysis['성능 지표']['신뢰도 표준편차']}</div>
            </div>
            <div class="metric-card">
                <div class="label">평균 추론 시간</div>
                <div class="value">{analysis['성능 지표']['평균 추론 시간']}</div>
            </div>
            <div class="metric-card">
                <div class="label">최고 신뢰도</div>
                <div class="value">{analysis['성능 지표']['최고 신뢰도']}</div>
            </div>
        </div>

        <h2>성능 평가</h2>
        <div class="success">
            <h3>우수한 성능</h3>
            <p><strong>FAKE 이미지 탐지:</strong> {analysis['FAKE 이미지 분석']['FAKE 정확도']} - AI 생성 이미지를 매우 잘 탐지합니다.</p>
        </div>
        
        <div class="warning">
            <h3>개선이 필요한 영역</h3>
            <p><strong>REAL 이미지 탐지:</strong> {analysis['REAL 이미지 분석']['REAL 정확도']} - 실제 이미지를 FAKE로 오분류하는 경우가 많습니다.</p>
        </div>

        <h2>개선 제안사항</h2>
        <div class="recommendations">
            <h3>데이터 및 모델 개선</h3>
            <ul>
                <li><strong>데이터 불균형 해결:</strong> REAL 이미지 데이터 증강 필요</li>
                <li><strong>전처리 개선:</strong> 다양한 이미지 크기와 비율에 대한 적응적 처리</li>
                <li><strong>앙상블 모델:</strong> 여러 모델의 예측 결과 결합</li>
                <li><strong>피드백 학습:</strong> 사용자 피드백을 통한 지속적 모델 개선</li>
            </ul>
        </div>

        <h2>결론</h2>
        <div class="warning">
            <p><strong>현재 모델의 전체 정확도는 {analysis['전체 통계']['전체 정확도']}로,</strong></p>
            <p>FAKE 이미지 탐지는 우수하지만 REAL 이미지 탐지에서 개선이 필요합니다.</p>
            <p>특히 REAL 이미지의 50%가 FAKE로 오분류되고 있어, 이는 실제 사용자 경험에 부정적인 영향을 줄 수 있습니다.</p>
        </div>

        <div class="footer">
            <p>테스트 일시: {results['test_info']['test_date']}</p>
            <p>모델: {results['test_info']['model_name']}</p>
            <p>총 테스트 이미지: {results['test_info']['total_images']}개</p>
        </div>
    </div>
</body>
</html>
"""
    
    with open('analysis_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("HTML 분석 리포트가 'analysis_report.html'로 생성되었습니다.")

if __name__ == "__main__":
    create_html_report()

