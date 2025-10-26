#!/usr/bin/env python3
"""
AI Image Detector - Main Application Entry Point
Render.com 배포를 위한 진입점
"""

import os
import sys

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# web_app 모듈에서 Flask 앱 가져오기
from web_app import app

if __name__ == '__main__':
    # Render 환경 변수에서 포트 가져오기
    port = int(os.environ.get('PORT', 5000))
    
    # 프로덕션 모드로 실행
    app.run(
        debug=False,
        host='0.0.0.0',
        port=port,
        threaded=True
    )
