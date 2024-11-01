# PDF/HWP 문서 정보 추출기

공고문에서 지원 기간과 지원 금액 정보를 자동으로 추출하는 프로젝트입니다.

## 프로젝트 구조

pdf_extract_v2/
├── config.py # 디렉토리 설정
├── main.py # 메인 실행 파일
├── data/ # 원본 PDF/HWP 파일 저장
├── extract/ # 텍스트 변환 파일 저장
├── output/ # 추출 결과 저장
├── models/ # 학습된 모델 저장
├── logs/ # 로그 파일 저장
└── utils/
├── file_extract.py # 파일 추출 관련 유틸리티
└── text_classifier.py # 텍스트 분류 관련 유틸리티

## 기능 설명

### 1. 파일 변환 (file_extract.py)

- PDF/HWP 파일을 텍스트로 변환
- PDF: pdfplumber 라이브러리 사용
- HWP: hwp5txt 도구 사용
- 변환된 텍스트는 extract 폴더에 저장

### 2. 텍스트 분류 (text_classifier.py)

- klue/roberta-base 모델 기반
- 두 가지 카테고리로 분류:
  - 지원 기간 관련 정보
  - 지원 금액 관련 정보
- 높은 신뢰도(0.85 이상)의 결과만 추출

## 설치 방법

1. 필요한 라이브러리 설치:

```bash
pip install transformers torch pdfplumber hwp5txt
```

## 실행

1. 명령어:

python main.py
