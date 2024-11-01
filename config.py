from pathlib import Path

# 프로젝트 루트 디렉토리
ROOT_DIR = Path(__file__).parent

# 데이터 디렉토리
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# 출력 디렉토리
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 임시 파일 저장 디렉토리
EXTRACT_DIR = ROOT_DIR / "extract"
EXTRACT_DIR.mkdir(exist_ok=True)

# 학습 결과 저장 디렉토리
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# 로그 디렉토리
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)