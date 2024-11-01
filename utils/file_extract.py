from pathlib import Path
import subprocess
import traceback
from config import DATA_DIR, EXTRACT_DIR
import pdfplumber

def convert_hwp_to_text(hwp_path: Path | str) -> str | None:
    """HWP 파일을 텍스트로 변환"""
    try:
        # 경로를 Windows 형식으로 정규화
        hwp_path = Path(hwp_path).resolve()
        print(f"\nHWP 변환 디버깅 정보:")
        print(f"1. 입력된 경로: {hwp_path}")
        print(f"2. 파일 존재 여부: {hwp_path.exists()}")
        print(f"3. 현재 작업 디렉토리: {Path.cwd()}")
        
        # 파일 접근 가능 여부 확인
        try:
            with open(hwp_path, 'rb') as f:
                f.read(1)
            print("4. 파일 접근 가능 확인됨")
        except Exception as e:
            print(f"4. 파일 접근 실패: {e}")
            return None

        # hwp5txt 실행
        process = subprocess.run(
            ['hwp5txt', str(hwp_path)],
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=30,
            shell=True
        )
        
        print(f"5. hwp5txt 실행 결과 코드: {process.returncode}")
        if process.stderr:
            print(f"6. 오류 출력: {process.stderr}")
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, 'hwp5txt')
        
        if not process.stdout:
            print("7. 경고: 변환된 텍스트가 비어있습니다")
            return None
            
        print("8. 텍스트 변환 성공")
        return process.stdout

    except Exception as e:
        print(f"HWP 변환 중 오류 발생: {type(e).__name__}: {e}")
        print(f"스택 트레이스: {traceback.format_exc()}")
        return None

def convert_pdf_to_text(pdf_path: Path) -> str | None:
    """PDF 파일을 텍스트로 변환"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        print(f"PDF 변환 중 오류 발생: {e}")
        return None

def save_text_to_file(filepath: Path | str, text: str, encoding: str = 'utf-8') -> bool:
    """텍스트를 파일로 저장"""
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(text, encoding=encoding)
        print(f"파일 저장 성공: {filepath}")
        return True
    except Exception as e:
        print(f"파일 저장 실패: {e}")
        return False

def extract_files():
    """데이터 디렉토리의 파일들을 텍스트로 변환하여 추출"""
    # 기존 추출 디렉토리 내용 삭제
    if EXTRACT_DIR.exists():
        for item in EXTRACT_DIR.iterdir():
            if item.is_file():
                item.unlink()
    
    # PDF와 HWP 파일 찾기
    pdf_files = list(DATA_DIR.glob("**/*.pdf"))
    hwp_files = list(DATA_DIR.glob("**/*.hwp"))
    
    print(f"발견된 PDF 파일: {len(pdf_files)}개")
    print(f"발견된 HWP 파일: {len(hwp_files)}개")
    
    # PDF 파일 처리
    for pdf_path in pdf_files:
        try:
            print(f"\nPDF 파일 처리 중: {pdf_path.name}")
            text = convert_pdf_to_text(pdf_path)
            if text:
                output_path = EXTRACT_DIR / f"{pdf_path.stem}.txt"
                save_text_to_file(output_path, text)
        except Exception as e:
            print(f"PDF 파일 처리 실패 ({pdf_path.name}): {e}")
    
    # HWP 파일 처리
    for hwp_path in hwp_files:
        try:
            print(f"\nHWP 파일 처리 중: {hwp_path.name}")
            text = convert_hwp_to_text(hwp_path)
            if text:
                output_path = EXTRACT_DIR / f"{hwp_path.stem}.txt"
                save_text_to_file(output_path, text)
        except Exception as e:
            print(f"HWP 파일 처리 실패 ({hwp_path.name}): {e}")