from utils.file_extract import extract_files
from utils.text_classifier import TextClassifier
from data.training_data import period_data, support_data
import shutil

# 실행마다 models, extract, output, logs 디렉토리 초기화
shutil.rmtree("models", ignore_errors=True)
shutil.rmtree("extract", ignore_errors=True)
shutil.rmtree("output", ignore_errors=True)
shutil.rmtree("logs", ignore_errors=True)

def main():
    # 파일 추출 실행
    print("=== 파일 추출 시작 ===")
    extract_files()
    print("\n=== 파일 추출 완료 ===")

    # 분류기 초기화
    classifier = TextClassifier()

     # 학습 데이터가 있는 경우에만 학습 수행
    if not classifier.model_path.exists():
        print("모델 학습 시작...")
        train_dataset, val_dataset = classifier.prepare_data(period_data, support_data)
        classifier.train(train_dataset, val_dataset)
        print("모델 학습 완료!")

    # 문서 처리 및 결과 저장
    print("문서 분류 시작...")
    classifier.process_documents()
    print("문서 분류 완료!")    

if __name__ == "__main__":
    main()