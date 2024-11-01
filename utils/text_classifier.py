from pathlib import Path
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
import numpy as np
from datasets import Dataset
import evaluate
from config import EXTRACT_DIR, OUTPUT_DIR, MODEL_DIR
import json

class TextClassifier:
    def __init__(self, model_name="klue/roberta-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_path = MODEL_DIR / "text_classifier"
        
        # 모델 로드 또는 초기화
        if self.model_path.exists():
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=2,  # 지원기간과 지원내용만 분류
                ignore_mismatched_sizes=True  # 사이즈 불일치 무시
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2
            )
        
        self.model.to(self.device)

    def prepare_data(self, period_data, support_data):
        # 데이터 통합
        all_data = period_data + support_data 
        texts = [item["text"] for item in all_data]
        labels = [item["label"] for item in all_data]
        
        # 학습/검증 데이터 분할
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Dataset 객체 생성
        train_dataset = Dataset.from_dict({
            "text": train_texts,
            "label": train_labels
        })
        val_dataset = Dataset.from_dict({
            "text": val_texts,
            "label": val_labels
        })
        
        # 토크나이징
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        val_dataset = val_dataset.map(self.tokenize_function, batched=True)
        
        return train_dataset, val_dataset

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,  # 문장이 짧으므로 128로 충분
            padding="max_length"
        )

    def compute_metrics(self, eval_pred):
        metric = evaluate.load("accuracy")
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    def train(self, train_dataset, val_dataset):
        training_args = TrainingArguments(
            output_dir=str(self.model_path),
            learning_rate=5e-5,  # 학습률 증가
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=5,  # epoch 증가
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            warmup_steps=50,  # warmup 추가
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        trainer.save_model(str(self.model_path))

    def classify_text(self, text: str):
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1).item()
            confidence = predictions[0][predicted_class].item()

        return predicted_class, confidence

    def process_documents(self):

        # output 디렉토리가 없으면 생성
        OUTPUT_DIR.mkdir(exist_ok=True)
        # 신청기간 패턴: 숫자로 시작하고 "기간" 또는 "신청기간"이 포함된 문장
        period_patterns = [
            "신청기간",
            "접수기간",
            "모집기간",
            "신 청 기 간"  # 공백이 있는 경우도 포함
        ]
        
        # 사업비 패턴: 숫자로 시작하고 "사업비", "지원금액", "총사업비" 등이 포함된 문장
        support_patterns = [
        "사업비",
        "총사업비",
        "지원금액",
        "지원규모",
        "지원단가",
        "백만원",
        "억원",
        "원/20kg",
        "원/포"
        ]
        
        for txt_file in EXTRACT_DIR.glob("*.txt"):
            try:
                text = txt_file.read_text(encoding='utf-8')
                sentences = [s.strip() for s in text.split('\n') if s.strip()]
                
                results = {0: [], 1: []}
                
                for sentence in sentences:
                    label, confidence = self.classify_text(sentence)
                    
                    if confidence >= 0.85:
                        # 지원기간인 경우
                        if label == 0 and any(pattern in sentence for pattern in period_patterns):
                            results[label].append({
                                "text": sentence,
                                "confidence": confidence
                            })
                        # 지원내용인 경우
                        elif label == 1 and any(pattern in sentence for pattern in support_patterns):
                            results[label].append({
                                "text": sentence,
                                "confidence": confidence
                            })
                        
                # 각 txt 파일별로 결과 저장
                output_path = OUTPUT_DIR / f"{txt_file.stem}_results.txt"
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("=== 지원 기간 관련 문장 ===\n")
                    for item in sorted(results[0], key=lambda x: x['confidence'], reverse=True):
                        f.write(f"추출 문장 [신뢰도: {item['confidence']:.2f}]:\n")
                        f.write(f"{item['text']}\n\n")
                    
                    f.write("\n=== 지원 내용 관련 문장 ===\n")
                    for item in sorted(results[1], key=lambda x: x['confidence'], reverse=True):
                        f.write(f"추출 문장 [신뢰도: {item['confidence']:.2f}]:\n")
                        f.write(f"{item['text']}\n\n")
            
            except Exception as e:
                print(f"Error processing {txt_file}: {e}")