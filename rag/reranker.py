from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

class Reranker():
    
    def __init__(self):
        model_path = "Dongjin-kr/ko-reranker"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device("mps")
        self.model.to(self.device)
        self.model.eval()
        
    def exp_normalize(self, x):
        b = x.max()
        y = np.exp(x - b)
        return y / y.sum()

        
    def rerank(self, pairs):
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=5000)
            # 입력을 MPS 디바이스로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = scores.cpu().numpy()
            scores = self.exp_normalize(scores)
        return scores