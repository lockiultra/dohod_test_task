import nltk
from nltk.translate.bleu_score import sentence_bleu
from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm

nltk.download('stopwords')
nltk.download('punkt')

import warnings
warnings.filterwarnings('ignore')


class Metrics:
    def __init__(self, model_api: str = 'ai-forever/rugpt3small_based_on_gpt2', dataset: str = 'd0rj/truthful_qa-gen-ru') -> None:
        self.model_pipeline = pipeline('text-generation', model=model_api)
        self.dataset = load_dataset(dataset)
        self.questions = self.dataset['validation']['question']
        self.answers = self.dataset['validation']['best_answer']
    
    def evaluate_metrics(self, model_answer: str, true_answer: str) -> dict:
        model_answer = model_answer.lower()
        true_answer = true_answer.lower()

        score = sentence_bleu([model_answer], true_answer)

        return score
    
    def get_scores(self) -> None:
        score = 0
        n_samples = len(self.answers)

        for question, answer in tqdm(zip(self.questions[2:], self.answers[2:])):
            model_answer = self.model_pipeline(question, return_full_text=False, max_length=32, max_new_tokens=10)[0]['generated_text']
            idx_score = self.evaluate_metrics(model_answer, answer)
            score += idx_score

        print(score / n_samples)