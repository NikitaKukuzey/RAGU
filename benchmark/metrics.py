from bert_score import BERTScorer
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy

from benchmark.common import BaseMetric, settings


class BertScore(BaseMetric):
    def compute(self, query, prediction, reference, context):
        bert_scorer = BERTScorer(
            lang="ru",
            device="cuda"
        )
        return float(bert_scorer.score(
            cands=list(prediction),
            refs=list(reference)
        )[-1].mean())

    def __name__(self):
        return "Bert-Score"


class LLMJudgeEvaluator(BaseMetric):
    def __init__(self, client):
        self.openai_client = client

    def _extract_score(self, text: str) -> float:
        """Extract numerical score from text"""
        import re
        match = re.search(r'\b[0-9](?:\.\d+)?\b', text)
        return float(match.group(0)) if match else 4.5

    def _gpt_judge(self, question: str, reference: str, candidate: str, valid: str) -> float:
        """Use GPT as judge"""

        prompt = f"""

        Вопрос: {question}

        Информация на которой был сделан ответ: {reference}

        Ответ кандидата: {candidate}
        
        Эталонный ответ:{valid}

        Оцени ответ кандидата и качество подобранной информации по шкале от 1 до 5, учитывая точность, полноту и соответствие эталону. 
        Ответь только числом от 1 до 5."""

        response = self.openai_client.chat.completions.create(
            model=settings.llm_model_name,
            messages=[
                {"role": "system", "content": "Ты — опытный оценщик ответов ИИ. Оцени ответ кандидата на вопрос ниже:"},
                {"role": "user", "content": prompt}
            ]
        )
        return self._extract_score(response.choices[0].message.content.strip())

    def compute(self, query, prediction, reference, context) -> float:
        return self._gpt_judge(question=query,
                               reference=context,
                               candidate=prediction,
                               valid=reference)


class FaithfulnessMetric(BaseMetric):
    def __init__(self, llm):
        self.llm = llm
        self.scorer = Faithfulness(llm=self.llm)

    def compute(self, query, prediction, reference, context):
        result = []
        for q, p, _, c in zip(query, prediction, reference, context):
            sample = SingleTurnSample(
                user_input=q,
                response=p,
                retrieved_contexts=c
            )
            result.append(self.scorer.single_turn_score(sample))
        return sum(result) / len(result) if result else 0

    def __name__(self):
        return "Faithfulness"
    
    
class ResponseRelevancyMetric(BaseMetric):
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.scorer = ResponseRelevancy(
            llm=self.llm, embeddings=self.embeddings)

    def compute(self, query, prediction, reference, context):
        result = []
        for q, p, _, c in zip(query, prediction, reference, context):
            sample = SingleTurnSample(
                user_input=q,
                response=p,
                retrieved_contexts=c
            )
            result.append(self.scorer.single_turn_score(sample))
        return sum(result) / len(result) if result else 0

    def __name__(self):
        return "ResponseRelevancy"
    

