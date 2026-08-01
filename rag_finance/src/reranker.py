import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


class Reranker:

    def __init__(

        self,

        model_name="BAAI/bge-reranker-base",

    ):

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
        )

        self.model.eval()

    def rerank(

        self,

        query,

        candidates,

        top_k=10,

    ):

        reranked = []

        for candidate in candidates:

            final_score, cosine, bm25, document = candidate

            inputs = self.tokenizer(

                query,

                document.page_content,

                truncation=True,

                max_length=512,

                return_tensors="pt",

            )

            with torch.no_grad():

                score = self.model(**inputs).logits.squeeze().item()

            reranked.append(

                (

                    score,

                    final_score,

                    cosine,

                    bm25,

                    document,

                )

            )

        reranked.sort(

            key=lambda x: x[0],

            reverse=True,

        )

        return reranked[:top_k]