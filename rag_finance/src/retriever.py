import re
import numpy as np
import ollama

from rank_bm25 import BM25Okapi


class Retriever:

    def __init__(
        self,
        model="nomic-embed-text",
    ):

        self.model = model

        self.documents = None
        self.bm25 = None

    # --------------------------------------------------
    # Build BM25 index (call once)
    # --------------------------------------------------

    def build_index(
        self,
        documents,
    ):

        self.documents = documents

        corpus = [

            self.tokenize(
                doc.page_content
            )

            for doc in documents

        ]

        self.bm25 = BM25Okapi(corpus)

    # --------------------------------------------------
    # Embed Query
    # --------------------------------------------------

    def embed_query(self, query):

        response = ollama.embed(

            model=self.model,
            input=query,

        )

        return np.array(
            response["embeddings"][0],
            dtype=np.float32,
        )

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    def tokenize(self, text):

        return re.findall(
            r"[A-Za-z0-9.%$]+",
            text.lower(),
        )

    # --------------------------------------------------
    # Cosine
    # --------------------------------------------------

    def cosine_similarity(
        self,
        query_embedding,
        document_embeddings,
    ):

        query_embedding = (
            query_embedding
            / np.linalg.norm(query_embedding)
        )

        document_embeddings = (

            document_embeddings

            / np.linalg.norm(
                document_embeddings,
                axis=1,
                keepdims=True,
            )

        )

        return document_embeddings @ query_embedding

    # --------------------------------------------------
    # BM25
    # --------------------------------------------------

    def bm25_scores(
        self,
        query,
    ):

        scores = self.bm25.get_scores(

            self.tokenize(query)

        )

        scores = np.array(
            scores,
            dtype=np.float32,
        )

        if scores.max() > 0:

            scores /= scores.max()

        return scores

    # --------------------------------------------------
    # Bonus
    # --------------------------------------------------

    def compute_bonus(
        self,
        query,
        document,
    ):

        bonus = 0.0

        text = document.page_content.lower()

        ##################################################
        # Length
        ##################################################

        words = len(document.page_content.split())

        if words >= 80:
            bonus += 0.06

        elif words >= 50:
            bonus += 0.05

        elif words >= 30:
            bonus += 0.03

        elif words >= 15:
            bonus += 0.01

        elif words <= 5:
            bonus -= 0.15

        elif words <= 10:
            bonus -= 0.08

        ##################################################
        # Label
        ##################################################

        label = document.metadata.get(
            "label",
            "",
        )

        if label == "text":
            bonus += 0.08

        elif label == "list_item":
            bonus += 0.06

        elif label == "window":
            bonus += 0.02

        elif label == "section_header":
            bonus -= 0.10

        ##################################################
        # Table
        ##################################################

        if document.metadata.get("type") == "table":

            bonus += 0.05

        ##################################################
        # Numeric
        ##################################################

        digits = len(
            re.findall(
                r"\d",
                text,
            )
        )

        bonus += min(
            digits * 0.001,
            0.04,
        )

        if "%" in text:
            bonus += 0.02

        if "$" in text:
            bonus += 0.01

        ##################################################
        # Keyword overlap
        ##################################################

        query_words = {

            w.lower()

            for w in re.findall(
                r"[A-Za-z]{4,}",
                query,
            )

        }

        doc_words = {

            w.lower()

            for w in re.findall(
                r"[A-Za-z]{4,}",
                text,
            )

        }

        overlap = len(
            query_words & doc_words
        )

        bonus += min(
            overlap * 0.02,
            0.08,
        )

        ##################################################
        # Year
        ##################################################

        query_years = set(
            re.findall(
                r"\b20\d{2}\b",
                query,
            )
        )

        if query_years:

            doc_years = set(
                re.findall(
                    r"\b20\d{2}\b",
                    text,
                )
            )

            if doc_years:

                if query_years & doc_years:

                    bonus += 0.05

                else:

                    bonus -= 0.20

        return bonus

    # --------------------------------------------------
    # Remove duplicate windows
    # --------------------------------------------------

    def remove_duplicates(
        self,
        results,
    ):

        filtered = []

        seen = []

        for item in results:

            score, cosine, bm25, doc = item

            text = doc.page_content.strip()

            duplicate = False

            for prev in seen:

                if (

                    text in prev
                    or prev in text

                ):

                    duplicate = True
                    break

            if duplicate:
                continue

            filtered.append(item)

            seen.append(text)

        return filtered

    # --------------------------------------------------
    # Retrieve
    # --------------------------------------------------

    def retrieve(
        self,
        query,
        embeddings,
        top_k=10,
    ):

        if self.bm25 is None:

            raise RuntimeError(
                "Call build_index(documents) first."
            )

        ##################################################
        # Dense
        ##################################################

        query_embedding = self.embed_query(
            query
        )

        cosine_scores = self.cosine_similarity(
            query_embedding,
            embeddings,
        )

        ##################################################
        # Sparse
        ##################################################

        bm25_scores = self.bm25_scores(
            query
        )

        ##################################################
        # Merge
        ##################################################

        results = []

        for cosine, bm25, document in zip(

            cosine_scores,
            bm25_scores,
            self.documents,

        ):

            bonus = self.compute_bonus(
                query,
                document,
            )

            final_score = (

                0.65 * float(cosine)
                + 0.35 * float(bm25)
                + bonus

            )

            results.append(

                (
                    final_score,
                    float(cosine),
                    float(bm25),
                    document,
                )

            )

        ##################################################

        results.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        results = self.remove_duplicates(
            results
        )

        return results[:top_k]