import ollama
import numpy as np

from tqdm import tqdm


class EmbeddingModel:

    def __init__(
        self,
        model="nomic-embed-text",
    ):

        self.model = model

    def embed_documents(
        self,
        documents,
    ):

        embeddings = []

        for doc in tqdm(documents):

            response = ollama.embed(

                model=self.model,

                input=doc.page_content,

            )

            embeddings.append(
                response["embeddings"][0]
            )

        return np.array(
            embeddings,
            dtype=np.float32,
        )