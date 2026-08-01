from collections import OrderedDict


class PromptBuilder:

    def __init__(
        self,
        max_characters=12000,
    ):

        self.max_characters = max_characters

    # --------------------------------------------------
    # Build Prompt
    # --------------------------------------------------

    def build(
        self,
        query,
        reranked_results,
    ):

        prompt = f"""
You are an expert assistant for financial reports.

Rules:

1. Answer ONLY using the provided evidence.

2. Never invent numbers.

3. If multiple values exist
   (different years, business segments,
   adjusted vs GAAP),
   explain each separately.

4. Prefer evidence matching
   the requested year.

5. Always mention page numbers.

6. If the answer cannot be found,
   explicitly say so.

7. Never merge unrelated tables.

========================================================

QUESTION

{query}

========================================================

EVIDENCE

"""

        current_size = len(prompt)

        ##################################################
        # Keep page order according to reranker
        ##################################################

        pages = OrderedDict()

        for item in reranked_results:

            score, cosine, bm25, rerank_score, document = item

            page = document.metadata.get(
                "page",
                -1,
            )

            if page not in pages:

                pages[page] = []

            pages[page].append(

                (
                    score,
                    cosine,
                    bm25,
                    rerank_score,
                    document,
                )

            )

        ##################################################
        # Build page by page
        ##################################################

        for page, evidence_list in pages.items():

            page_text = "\n"
            page_text += "=" * 70
            page_text += "\n"
            page_text += f"PAGE {page}\n"
            page_text += "=" * 70
            page_text += "\n\n"

            for idx, item in enumerate(
                evidence_list,
                start=1,
            ):

                (
                    score,
                    cosine,
                    bm25,
                    rerank_score,
                    document,
                ) = item

                meta = document.metadata

                page_text += "-" * 60
                page_text += "\n"

                page_text += f"Evidence {idx}\n\n"

                page_text += (
                    f"Type : {meta.get('type','')}\n"
                )

                if meta.get("label"):

                    page_text += (
                        f"Label : {meta.get('label')}\n"
                    )

                if meta.get("table") is not None:

                    page_text += (
                        f"Table ID : {meta.get('table')}\n"
                    )

                if meta.get("title"):

                    page_text += (
                        f"Title : {meta.get('title')}\n"
                    )

                page_text += (
                    f"Page : {page}\n\n"
                )

                ##################################################
                # Context
                ##################################################

                context = meta.get(
                    "context",
                    "",
                )

                if context:

                    page_text += "Context\n"
                    page_text += "-------\n"
                    page_text += context.strip()
                    page_text += "\n\n"

                ##################################################
                # Content
                ##################################################

                page_text += "Content\n"
                page_text += "-------\n"

                page_text += document.page_content.strip()

                page_text += "\n\n"

            ##################################################
            # Token / character budget
            ##################################################

            if (
                current_size + len(page_text)
                > self.max_characters
            ):
                break

            prompt += page_text
            current_size += len(page_text)

        return prompt