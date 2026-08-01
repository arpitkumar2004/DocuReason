class CandidateMerger:

    def merge(
        self,
        text_results,
        table_results,
        top_k=40,
    ):

        merged = []

        merged.extend(text_results)
        merged.extend(table_results)

        merged.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        return merged[:top_k]