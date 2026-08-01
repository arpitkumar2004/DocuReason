from langchain_core.documents import Document


class TextProcessor:

    def process(self, text_blocks):

        # -------------------------------------------------
        # Step 1 : Clean Docling blocks
        # -------------------------------------------------

        cleaned_blocks = []

        for block in text_blocks:

            text = block.text.strip()

            if len(text) == 0:
                continue

            label = str(block.label).lower()

            # Remove repeated headers / footers
            if "page_header" in label:
                continue

            if "page_footer" in label:
                continue

            page = block.prov[0].page_no + 1

            cleaned_blocks.append(
                {
                    "text": text,
                    "page": page,
                    "label": label,
                }
            )

        documents = []

        # -------------------------------------------------
        # Step 2 : Original semantic blocks
        # -------------------------------------------------

        for block in cleaned_blocks:

            documents.append(

                Document(

                    page_content=block["text"],

                    metadata={

                        "type": "text",

                        "page": block["page"],

                        "label": block["label"],

                        "window": 1,

                    },

                )

            )

        # -------------------------------------------------
        # Step 3 : Smart windows
        # Only create windows for short headings
        # -------------------------------------------------

        seen = set()

        SHORT_BLOCK_THRESHOLD = 12

        for i in range(len(cleaned_blocks)):

            current = cleaned_blocks[i]

            current_words = len(
                current["text"].split()
            )

            # Long paragraph already has context
            if current_words > SHORT_BLOCK_THRESHOLD:
                continue

            pieces = []

            # Previous block
            if i > 0:

                previous = cleaned_blocks[i - 1]

                if previous["page"] == current["page"]:

                    pieces.append(previous["text"])

            # Current block
            pieces.append(current["text"])

            # Next block
            if i < len(cleaned_blocks) - 1:

                nxt = cleaned_blocks[i + 1]

                if nxt["page"] == current["page"]:

                    pieces.append(nxt["text"])

            combined_text = "\n\n".join(pieces).strip()

            # Avoid duplicate windows
            if combined_text in seen:
                continue

            seen.add(combined_text)

            documents.append(

                Document(

                    page_content=combined_text,

                    metadata={

                        "type": "text",

                        "page": current["page"],

                        "label": "window",

                        "window": len(pieces),

                    },

                )

            )

        return documents