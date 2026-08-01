import pandas as pd
from collections import defaultdict
from langchain_core.documents import Document


class TableProcessor:

    # --------------------------------------------------
    # Convert Docling table to DataFrame
    # --------------------------------------------------

    def table_to_dataframe(self, table):

        rows = table.data.num_rows
        cols = table.data.num_cols

        grid = [["" for _ in range(cols)] for _ in range(rows)]
        header_mask = [[False for _ in range(cols)] for _ in range(rows)]

        for cell in table.data.table_cells:

            text = cell.text.strip()

            for r in range(
                cell.start_row_offset_idx,
                cell.end_row_offset_idx,
            ):

                for c in range(
                    cell.start_col_offset_idx,
                    cell.end_col_offset_idx,
                ):

                    grid[r][c] = text

                    if cell.column_header:
                        header_mask[r][c] = True

        return pd.DataFrame(grid), header_mask

    # --------------------------------------------------
    # Merge multi-row headers
    # --------------------------------------------------

    def merge_headers(self, df, header_mask):

        header_rows = []

        for i in range(len(df)):

            if any(header_mask[i]):
                header_rows.append(i)
            else:
                break

        if len(header_rows) == 0:

            df.columns = df.iloc[0]

            return df.iloc[1:].reset_index(drop=True)

        columns = []

        for col in range(df.shape[1]):

            pieces = []

            for row in header_rows:

                value = str(df.iat[row, col]).strip()

                if value != "":
                    pieces.append(value)

            columns.append(" - ".join(pieces))

        body = df.iloc[len(header_rows):].reset_index(drop=True)

        body.columns = columns

        return body

    # --------------------------------------------------
    # Page -> text index
    # --------------------------------------------------

    def build_text_index(self, text_blocks):

        page_index = defaultdict(list)

        for block in text_blocks:

            page = block.prov[0].page_no

            page_index[page].append(block)

        return page_index

    # --------------------------------------------------
    # Collect nearby context
    # (Saved only as metadata)
    # --------------------------------------------------

    def get_table_context(
        self,
        table,
        text_index,
        max_chars=700,
    ):

        page = table.prov[0].page_no

        if page not in text_index:
            return ""

        table_top = table.prov[0].bbox.t

        candidates = []

        for block in text_index[page]:

            label = str(block.label).lower()

            if "page_header" in label:
                continue

            if "page_footer" in label:
                continue

            text = block.text.strip()

            if len(text) == 0:
                continue

            if text.lower() == "table of contents":
                continue

            block_bottom = block.prov[0].bbox.b

            if block_bottom < table_top:

                candidates.append(block)

        candidates.sort(
            key=lambda x: x.prov[0].bbox.b,
            reverse=True,
        )

        context = []

        total = 0

        for block in candidates:

            text = block.text.strip()

            context.insert(0, text)

            total += len(text)

            if total >= max_chars:
                break

        return "\n\n".join(context)

    # --------------------------------------------------
    # Guess table title from nearby context
    # --------------------------------------------------

    def extract_title(self, context):

        if context == "":
            return ""

        lines = context.split("\n")

        for line in reversed(lines):

            line = line.strip()

            if len(line) < 4:
                continue

            if len(line) > 120:
                continue

            return line

        return ""

    # --------------------------------------------------
    # Convert dataframe to LangChain Document
    # --------------------------------------------------

    def dataframe_to_document(
        self,
        df,
        page,
        table_id,
        title="",
        context="",
    ):

        lines = []

        if title:

            lines.append(f"Table Title: {title}")
            lines.append("")

        lines.append("Columns:")
        lines.append("")

        for col in df.columns:

            lines.append(str(col))

        lines.append("")
        lines.append("Rows:")

        for _, row in df.iterrows():

            values = [str(v) for v in row.values]

            lines.append(" | ".join(values))

        text = "\n".join(lines)

        return Document(

            page_content=text,

            metadata={

                "page": page,
                "table": table_id,
                "title": title,
                "context": context,
                "type": "table",

            },

        )

    # --------------------------------------------------
    # Main pipeline
    # --------------------------------------------------

    def process(
        self,
        tables,
        text_blocks,
    ):

        text_index = self.build_text_index(text_blocks)

        documents = []

        for idx, table in enumerate(tables):

            df, mask = self.table_to_dataframe(table)

            df = self.merge_headers(df, mask)

            page = table.prov[0].page_no

            context = self.get_table_context(
                table,
                text_index,
            )

            title = self.extract_title(context)

            doc = self.dataframe_to_document(
                df,
                page,
                idx,
                title,
                context,
            )

            documents.append(doc)

        return documents