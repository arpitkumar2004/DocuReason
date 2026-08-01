import fitz
import pdfplumber
import pickle

from pathlib import Path


class EnterprisePDFParser:

    def __init__(self, pdf_path):

        self.pdf_path = Path(pdf_path)

    def extract_text(self):

        document = fitz.open(self.pdf_path)

        pages = []

        for page_no, page in enumerate(document):

            text = page.get_text("text").strip()

            pages.append({

                "page": page_no + 1,

                "source": self.pdf_path.name,

                "text": text

            })

        document.close()

        return pages

    def extract_tables(self):

        pages = []

        with pdfplumber.open(self.pdf_path) as pdf:

            for page_no, page in enumerate(pdf.pages):

                raw_tables = page.extract_tables()

                tables = []

                for table in raw_tables:

                    if table is None:

                        continue

                    tables.append({

                        "page": page_no + 1,

                        "table": table

                    })

                pages.append({

                    "page": page_no + 1,

                    "tables": tables

                })

        return pages

    def save(self,
             obj,
             path):

        with open(path, "wb") as f:

            pickle.dump(obj, f)