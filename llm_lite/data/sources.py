from llm_lite.config.models import InlineTextDatasetConfiguration
from llm_lite.data.document import Document


def load_inline_documents(dataset_configuration: InlineTextDatasetConfiguration) -> list[Document]:
    return [
        Document(
            document_id=f"inline-{document_index:06d}",
            text=document_text,
            metadata={"source": "inline_text", "index": document_index},
        )
        for document_index, document_text in enumerate(dataset_configuration.documents)
    ]
