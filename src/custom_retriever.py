from collections import defaultdict
from typing import List

from langchain_core.documents import Document
from langchain.retrievers import MultiVectorRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun



class CustomMultiVectorRetriever(MultiVectorRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents. Return Format: [{"doc": Document, "sub_docs": List[Document]}]
        """

        results = self.vectorstore._similarity_search_with_relevance_scores(
            query, **self.search_kwargs
        )

        # Map doc_ids to list of sub-documents, adding scores to metadata
        id_to_doc = defaultdict(list)
        for doc, score in results:
            doc_id = doc.metadata.get("doc_id")
            if doc_id:
                doc.metadata["score"] = score
                id_to_doc[doc_id].append(doc)

        # Fetch documents corresponding to doc_ids, retaining sub_docs in metadata
        docs = []
        for _id, sub_docs in id_to_doc.items():
            docstore_docs = self.docstore.mget([_id])
            # print("docstore_docs[0].metadata.to_dict(): ", docstore_docs[0].metadata.to_dict())
            # print("\n\n docstore[0].metadata: ", docstore_docs[0].metadata)
            # # print("\n\n docstore_docs[0].metadata.sub_docs: ", docstore_docs[0].metadata.sub_docs)
            # print("sub_docs: ", sub_docs)
            # print("type of sub_docs", type(sub_docs))
            if docstore_docs:
                if doc := docstore_docs[0]:
                    doc_w_sub_docs = {"doc": doc, "sub_docs": sub_docs} # This is done as I am unable to edit the metadata of the doc directly
                    docs.append(doc_w_sub_docs)

        return docs