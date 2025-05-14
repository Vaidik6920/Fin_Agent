from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

class FinancialRAG:
    def __init__(self, urls):
        self.loader = WebBaseLoader(urls)
        self.embeddings = OpenAIEmbeddings()
        self.docs = self.loader.load()
        self.vectorstore = FAISS.from_documents(self.docs, self.embeddings)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            retriever=self.vectorstore.as_retriever()
        )

    def query(self, question: str) -> str:
        return self.qa_chain.run(question)
