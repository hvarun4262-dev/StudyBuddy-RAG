import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

class RAGSearch:
    def __init__(self, vector_store, embedding_pipeline, model_name="gemini-2.0-flash"):
        self.vector_store = vector_store
        self.embedding_pipeline = embedding_pipeline
        
        # Initialize Gemini via LangChain
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.2,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # System instructions tailored for an AI Study Buddy
        self.prompt_template = PromptTemplate(
            input_variables=["context", "query"],
            template=(
                "You are an expert AI Research Mentor and Technical Study Buddy. "
                "Use the provided context extracted from technical AI, Machine Learning, "
                "and Deep Learning literature to explain the concept clearly, "
                "step-by-step, and accurately.\n\n"
                "If the answer is not present in the provided context, state that the topic "
                "is not covered in the current study materials.\n\n"
                "Context:\n{context}\n\n"
                "Student Question: {query}\n\n"
                "Technical Explanation:"
            )
        )

    def search_and_summarize(self, query: str, top_k=4):
        query_vector = self.embedding_pipeline.embed_text([query])[0]
        retrieved_chunks = self.vector_store.similarity_search(query_vector, top_k=top_k)
        
        if not retrieved_chunks:
            return "No relevant context found in study materials."

        context_str = "\n\n---\n\n".join([chunk.page_content for chunk in retrieved_chunks])
        
        formatted_prompt = self.prompt_template.format(context=context_str, query=query)
        response = self.llm.invoke(formatted_prompt)
        
        return response.content