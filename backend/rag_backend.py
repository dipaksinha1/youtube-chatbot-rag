from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


class YouTubeRAGChatbot:
    def __init__(self):
        self.vector_store = None
        self.retriever = None

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2
        )

        self.conversation_history = []

        self.prompt = PromptTemplate(
            template="""You are a helpful assistant.
            You are provided with a transcript of a YouTube video.
            Answer ONLY from the provided transcript context.
            If the answer is not in the context, say "Please ask related to transcript content"
            If unclear, ask a clarification question.

            Conversation History:
            {conversation_history}

            Context:
            {context}

            Question:
            {question}""",
            input_variables=["context", "question", "conversation_history"]
        )

    def ingest_youtube_video(self, video_id: str):
        try:
            # Create API instance
            ytt_api = YouTubeTranscriptApi()
            
            # Fetch transcript (defaults to English)
            transcript_data = ytt_api.fetch(video_id)
            
        except Exception as e:
            raise ValueError(f"Transcript not available in English")

        # Convert transcript to text
        transcript_text = " ".join(chunk.text for chunk in transcript_data)

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120
        )
        documents = splitter.create_documents([transcript_text])

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large"
        )
        self.vector_store = FAISS.from_documents(documents, embeddings)

        # Access the underlying FAISS index
        faiss_index = self.vector_store.index
        num_vectors = faiss_index.ntotal

        # Iterate and display actual vector values
        for i in range(num_vectors):
            vector = faiss_index.reconstruct(i)
            print(f"Vector {i}: {vector}")  # Shows full numpy array
            print(f"Vector {i} shape: {vector.shape}")  # Shows dimensions
            print(f"Vector {i} first 5 values: {vector[:5]}")  # Shows first 5 elements
            print("---")

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )

    def chat(self, question: str) -> str:
        if not self.retriever:
            raise RuntimeError("Vector database not initialized. Call ingest_youtube_video() first.")

        if len(self.conversation_history) > 6:
            # Remove the oldest message
            self.conversation_history = self.conversation_history.pop(0)

        # Add user message to history
        self.conversation_history.append({"role":"user", "content":question})

        # Retrieve relevant docs
        docs = self.retriever.invoke(question)

        context = "\n\n".join(doc.page_content for doc in docs)

        # Format and invoke
        final_prompt = self.prompt.format(
            context=context,
            question=question,
            conversation_history=self.conversation_history
        )
        response = self.llm.invoke(final_prompt)
        
        #Add assistant message to history
        self.conversation_history.append({"role":"assistant", "content":response.content})
        return response.content