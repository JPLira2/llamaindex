import asyncio
import os
from typing import List
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.workflow import Context


class DocumentRAGAssistant:
    """
    A complete RAG (Retrieval-Augmented Generation) assistant using LlamaIndex
    with open-source models for commercial use.
    """
    
    def __init__(self, 
                 data_directory: str = "data",
                 llm_model: str = "llama3.1",
                 embedding_model: str = "BAAI/bge-base-en-v1.5",
                 context_window: int = 8000,
                 request_timeout: float = 360.0):
        """
        Initialize the RAG assistant.
        
        Args:
            data_directory: Directory containing .md documents
            llm_model: Ollama model name (e.g., "llama3.1", "mistral", "codellama")
            embedding_model: HuggingFace embedding model
            context_window: Context window size for the LLM
            request_timeout: Request timeout in seconds
        """
        self.data_directory = data_directory
        
        # Configure global settings for LlamaIndex
        Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        Settings.llm = Ollama(
            model=llm_model,
            request_timeout=request_timeout,
            context_window=context_window,
        )
        
        # Initialize components
        self.index = None
        self.query_engine = None
        self.agent = None
        
    def load_documents(self) -> List:
        """Load documents from the specified directory."""
        if not os.path.exists(self.data_directory):
            raise FileNotFoundError(f"Data directory '{self.data_directory}' not found. "
                                  f"Please create it and add your .md documents.")
        
        # Load documents from directory (supports .md, .txt, and other formats)
        reader = SimpleDirectoryReader(
            input_dir=self.data_directory,
            recursive=True,  # Include subdirectories
            required_exts=[".md", ".txt"]  # Only load markdown and text files
        )
        
        documents = reader.load_data()
        
        if not documents:
            raise ValueError(f"No documents found in '{self.data_directory}'. "
                           f"Please add at least 10 .md documents.")
        
        print(f"Loaded {len(documents)} documents from {self.data_directory}")
        return documents
    
    def create_index(self):
        """Create vector store index from documents."""
        documents = self.load_documents()
        
        # Create vector store index with the loaded documents
        self.index = VectorStoreIndex.from_documents(
            documents,
            embed_model=Settings.embed_model,
        )
        
        # Create query engine from the index
        self.query_engine = self.index.as_query_engine(
            llm=Settings.llm,
            similarity_top_k=5,  # Return top 5 most relevant chunks
        )
        
        print("Vector index created successfully!")
    
    def _define_tools(self):
        """Define tools for the agent."""
        
        def multiply(a: float, b: float) -> float:
            """Useful for multiplying two numbers."""
            return a * b
        
        def add(a: float, b: float) -> float:
            """Useful for adding two numbers."""
            return a + b
        
        def divide(a: float, b: float) -> float:
            """Useful for dividing two numbers."""
            if b == 0:
                return "Error: Division by zero"
            return a / b
        
        async def search_documents(query: str) -> str:
            """
            Search through the loaded documents to answer questions.
            This tool has access to all the markdown documents in your knowledge base.
            """
            if not self.query_engine:
                return "Error: Documents not loaded. Please initialize the system first."
            
            try:
                response = await self.query_engine.aquery(query)
                return str(response)
            except Exception as e:
                return f"Error searching documents: {str(e)}"
        
        return [multiply, add, divide, search_documents]
    
    def create_agent(self):
        """Create the agent workflow with tools."""
        if not self.query_engine:
            raise ValueError("Index not created. Call create_index() first.")
        
        tools = self._define_tools()
        
        # Create agent workflow with all tools
        self.agent = AgentWorkflow.from_tools_or_functions(
            tools,
            llm=Settings.llm,
            system_prompt="""You are a helpful AI assistant with access to a knowledge base of documents.
            
            You can:
            1. Answer questions about the content in the documents using the search_documents tool
            2. Maintain conversation context and remember previous interactions
            
            When answering questions about the documents:
            - Always search the documents first using the search_documents tool
            - Provide detailed, accurate answers based on the retrieved information
            - If information is not found in the documents, clearly state that
            - Cite or reference the source material when possible
            
            Be helpful, accurate, and conversational.""",
        )
        
        print("Agent created successfully!")
    
    async def initialize(self):
        """Initialize the complete RAG system."""
        print("Initializing RAG system...")
        self.create_index()
        self.create_agent()
        print("RAG system ready!")
    
    async def chat(self, message: str, context: Context = None) -> str:
        """
        Chat with the assistant.
        
        Args:
            message: User's message/question
            context: Optional context for maintaining conversation history
            
        Returns:
            Assistant's response
        """
        if not self.agent:
            raise ValueError("Agent not initialized. Call initialize() first.")
        
        try:
            if context:
                response = await self.agent.run(message, ctx=context)
            else:
                response = await self.agent.run(message)
            return str(response)
        except Exception as e:
            return f"Error processing request: {str(e)}"


async def main():
    """Example usage of the DocumentRAGAssistant."""
    
    # Initialize the assistant
    assistant = DocumentRAGAssistant(
        data_directory="data",
        llm_model="llama3.1",  # You can also try "mistral", "codellama", etc.
        embedding_model="BAAI/bge-base-en-v1.5"
    )
    
    try:
        # Initialize the RAG system
        await assistant.initialize()
        
        # Create context for conversation history
        ctx = Context(assistant.agent)
        
        # Example conversations
        print("=== RAG Assistant Demo ===\n")
        
        # Test document search
        print("1. Searching documents...")
        response1 = await assistant.chat(
            "What are the main topics covered in the documents?", 
            context=ctx
        )
        print(f"Assistant: {response1}\n")
        
        # Test with context/memory
        print("2. Testing conversation memory...")
        response2 = await assistant.chat("My name is JP", context=ctx)
        print(f"Assistant: {response2}\n")
        
        response3 = await assistant.chat("What's my name?", context=ctx)
        print(f"Assistant: {response3}\n")
        
        # Test calculation + document search
        print("3. Testing mixed capabilities...")
        response4 = await assistant.chat(
            "Can you search for information about implementation details and also calculate 125 * 8?",
            context=ctx
        )
        print(f"Assistant: {response4}\n")
        
        # Interactive mode
        print("4. Interactive mode (type 'quit' to exit):")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
            
            response = await assistant.chat(user_input, context=ctx)
            print(f"Assistant: {response}")
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Created a 'data' directory")
        print("2. Added at least 10 .md documents to the data directory")
        print("3. Installed required packages: pip install llama-index llama-index-llms-ollama llama-index-embeddings-huggingface")
        print("4. Have Ollama running with llama3.1 model: ollama pull llama3.1")


if __name__ == "__main__":
    asyncio.run(main())