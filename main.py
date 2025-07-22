import asyncio
import os
import time
from typing import List
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.workflow import Context
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler


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
        """Load documents from the specified directory with progress feedback."""
        print(f"🔍 Scanning directory: {self.data_directory}")
        
        if not os.path.exists(self.data_directory):
            raise FileNotFoundError(f"❌ Data directory '{self.data_directory}' not found. "
                                  f"Please create it and add your .md documents.")
        
        # Count files first for progress tracking
        file_count = 0
        for root, dirs, files in os.walk(self.data_directory):
            for file in files:
                if file.endswith(('.md', '.txt')):
                    file_count += 1
        
        print(f"📁 Found {file_count} markdown/text files")
        
        if file_count == 0:
            raise ValueError(f"❌ No .md or .txt files found in '{self.data_directory}'. "
                           f"Please add at least 10 .md documents.")
        
        if file_count < 10:
            print(f"⚠️  Warning: Only {file_count} documents found. For best results, consider adding more documents.")
        
        # Load documents from directory
        print("📄 Loading document contents...")
        reader = SimpleDirectoryReader(
            input_dir=self.data_directory,
            recursive=True,
            required_exts=[".md", ".txt"]
        )
        
        documents = reader.load_data()
        
        # Display document statistics
        total_chars = sum(len(doc.text) for doc in documents)
        avg_chars = total_chars // len(documents) if documents else 0
        
        print(f"✅ Successfully loaded {len(documents)} documents")
        print(f"📊 Total characters: {total_chars:,}")
        print(f"📊 Average document size: {avg_chars:,} characters")
        
        # Show document names for verification
        print("\n📋 Document list:")
        for i, doc in enumerate(documents[:10], 1):  # Show first 10
            filename = doc.metadata.get('file_name', 'Unknown')
            size = len(doc.text)
            print(f"   {i:2d}. {filename} ({size:,} chars)")
        
        if len(documents) > 10:
            print(f"   ... and {len(documents) - 10} more documents")
        
        return documents
    
    def create_index(self):
        """Create vector store index from documents with detailed progress feedback."""
        print("\n🚀 Starting document indexing process...")
        start_time = time.time()
        
        # Load documents with feedback
        documents = self.load_documents()
        load_time = time.time()
        
        print(f"\n🧠 Initializing embedding model: {Settings.embed_model.model_name}")
        print("   This may take a moment on first run (downloading model)...")
        
        # Test embedding to trigger model download
        try:
            test_embedding = Settings.embed_model.get_text_embedding("test")
            print(f"✅ Embedding model ready (vector dimension: {len(test_embedding)})")
        except Exception as e:
            print(f"⚠️  Embedding model initialization warning: {e}")
        
        embedding_init_time = time.time()
        
        print(f"\n🔄 Creating vector embeddings for {len(documents)} documents...")
        print("   This process will:")
        print("   1. Split documents into chunks")
        print("   2. Generate embeddings for each chunk")
        print("   3. Build searchable index")
        print("   Please wait, this may take several minutes...")
        
        # Add progress callback
        progress_callback = self._create_progress_callback(len(documents))
        
        # Create vector store index with progress tracking
        try:
            self.index = VectorStoreIndex.from_documents(
                documents,
                embed_model=Settings.embed_model,
                show_progress=True,
                callback_manager=CallbackManager([progress_callback])
            )
            
            indexing_time = time.time()
            
            print(f"\n🎯 Creating optimized query engine...")
            # Create query engine from the index
            self.query_engine = self.index.as_query_engine(
                llm=Settings.llm,
                similarity_top_k=5,
                response_mode="compact",
            )
            
            query_engine_time = time.time()
            
            # Display timing summary
            print(f"\n✅ Vector index created successfully!")
            print(f"⏱️  Timing Summary:")
            print(f"   📄 Document loading: {load_time - start_time:.2f}s")
            print(f"   🧠 Embedding setup: {embedding_init_time - load_time:.2f}s") 
            print(f"   🔄 Vector indexing: {indexing_time - embedding_init_time:.2f}s")
            print(f"   🎯 Query engine setup: {query_engine_time - indexing_time:.2f}s")
            print(f"   🏁 Total time: {query_engine_time - start_time:.2f}s")
            
            # Test the index
            print(f"\n🧪 Testing index with sample query...")
            try:
                test_response = self.query_engine.query("What topics are covered in these documents?")
                print(f"✅ Index test successful! Response length: {len(str(test_response))} chars")
            except Exception as e:
                print(f"⚠️  Index test warning: {e}")
                
        except Exception as e:
            print(f"\n❌ Error during indexing: {e}")
            print("💡 Troubleshooting tips:")
            print("   - Check if Ollama is running: ollama list")
            print("   - Try a smaller embedding model")
            print("   - Reduce document size or count")
            raise
    
    def _create_progress_callback(self, total_docs):
        """Create a callback for tracking indexing progress."""
        
        class ProgressCallback(LlamaDebugHandler):
            def __init__(self):
                super().__init__()
                self.chunk_count = 0
                self.embedding_count = 0
                self.last_update = time.time()
                
            def on_event_start(self, event_type, payload=None, event_id=None, **kwargs):
                current_time = time.time()
                
                if event_type.value == "embedding":
                    self.embedding_count += 1
                    # Update every 10 embeddings or every 5 seconds
                    if self.embedding_count % 10 == 0 or current_time - self.last_update > 5:
                        print(f"   🔄 Processing embeddings... ({self.embedding_count} completed)")
                        self.last_update = current_time
                
                elif event_type.value == "chunking":
                    self.chunk_count += 1
                    if self.chunk_count % 20 == 0:
                        print(f"   ✂️  Chunking documents... ({self.chunk_count} chunks created)")
        
        return ProgressCallback()
    
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
                print(f"🔍 Searching knowledge base for: '{query[:50]}{'...' if len(query) > 50 else ''}'")
                response = await self.query_engine.aquery(query)
                print(f"✅ Found relevant information ({len(str(response))} chars)")
                return str(response)
            except Exception as e:
                print(f"❌ Search error: {str(e)}")
                return f"Error searching documents: {str(e)}"
        
        return [multiply, add, divide, search_documents]
    
    def create_agent(self):
        """Create the agent workflow with tools."""
        if not self.query_engine:
            raise ValueError("❌ Index not created. Call create_index() first.")
        
        print(f"\n🤖 Setting up AI agent...")
        print(f"   🧠 LLM Model: {Settings.llm.model}")
        print(f"   🔧 Available tools: Document search, Calculator functions")
        
        tools = self._define_tools()
        
        # Create agent workflow with all tools
        self.agent = AgentWorkflow.from_tools_or_functions(
            tools,
            llm=Settings.llm,
            system_prompt="""You are a helpful AI assistant with access to a knowledge base of documents.
            
            You can:
            1. Answer questions about the content in the documents using the search_documents tool
            2. Perform basic mathematical calculations (multiply, add, divide)
            3. Maintain conversation context and remember previous interactions
            
            When answering questions about the documents:
            - Always search the documents first using the search_documents tool
            - Provide detailed, accurate answers based on the retrieved information
            - If information is not found in the documents, clearly state that
            - Cite or reference the source material when possible
            
            Be helpful, accurate, and conversational.""",
        )
        
        print("✅ Agent created successfully!")
        print("🎉 RAG system is now ready to answer questions!")
    
    async def initialize(self):
        """Initialize the complete RAG system with progress feedback."""
        print("=" * 60)
        print("🚀 LLAMAINDEX RAG SYSTEM INITIALIZATION")
        print("=" * 60)
        
        overall_start = time.time()
        
        try:
            # Step 1: Create index
            self.create_index()
            
            # Step 2: Create agent
            self.create_agent()
            
            overall_time = time.time() - overall_start
            
            print(f"\n" + "=" * 60)
            print("🎉 INITIALIZATION COMPLETE!")
            print(f"⏱️  Total setup time: {overall_time:.2f} seconds")
            print("💬 You can now start asking questions!")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ INITIALIZATION FAILED: {e}")
            print("\n🔧 Troubleshooting checklist:")
            print("   1. ✓ Ollama is installed and running")
            print("   2. ✓ Model is downloaded: ollama pull llama3.1")
            print("   3. ✓ Data directory exists with .md files")
            print("   4. ✓ All Python packages are installed")
            raise
    
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
        response2 = await assistant.chat("My name is Alice", context=ctx)
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