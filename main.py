import asyncio
import os
import time
import json
import hashlib
import re
from typing import List, Dict, Set, Any
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.workflow import Context
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle


class DocumentRAGAssistant:
    """
    A complete RAG (Retrieval-Augmented Generation) assistant using LlamaIndex
    with open-source models for commercial use.
    """
    
    def __init__(self, 
                 data_directory: str = "data",
                 index_directory: str = "index_storage",
                 llm_model: str = "llama3.1",
                 embedding_model: str = "BAAI/bge-base-en-v1.5",
                 context_window: int = 8000,
                 request_timeout: float = 360.0):
        """
        Initialize the RAG assistant with persistent indexing.
        
        Args:
            data_directory: Directory containing .md documents
            index_directory: Directory to store persistent index
            llm_model: Ollama model name (e.g., "llama3.1", "mistral", "codellama")
            embedding_model: HuggingFace embedding model
            context_window: Context window size for the LLM
            request_timeout: Request timeout in seconds
        """
        self.data_directory = data_directory
        self.index_directory = index_directory
        self.metadata_file = os.path.join(index_directory, "document_metadata.json")
        
        # Create index directory if it doesn't exist
        Path(index_directory).mkdir(exist_ok=True)
        
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
        self.document_metadata = {}
        
        # Enhanced multi-context components
        self.enhanced_retrievers = {}
        self.entity_tracker = {}
        self.multi_context_enabled = False
        
    def _get_file_hash(self, filepath: str) -> str:
        """Generate hash of file content for change detection."""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def _get_current_document_state(self) -> Dict[str, Dict]:
        """Get current state of all documents in the data directory."""
        current_state = {}
        
        for root, dirs, files in os.walk(self.data_directory):
            for file in files:
                if file.endswith(('.md', '.txt')):
                    filepath = os.path.join(root, file)
                    relative_path = os.path.relpath(filepath, self.data_directory)
                    
                    try:
                        stat = os.stat(filepath)
                        current_state[relative_path] = {
                            'size': stat.st_size,
                            'modified': stat.st_mtime,
                            'hash': self._get_file_hash(filepath)
                        }
                    except Exception as e:
                        print(f"⚠️  Warning: Could not read {relative_path}: {e}")
        
        return current_state
    
    def _load_document_metadata(self) -> Dict[str, Dict]:
        """Load previously stored document metadata."""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Could not load metadata: {e}")
        return {}
    
    def _save_document_metadata(self, metadata: Dict[str, Dict]):
        """Save document metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"⚠️  Warning: Could not save metadata: {e}")
    
    def _detect_changes(self) -> Dict[str, Set[str]]:
        """Detect changes in documents since last indexing."""
        print("🔍 Checking for document changes...")
        
        current_state = self._get_current_document_state()
        previous_state = self._load_document_metadata()
        
        changes = {
            'added': set(),
            'modified': set(),
            'removed': set()
        }
        
        # Find added and modified files
        for filepath, current_info in current_state.items():
            if filepath not in previous_state:
                changes['added'].add(filepath)
            elif (current_info['hash'] != previous_state[filepath].get('hash', '') or
                  current_info['modified'] != previous_state[filepath].get('modified', 0)):
                changes['modified'].add(filepath)
        
        # Find removed files
        for filepath in previous_state:
            if filepath not in current_state:
                changes['removed'].add(filepath)
        
        # Update metadata
        self.document_metadata = current_state
        
        return changes
    
    def _check_index_exists(self) -> bool:
        """Check if a valid index exists on disk."""
        storage_path = os.path.join(self.index_directory, "docstore.json")
        return os.path.exists(storage_path)
    
    def _report_changes(self, changes: Dict[str, Set[str]]) -> bool:
        """Report detected changes and return True if any changes found."""
        total_changes = sum(len(change_set) for change_set in changes.values())
        
        if total_changes == 0:
            print("✅ No changes detected - using existing index")
            return False
        
        print(f"📊 Detected {total_changes} document changes:")
        
        if changes['added']:
            print(f"   ➕ Added ({len(changes['added'])}): {', '.join(list(changes['added'])[:3])}{'...' if len(changes['added']) > 3 else ''}")
        
        if changes['modified']:
            print(f"   📝 Modified ({len(changes['modified'])}): {', '.join(list(changes['modified'])[:3])}{'...' if len(changes['modified']) > 3 else ''}")
        
        if changes['removed']:
            print(f"   ❌ Removed ({len(changes['removed'])}): {', '.join(list(changes['removed'])[:3])}{'...' if len(changes['removed']) > 3 else ''}")
        
        return True
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
    
    def create_index(self, force_rebuild: bool = False):
        """Create or load vector store index with intelligent caching and incremental updates."""
        print("\n🚀 Starting document indexing process...")
        start_time = time.time()
        
        # Check for existing index and document changes
        index_exists = self._check_index_exists()
        changes = self._detect_changes() if not force_rebuild else {'added': {'force_rebuild'}, 'modified': set(), 'removed': set()}
        needs_rebuild = not index_exists or self._report_changes(changes) or force_rebuild
        
        if not needs_rebuild and index_exists:
            print("⚡ Loading existing index from disk...")
            try:
                storage_context = StorageContext.from_defaults(persist_dir=self.index_directory)
                self.index = load_index_from_storage(storage_context)
                
                print(f"🎯 Creating optimized query engine...")
                self.query_engine = self.index.as_query_engine(
                    llm=Settings.llm,
                    similarity_top_k=5,
                    response_mode="compact",
                )
                
                load_time = time.time() - start_time
                print(f"✅ Index loaded successfully in {load_time:.2f}s!")
                
                # Test the loaded index
                self._test_index()
                
                # Setup enhanced retrieval if requested
                if hasattr(self, '_setup_enhanced_after_load') and self._setup_enhanced_after_load:
                    self._create_enhanced_query_engines()
                    
                return
                
            except Exception as e:
                print(f"⚠️  Failed to load existing index: {e}")
                print("🔄 Falling back to full rebuild...")
                needs_rebuild = True
        
        if needs_rebuild:
            if index_exists and not force_rebuild:
                print("🔄 Performing incremental update...")
            else:
                print("🆕 Creating new index from scratch...")
            
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
            print("   4. Save index to disk for future use")
            print("   Please wait, this may take several minutes...")
            
            # Add progress callback
            progress_callback = self._create_progress_callback(len(documents))
            
            # Create vector store index with clean progress tracking
            try:
                self.index = VectorStoreIndex.from_documents(
                    documents,
                    embed_model=Settings.embed_model,
                    show_progress=True,  # Use LlamaIndex's built-in progress bar
                    # No callback manager to avoid debug traces
                )
                
                indexing_time = time.time()
                
                # Persist the index to disk
                print(f"\n💾 Saving index to disk...")
                self.index.storage_context.persist(persist_dir=self.index_directory)
                
                # Save document metadata for change detection
                self._save_document_metadata(self.document_metadata)
                
                print(f"🎯 Creating optimized query engine...")
                self.query_engine = self.index.as_query_engine(
                    llm=Settings.llm,
                    similarity_top_k=5,
                    response_mode="compact",
                )
                
                query_engine_time = time.time()
                
                # Display timing summary
                print(f"\n✅ Vector index created and saved successfully!")
                print(f"⏱️  Timing Summary:")
                print(f"   📄 Document loading: {load_time - start_time:.2f}s")
                print(f"   🧠 Embedding setup: {embedding_init_time - load_time:.2f}s") 
                print(f"   🔄 Vector indexing: {indexing_time - embedding_init_time:.2f}s")
                print(f"   💾 Saving to disk: {query_engine_time - indexing_time:.2f}s")
                print(f"   🏁 Total time: {query_engine_time - start_time:.2f}s")
                print(f"   📁 Index saved to: {self.index_directory}")
                
                # Test the index
                self._test_index()
                
                # Setup enhanced retrieval if requested
                if hasattr(self, '_setup_enhanced_after_index') and self._setup_enhanced_after_index:
                    self._create_enhanced_query_engines()
                    
            except Exception as e:
                print(f"\n❌ Error during indexing: {e}")
                print("💡 Troubleshooting tips:")
                print("   - Check if Ollama is running: ollama list")
                print("   - Try a smaller embedding model")
                print("   - Reduce document size or count")
                print("   - Clear index directory and retry")
                raise
    
    def _test_index(self):
        """Test the index with a sample query."""
        print(f"\n🧪 Testing index with sample query...")
        try:
            test_response = self.query_engine.query("What topics are covered in these documents?")
            print(f"✅ Index test successful! Response length: {len(str(test_response))} chars")
        except Exception as e:
            print(f"⚠️  Index test warning: {e}")
    
    def clear_index(self):
        """Clear the persistent index and force a full rebuild on next initialization."""
        print("🗑️  Clearing persistent index...")
        try:
            import shutil
            if os.path.exists(self.index_directory):
                shutil.rmtree(self.index_directory)
                Path(self.index_directory).mkdir(exist_ok=True)
                print("✅ Index cleared successfully")
            else:
                print("ℹ️  No index found to clear")
        except Exception as e:
            print(f"❌ Error clearing index: {e}")
    
    def force_reindex(self):
        """Force a complete reindex of all documents."""
        print("🔄 Forcing complete reindex...")
        self.clear_index()
        self.create_index(force_rebuild=True)
    
    # ============================================================================
    # ENHANCED MULTI-CONTEXT RETRIEVAL CAPABILITIES
    # ============================================================================
    
    async def enable_multi_context(self):
        """Enable advanced multi-context retrieval capabilities."""
        if not self.index:
            raise ValueError("❌ Index not created. Call initialize() first.")
        
        print("\n🚀 Setting up enhanced multi-context retrieval...")
        self._create_enhanced_query_engines()
        self.multi_context_enabled = True
        print("✅ Multi-context retrieval enabled!")
        print("🎯 Now supports: Entity tracking, Cross-document synthesis, Relationship queries")
    
    def _create_enhanced_query_engines(self):
        """Create multiple specialized query engines for different types of questions."""
        
        print("   🔧 Creating specialized retrievers...")
        
        # 1. High-recall retriever for broad questions
        self.enhanced_retrievers['broad'] = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=15,  # More chunks for synthesis
        )
        
        # 2. Precise retriever for specific questions  
        self.enhanced_retrievers['precise'] = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=5,
        )
        
        # 3. Multi-document synthesizer
        synthesis_response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.TREE_SUMMARIZE,
            use_async=True,
        )
        
        self.enhanced_retrievers['synthesis_engine'] = RetrieverQueryEngine(
            retriever=self.enhanced_retrievers['broad'],
            response_synthesizer=synthesis_response_synthesizer,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.6),
            ]
        )
        
        # 4. Entity-focused retriever
        entity_response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT,
            use_async=True,
        )
        
        self.enhanced_retrievers['entity_engine'] = RetrieverQueryEngine(
            retriever=self.enhanced_retrievers['precise'],
            response_synthesizer=entity_response_synthesizer,
        )
        
        print("   ✅ Enhanced retrievers created")
    
    async def _analyze_query_type(self, query: str) -> str:
        """Analyze what type of query this is to choose the best retrieval strategy."""
        
        query_lower = query.lower()
        
        # Entity-specific queries
        entity_indicators = [
            "quien es", "who is", "what is", "donde esta", "when did",
            "quién es", "qué es", "dónde está", "cuándo", "como es"
        ]
        if any(indicator in query_lower for indicator in entity_indicators):
            return "entity"
            
        # Broad synthesis queries
        synthesis_indicators = [
            "summarize", "overview", "main topics", "what are the", "todos los", 
            "resumen", "principales", "temas", "general", "conjunto", "todas las",
            "cuales son", "cuáles son", "que contiene", "qué contiene"
        ]
        if any(indicator in query_lower for indicator in synthesis_indicators):
            return "synthesis"
            
        # Relationship queries
        relationship_indicators = [
            "relationship", "connection", "related to", "compared to", "relacion",
            "relación", "conexión", "relacionado", "comparado", "entre",
            "vinculo", "vínculo", "asociado", "conectado"
        ]
        if any(indicator in query_lower for indicator in relationship_indicators):
            return "relationship"
            
        # Multi-document queries
        multi_doc_indicators = [
            "across documents", "in all", "en todos", "através", "multiple",
            "varios", "diferentes", "conjunto", "total", "completo"
        ]
        if any(indicator in query_lower for indicator in multi_doc_indicators):
            return "synthesis"
            
        # Default to precise
        return "precise"
    
    async def _multi_context_search(self, query: str) -> Dict[str, Any]:
        """Perform multi-context search using different strategies and combine results."""
        
        if not self.multi_context_enabled:
            # Fallback to regular search
            response = await self.query_engine.aquery(query)
            return {
                "query_type": "standard",
                "primary_response": str(response),
                "supporting_evidence": [],
                "cross_references": [],
                "method_used": "standard_retrieval"
            }
        
        query_type = await self._analyze_query_type(query)
        print(f"🧠 Query type detected: {query_type}")
        
        results = {
            "query_type": query_type,
            "primary_response": "",
            "supporting_evidence": [],
            "cross_references": [],
            "method_used": f"enhanced_{query_type}"
        }
        
        try:
            if query_type == "entity":
                print("🔍 Using entity-focused retrieval...")
                response = await self.enhanced_retrievers['entity_engine'].aquery(query)
                results["primary_response"] = str(response)
                
                # Try to find related entities
                entities = await self._extract_entities_from_response(str(response))
                if entities:
                    results["cross_references"] = await self._find_entity_cross_references(entities)
                
            elif query_type == "synthesis":
                print("📊 Using synthesis retrieval across multiple contexts...")
                response = await self.enhanced_retrievers['synthesis_engine'].aquery(query)
                results["primary_response"] = str(response)
                
                # Get supporting evidence from multiple documents
                supporting_nodes = await self._get_supporting_evidence(query)
                results["supporting_evidence"] = [
                    {
                        "text": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                        "score": getattr(node, 'score', 0.0),
                        "source": node.metadata.get('file_name', 'Unknown') if hasattr(node, 'metadata') else 'Unknown'
                    }
                    for node in supporting_nodes[:3]
                ]
                
            elif query_type == "relationship":
                print("🕸️ Using relationship-aware retrieval...")
                # Use both engines and cross-reference
                precise_response = await self.enhanced_retrievers['entity_engine'].aquery(query)
                broad_response = await self.enhanced_retrievers['synthesis_engine'].aquery(query)
                
                # Combine and synthesize
                results["primary_response"] = await self._synthesize_relationships(
                    query, str(precise_response), str(broad_response)
                )
                
            else:  # precise
                print("🎯 Using precise retrieval...")
                response = await self.enhanced_retrievers['entity_engine'].aquery(query)
                results["primary_response"] = str(response)
        
        except Exception as e:
            print(f"❌ Error in multi-context search: {e}")
            # Fallback to basic search
            fallback_response = await self.query_engine.aquery(query)
            results["primary_response"] = str(fallback_response)
            results["method_used"] = "fallback_to_standard"
        
        return results
    
    async def _extract_entities_from_response(self, response: str) -> List[str]:
        """Extract named entities from a response (simplified version)."""
        
        # Simple patterns for names and entities
        name_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # FirstName LastName
            r'\b[A-Z][a-záéíóúñ]+ [A-Z][a-záéíóúñ]+\b',  # Spanish names
            r'\b[A-Z][a-z]*[A-Z][a-z]*\b',  # CamelCase (company names, etc.)
        ]
        
        entities = []
        for pattern in name_patterns:
            matches = re.findall(pattern, response)
            entities.extend(matches)
        
        # Filter out common false positives
        false_positives = {
            'AI', 'API', 'URL', 'HTTP', 'JSON', 'XML', 'SQL', 'CSS', 'HTML',
            'PDF', 'DOC', 'TXT', 'MD', 'PNG', 'JPG', 'GIF'
        }
        
        filtered_entities = [e for e in entities if e not in false_positives and len(e) > 3]
        return list(set(filtered_entities))  # Remove duplicates
    
    async def _find_entity_cross_references(self, entities: List[str]) -> List[Dict]:
        """Find cross-references to entities across documents."""
        cross_refs = []
        
        for entity in entities[:5]:  # Limit to prevent too many searches
            try:
                # Search for each entity across all documents
                entity_query = f"{entity}"
                nodes = await self.enhanced_retrievers['broad'].aretrieve(QueryBundle(entity_query))
                
                if nodes and len(nodes) > 1:  # Only show if found in multiple places
                    cross_refs.append({
                        "entity": entity,
                        "references": len(nodes),
                        "preview": nodes[0].text[:100] + "..." if nodes else "",
                        "sources": list(set([
                            node.metadata.get('file_name', 'Unknown')[:20] 
                            for node in nodes[:3] 
                            if hasattr(node, 'metadata')
                        ]))
                    })
            except Exception as e:
                print(f"⚠️ Error finding cross-references for {entity}: {e}")
        
        return cross_refs
    
    async def _get_supporting_evidence(self, query: str) -> List[NodeWithScore]:
        """Get supporting evidence from multiple document sources."""
        try:
            nodes = await self.enhanced_retrievers['broad'].aretrieve(QueryBundle(query))
            return nodes[:5]  # Top 5 supporting pieces
        except Exception as e:
            print(f"⚠️ Error getting supporting evidence: {e}")
            return []
    
    async def _synthesize_relationships(self, query: str, precise_info: str, broad_info: str) -> str:
        """Synthesize relationship information from multiple contexts."""
        
        synthesis_prompt = f"""Based on the following information from multiple sources, provide a comprehensive answer that shows relationships and connections:

Specific Information: {precise_info}

Broader Context: {broad_info}

Original Question: {query}

Please provide a synthesized answer that:
1. Directly answers the question
2. Shows relationships between entities or concepts
3. Connects information from different sources
4. Highlights any important patterns or connections

Answer:"""
        
        try:
            # Use the base LLM to synthesize
            response = await Settings.llm.acomplete(synthesis_prompt)
            return str(response)
        except Exception as e:
            # Fallback to simpler combination
            return f"{precise_info}\n\nAdditional context: {broad_info}"
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
            Use this tool to find information about any topic mentioned in the documents.
            Now enhanced with multi-context capabilities for better cross-document reasoning.
            """
            if not self.query_engine:
                return "Error: Documents not loaded. Please initialize the system first."
            
            try:
                # Use enhanced search if available
                if self.multi_context_enabled:
                    print(f"🚀 Using enhanced multi-context search for: '{query[:50]}{'...' if len(query) > 50 else ''}'")
                    results = await self._multi_context_search(query)
                    
                    # Format the enhanced response
                    response = results["primary_response"]
                    
                    # Add cross-references if found
                    if results["cross_references"]:
                        response += "\n\n🔗 Related entities found across documents:"
                        for ref in results["cross_references"][:3]:  # Limit to top 3
                            sources_str = ", ".join(ref["sources"][:2])
                            response += f"\n• {ref['entity']}: {ref['references']} references in {sources_str}"
                    
                    # Add supporting evidence for synthesis queries
                    if results["supporting_evidence"] and results["query_type"] == "synthesis":
                        response += "\n\n📚 Supporting evidence:"
                        for evidence in results["supporting_evidence"][:2]:  # Top 2
                            response += f"\n• From {evidence['source']}: {evidence['text']}"
                    
                    print(f"✅ Enhanced search complete ({len(response)} chars, method: {results['method_used']})")
                    return response
                
                else:
                    # Standard search
                    print(f"🔍 Searching knowledge base for: '{query[:50]}{'...' if len(query) > 50 else ''}'")
                    response = await self.query_engine.aquery(query)
                    result = str(response)
                    print(f"✅ Found relevant information ({len(result)} chars)")
                    return result
                    
            except Exception as e:
                error_msg = f"Error searching documents: {str(e)}"
                print(f"❌ Search error: {str(e)}")
                return error_msg
        
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

            IMPORTANT INSTRUCTIONS:
            1. When users ask questions, ALWAYS use the search_documents tool first to find relevant information
            2. Base your answers ONLY on the information returned by the search_documents tool
            3. If the search_documents tool returns relevant information, use that information to answer the question directly
            4. DO NOT make up information or give generic responses when specific information is found
            5. If no relevant information is found, clearly state that the information is not available in the knowledge base
            
            RESPONSE GUIDELINES:
            - Always trust and use the results from search_documents
            - Quote or reference specific information found in the documents
            - Be direct and specific in your answers
            - If mathematical calculations are needed, use the appropriate tools (multiply, add, divide)
            
            Remember: Your primary job is to answer questions based on the content in the knowledge base. Always search first, then provide accurate answers based on what you find.""",
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
    
    async def chat(self, message: str, context: Context = None, debug: bool = False) -> str:
        """
        Chat with the assistant.
        
        Args:
            message: User's message/question
            context: Optional context for maintaining conversation history
            debug: If True, shows detailed debug information
            
        Returns:
            Assistant's response
        """
        if not self.agent:
            raise ValueError("Agent not initialized. Call initialize() first.")
        
        try:
            if debug:
                print(f"\n🐛 DEBUG: Processing message: '{message}'")
            
            if context:
                response = await self.agent.run(message, ctx=context)
            else:
                response = await self.agent.run(message)
            
            if debug:
                print(f"🐛 DEBUG: Agent response length: {len(str(response))} chars")
                print(f"🐛 DEBUG: Response preview: {str(response)[:200]}...")
            
            return str(response)
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            if debug:
                print(f"🐛 DEBUG ERROR: {error_msg}")
            return error_msg


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
        
        # Show index management options
        print("=" * 60)
        print("🔧 INDEX MANAGEMENT OPTIONS")
        print("=" * 60)
        print("The index is now persistent! You can:")
        print("- Add/remove/modify documents in the 'data' directory")
        print("- Restart the script - it will detect changes automatically")
        print("- Use assistant.force_reindex() to rebuild everything")
        print("- Use assistant.clear_index() to start fresh")
        print("=" * 60)
        
        # Test multi-context capabilities
        print("\n🚀 Testing Multi-Context Capabilities...")
        await assistant.enable_multi_context()
        
        print("\n5. Enhanced Multi-Context Examples:")
        
        # Test entity query
        print("   a) Entity query with cross-references:")
        response5 = await assistant.chat("¿Quién es Felipe Monge?", context=ctx)
        print(f"   Assistant: {response5}\n")
        
        # Test synthesis query
        print("   b) Document synthesis query:")
        response6 = await assistant.chat("¿Cuáles son los principales temas en todos los documentos?", context=ctx)
        print(f"   Assistant: {response6}\n")
        
        # Test relationship query
        print("   c) Relationship query:")
        response7 = await assistant.chat("¿Qué información relacionada con Felipe Monge aparece en diferentes documentos?", context=ctx)
        print(f"   Assistant: {response7}\n")
        
        # Interactive mode
        print("4. Interactive mode (type 'quit' to exit, 'reindex' to force rebuild, 'debug' to toggle debug mode, 'enhance' to toggle multi-context):")
        debug_mode = False
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
            elif user_input.lower() == 'reindex':
                print("🔄 Forcing reindex...")
                await assistant.force_reindex()
                await assistant.create_agent()  # Recreate agent with new index
                if assistant.multi_context_enabled:
                    await assistant.enable_multi_context()  # Re-enable if it was on
                print("✅ Reindex complete!")
                continue
            elif user_input.lower() == 'clear':
                assistant.clear_index()
                continue
            elif user_input.lower() == 'debug':
                debug_mode = not debug_mode
                print(f"🐛 Debug mode: {'ON' if debug_mode else 'OFF'}")
                continue
            elif user_input.lower() == 'enhance':
                if assistant.multi_context_enabled:
                    assistant.multi_context_enabled = False
                    print("📊 Multi-context mode: OFF (using standard retrieval)")
                else:
                    await assistant.enable_multi_context()
                    print("🚀 Multi-context mode: ON (using enhanced retrieval)")
                continue
            
            response = await assistant.chat(user_input, context=ctx, debug=debug_mode)
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