"""
RAG (Retrieval Augmented Generation) Pipeline
=============================================

Combine LLMs with vector search for enhanced responses.
"""

import os
import json
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document in the vector store."""
    
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    doc_id: Optional[str] = None
    
    def __post_init__(self):
        if self.doc_id is None:
            # Generate ID from content hash
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'doc_id': self.doc_id,
            'content': self.content,
            'metadata': self.metadata,
            'embedding': self.embedding
        }


@dataclass
class SearchResult:
    """Search result from vector store."""
    
    document: Document
    score: float
    rank: int


class VectorStore:
    """Simple in-memory vector store."""
    
    def __init__(
        self,
        embedding_dim: int = 768,
        similarity_metric: str = "cosine",
        persist_path: Optional[str] = None
    ):
        """
        Initialize vector store.
        
        Args:
            embedding_dim: Dimension of embeddings
            similarity_metric: "cosine", "euclidean", or "dot"
            persist_path: Path to persist store
        """
        self.embedding_dim = embedding_dim
        self.similarity_metric = similarity_metric
        self.persist_path = persist_path
        
        self.documents: Dict[str, Document] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.doc_ids: List[str] = []
        
        if persist_path and os.path.exists(persist_path):
            self.load()
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the store."""
        new_embeddings = []
        
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.doc_id} has no embedding")
            
            self.documents[doc.doc_id] = doc
            new_embeddings.append(doc.embedding)
            
            if doc.doc_id not in self.doc_ids:
                self.doc_ids.append(doc.doc_id)
        
        # Update embeddings matrix
        new_emb_array = np.array(new_embeddings)
        
        if self.embeddings is None:
            self.embeddings = new_emb_array
        else:
            self.embeddings = np.vstack([self.embeddings, new_emb_array])
        
        # Auto-save if persist path set
        if self.persist_path:
            self.save()
        
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_metadata: Metadata filters
            
        Returns:
            List of SearchResult objects
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
        
        query_vec = np.array(query_embedding).reshape(1, -1)
        
        # Calculate similarities
        if self.similarity_metric == "cosine":
            # Normalize vectors
            query_norm = query_vec / np.linalg.norm(query_vec)
            emb_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            similarities = np.dot(emb_norm, query_norm.T).flatten()
        elif self.similarity_metric == "euclidean":
            distances = np.linalg.norm(self.embeddings - query_vec, axis=1)
            similarities = -distances  # Negative so higher is better
        else:  # dot product
            similarities = np.dot(self.embeddings, query_vec.T).flatten()
        
        # Apply metadata filters if provided
        valid_indices = np.arange(len(self.doc_ids))
        
        if filter_metadata:
            filtered_indices = []
            for i, doc_id in enumerate(self.doc_ids):
                doc = self.documents[doc_id]
                if all(doc.metadata.get(k) == v for k, v in filter_metadata.items()):
                    filtered_indices.append(i)
            valid_indices = np.array(filtered_indices) if filtered_indices else np.array([])
        
        if len(valid_indices) == 0:
            return []
        
        # Get top-k
        valid_similarities = similarities[valid_indices]
        top_k_indices = np.argsort(valid_similarities)[-k:][::-1]
        
        # Create results
        results = []
        for rank, idx in enumerate(top_k_indices):
            actual_idx = valid_indices[idx]
            doc_id = self.doc_ids[actual_idx]
            doc = self.documents[doc_id]
            
            results.append(SearchResult(
                document=doc,
                score=float(similarities[actual_idx]),
                rank=rank + 1
            ))
        
        return results
    
    def delete_document(self, doc_id: str):
        """Delete a document from the store."""
        if doc_id not in self.documents:
            return
        
        # Remove from documents
        del self.documents[doc_id]
        
        # Remove from embeddings
        idx = self.doc_ids.index(doc_id)
        self.doc_ids.remove(doc_id)
        
        if self.embeddings is not None:
            self.embeddings = np.delete(self.embeddings, idx, axis=0)
        
        if self.persist_path:
            self.save()
    
    def save(self):
        """Save vector store to disk."""
        if not self.persist_path:
            return
        
        Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'doc_ids': self.doc_ids,
            'embedding_dim': self.embedding_dim,
            'similarity_metric': self.similarity_metric
        }
        
        with open(self.persist_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved vector store to {self.persist_path}")
    
    def load(self):
        """Load vector store from disk."""
        if not self.persist_path or not os.path.exists(self.persist_path):
            return
        
        with open(self.persist_path, 'rb') as f:
            data = pickle.load(f)
        
        self.documents = data['documents']
        self.embeddings = data['embeddings']
        self.doc_ids = data['doc_ids']
        self.embedding_dim = data['embedding_dim']
        self.similarity_metric = data['similarity_metric']
        
        logger.info(f"Loaded vector store from {self.persist_path}")
    
    def clear(self):
        """Clear all documents from store."""
        self.documents = {}
        self.embeddings = None
        self.doc_ids = []
        
        if self.persist_path:
            self.save()


class DocumentLoader:
    """Load documents from various sources."""
    
    @staticmethod
    def load_text(filepath: str, chunk_size: int = 1000, overlap: int = 200) -> List[Document]:
        """
        Load text file and split into chunks.
        
        Args:
            filepath: Path to text file
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks
            
        Returns:
            List of Document objects
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into chunks
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]
            
            # Try to break at sentence boundary
            if end < len(content):
                last_period = chunk.rfind('.')
                if last_period > chunk_size * 0.8:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            
            chunks.append(Document(
                content=chunk.strip(),
                metadata={
                    'source': filepath,
                    'chunk_index': len(chunks),
                    'start_char': start,
                    'end_char': end
                }
            ))
            
            start = end - overlap
        
        return chunks
    
    @staticmethod
    def load_pdf(filepath: str, chunk_size: int = 1000, overlap: int = 200) -> List[Document]:
        """Load PDF file and split into chunks."""
        try:
            import PyPDF2
        except ImportError:
            raise ImportError("PyPDF2 required for PDF loading. Install with: pip install PyPDF2")
        
        documents = []
        
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                
                # Split page into chunks
                chunks = DocumentLoader._split_text(text, chunk_size, overlap)
                
                for i, chunk in enumerate(chunks):
                    documents.append(Document(
                        content=chunk,
                        metadata={
                            'source': filepath,
                            'page': page_num + 1,
                            'chunk_index': i
                        }
                    ))
        
        return documents
    
    @staticmethod
    def load_json(filepath: str) -> List[Document]:
        """Load JSON file as documents."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    content = item.get('content', str(item))
                    metadata = {k: v for k, v in item.items() if k != 'content'}
                else:
                    content = str(item)
                    metadata = {}
                
                metadata['source'] = filepath
                metadata['index'] = i
                
                documents.append(Document(content=content, metadata=metadata))
        else:
            documents.append(Document(
                content=json.dumps(data, indent=2),
                metadata={'source': filepath}
            ))
        
        return documents
    
    @staticmethod
    def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks


class RAGPipeline:
    """Complete RAG pipeline."""
    
    def __init__(
        self,
        llm_provider: Any,
        embedding_provider: Any,
        vector_store: Optional[VectorStore] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k_documents: int = 5,
        prompt_template: Optional[str] = None
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            llm_provider: LLM provider for generation
            embedding_provider: Provider for embeddings
            vector_store: Vector store instance
            chunk_size: Document chunk size
            chunk_overlap: Overlap between chunks
            k_documents: Number of documents to retrieve
            prompt_template: Template for RAG prompt
        """
        self.llm = llm_provider
        self.embedder = embedding_provider
        self.vector_store = vector_store or VectorStore()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_documents = k_documents
        
        self.prompt_template = prompt_template or """Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer: """
        
        self.document_count = 0
        self.query_history: List[Dict[str, Any]] = []
    
    def add_documents(self, sources: Union[str, List[str]]):
        """
        Add documents from file paths.
        
        Args:
            sources: File path or list of file paths
        """
        if isinstance(sources, str):
            sources = [sources]
        
        all_documents = []
        
        for source in sources:
            # Determine file type and load
            if source.endswith('.txt'):
                docs = DocumentLoader.load_text(source, self.chunk_size, self.chunk_overlap)
            elif source.endswith('.pdf'):
                docs = DocumentLoader.load_pdf(source, self.chunk_size, self.chunk_overlap)
            elif source.endswith('.json'):
                docs = DocumentLoader.load_json(source)
            else:
                # Try as text
                docs = DocumentLoader.load_text(source, self.chunk_size, self.chunk_overlap)
            
            all_documents.extend(docs)
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(all_documents)} documents...")
        
        for doc in all_documents:
            if doc.embedding is None:
                doc.embedding = self.embedder.embed(doc.content)
        
        # Add to vector store
        self.vector_store.add_documents(all_documents)
        self.document_count += len(all_documents)
        
        logger.info(f"Added {len(all_documents)} documents. Total: {self.document_count}")
    
    def query(
        self,
        question: str,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        return_sources: bool = True
    ) -> Union[str, Dict[str, Any]]:
        """
        Query the RAG pipeline.
        
        Args:
            question: User question
            k: Number of documents to retrieve
            filter_metadata: Metadata filters for search
            return_sources: Whether to return source documents
            
        Returns:
            Answer string or dict with answer and sources
        """
        k = k or self.k_documents
        
        # Generate query embedding
        query_embedding = self.embedder.embed(question)
        
        # Search for relevant documents
        results = self.vector_store.search(
            query_embedding,
            k=k,
            filter_metadata=filter_metadata
        )
        
        if not results:
            answer = "No relevant documents found to answer your question."
            context_docs = []
        else:
            # Build context from retrieved documents
            context_parts = []
            context_docs = []
            
            for result in results:
                context_parts.append(result.document.content)
                context_docs.append({
                    'content': result.document.content[:200] + "...",
                    'metadata': result.document.metadata,
                    'score': result.score
                })
            
            context = "\n\n".join(context_parts)
            
            # Generate answer using LLM
            prompt = self.prompt_template.format(
                context=context,
                question=question
            )
            
            answer = self.llm.complete(prompt)
        
        # Save to history
        self.query_history.append({
            'question': question,
            'answer': answer,
            'sources': context_docs if return_sources else None,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        if return_sources:
            return {
                'answer': answer,
                'sources': context_docs,
                'num_sources': len(context_docs)
            }
        else:
            return answer
    
    def clear_documents(self):
        """Clear all documents from the pipeline."""
        self.vector_store.clear()
        self.document_count = 0
        logger.info("Cleared all documents from RAG pipeline")
    
    def save(self, filepath: str):
        """Save RAG pipeline state."""
        data = {
            'document_count': self.document_count,
            'query_history': self.query_history,
            'config': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'k_documents': self.k_documents,
                'prompt_template': self.prompt_template
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved RAG pipeline to {filepath}")
    
    def load(self, filepath: str):
        """Load RAG pipeline state."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.document_count = data['document_count']
        self.query_history = data['query_history']
        
        config = data['config']
        self.chunk_size = config['chunk_size']
        self.chunk_overlap = config['chunk_overlap']
        self.k_documents = config['k_documents']
        self.prompt_template = config['prompt_template']
        
        logger.info(f"Loaded RAG pipeline from {filepath}")