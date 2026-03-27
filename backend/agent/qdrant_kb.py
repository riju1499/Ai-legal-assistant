# """
# Qdrant Knowledge Base for legal documents
# Handles PDF ingestion and semantic search over legal knowledge
# """

# import logging
# from pathlib import Path
# from typing import List, Dict, Any, Optional
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams, PointStruct
# from sentence_transformers import SentenceTransformer
# import pypdf
# import hashlib
# import torch

# logger = logging.getLogger(__name__)


# class QdrantKnowledgeBase:
#     """
#     Qdrant vector database for legal PDF knowledge base
#     """
    
#     def __init__(
#         self, 
#         collection_name: str = "legal_knowledge",
#         embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
#         qdrant_path: str = "../qdrant_storage"
#     ):
#         """
#         Initialize Qdrant knowledge base
        
#         Args:
#             collection_name: Name of the Qdrant collection
#             embedding_model: SentenceTransformer model for embeddings
#             qdrant_path: Path to store Qdrant data (local mode)
#         """
#         self.collection_name = collection_name
#         self.qdrant_path = Path(qdrant_path)
        
#         # Initialize Qdrant client (local mode)
#         logger.info(f"Initializing Qdrant at {self.qdrant_path}")
#         self.client = QdrantClient(path=str(self.qdrant_path))
        
#         # Initialize embedding model
#         logger.info(f"Loading embedding model: {embedding_model}")
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.embedder = SentenceTransformer(embedding_model, device=self.device)
#         self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        
#         # Create collection if it doesn't exist
#         self._ensure_collection_exists()
    
#     def _ensure_collection_exists(self):
#         """Create collection if it doesn't exist"""
#         try:
#             collections = self.client.get_collections().collections
#             collection_names = [c.name for c in collections]
            
#             if self.collection_name not in collection_names:
#                 logger.info(f"Creating collection: {self.collection_name}")
#                 self.client.create_collection(
#                     collection_name=self.collection_name,
#                     vectors_config=VectorParams(
#                         size=self.embedding_dim,
#                         distance=Distance.COSINE
#                     )
#                 )
#                 logger.info(f"✓ Collection created")
#             else:
#                 logger.info(f"✓ Collection {self.collection_name} already exists")
#         except Exception as e:
#             logger.error(f"Error ensuring collection exists: {e}")
#             raise
    
#     def extract_text_from_pdf(self, pdf_path: Path) -> List[str]:
#         """
#         Extract text from PDF and split into pages
        
#         Args:
#             pdf_path: Path to PDF file
            
#         Returns:
#             List of text strings (one per page)
#         """
#         try:
#             with open(pdf_path, 'rb') as file:
#                 pdf_reader = pypdf.PdfReader(file)
#                 pages = []
                
#                 for page_num, page in enumerate(pdf_reader.pages):
#                     try:
#                         text = page.extract_text()
#                         if text and text.strip():
#                             pages.append(text.strip())
#                     except Exception as e:
#                         logger.warning(f"Failed to extract page {page_num} from {pdf_path.name}: {e}")
                
#                 logger.info(f"Extracted {len(pages)} pages from {pdf_path.name}")
#                 return pages
        
#         except Exception as e:
#             logger.error(f"Failed to extract text from {pdf_path}: {e}")
#             return []
    
#     def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
#         """
#         Split text into overlapping chunks
        
#         Args:
#             text: Input text
#             chunk_size: Size of each chunk in characters
#             overlap: Overlap between chunks
            
#         Returns:
#             List of text chunks
#         """
#         chunks = []
#         start = 0
        
#         while start < len(text):
#             end = start + chunk_size
#             chunk = text[start:end]
            
#             # Try to break at sentence or word boundary
#             if end < len(text):
#                 # Look for sentence boundary
#                 last_period = chunk.rfind('.')
#                 last_newline = chunk.rfind('\n')
#                 break_point = max(last_period, last_newline)
                
#                 if break_point > chunk_size * 0.5:  # Only break if we're past halfway
#                     chunk = text[start:start + break_point + 1]
#                     end = start + break_point + 1
            
#             chunks.append(chunk.strip())
#             start = end - overlap
        
#         return [c for c in chunks if len(c) > 50]  # Filter out very short chunks
    
#     def ingest_pdf(self, pdf_path: Path, chunk_size: int = 1000) -> int:
#         """
#         Ingest a PDF into the knowledge base
        
#         Args:
#             pdf_path: Path to PDF file
#             chunk_size: Size of text chunks
            
#         Returns:
#             Number of chunks added
#         """
#         logger.info(f"Ingesting {pdf_path.name}...")
        
#         # Extract text from PDF
#         pages = self.extract_text_from_pdf(pdf_path)
#         if not pages:
#             logger.warning(f"No text extracted from {pdf_path.name}")
#             return 0
        
#         # Combine and chunk
#         all_chunks = []
#         for page_num, page_text in enumerate(pages):
#             chunks = self.chunk_text(page_text, chunk_size=chunk_size)
#             for chunk_num, chunk in enumerate(chunks):
#                 all_chunks.append({
#                     'text': chunk,
#                     'source': pdf_path.name,
#                     'page': page_num + 1,
#                     'chunk': chunk_num
#                 })
        
#         if not all_chunks:
#             logger.warning(f"No chunks created from {pdf_path.name}")
#             return 0
        
#         # Generate embeddings
#         logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
#         texts = [c['text'] for c in all_chunks]
#         embeddings = self.embedder.encode(texts, show_progress_bar=False)
        
#         # Prepare points for Qdrant
#         points = []
#         for idx, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
#             # Generate unique ID based on content
#             content_hash = hashlib.md5(
#                 f"{chunk['source']}_{chunk['page']}_{chunk['chunk']}".encode()
#             ).hexdigest()
            
#             point = PointStruct(
#                 id=content_hash,
#                 vector=embedding.tolist(),
#                 payload={
#                     'text': chunk['text'],
#                     'source': chunk['source'],
#                     'page': chunk['page'],
#                     'chunk_id': chunk['chunk']
#                 }
#             )
#             points.append(point)
        
#         # Upload to Qdrant
#         logger.info(f"Uploading {len(points)} points to Qdrant...")
#         self.client.upsert(
#             collection_name=self.collection_name,
#             points=points
#         )
        
#         logger.info(f"✓ Ingested {pdf_path.name}: {len(points)} chunks")
#         return len(points)
    
#     def ingest_directory(self, pdf_dir: Path, chunk_size: int = 1000) -> Dict[str, int]:
#         """
#         Ingest all PDFs from a directory
        
#         Args:
#             pdf_dir: Directory containing PDFs
#             chunk_size: Size of text chunks
            
#         Returns:
#             Dictionary mapping filenames to chunk counts
#         """
#         pdf_dir = Path(pdf_dir)
#         pdf_files = list(pdf_dir.glob("*.pdf"))
        
#         logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
        
#         results = {}
#         for pdf_file in pdf_files:
#             try:
#                 count = self.ingest_pdf(pdf_file, chunk_size=chunk_size)
#                 results[pdf_file.name] = count
#             except Exception as e:
#                 logger.error(f"Failed to ingest {pdf_file.name}: {e}")
#                 results[pdf_file.name] = 0
        
#         return results
    
#     def search(
#         self, 
#         query: str, 
#         limit: int = 5,
#         score_threshold: float = 0.5
#     ) -> List[Dict[str, Any]]:
#         """
#         Search for relevant knowledge chunks
        
#         Args:
#             query: Search query
#             limit: Maximum number of results
#             score_threshold: Minimum similarity score
            
#         Returns:
#             List of relevant chunks with metadata
#         """
#         try:
#             # Generate query embedding
#             query_embedding = self.embedder.encode(query)
            
#             # Search Qdrant
#             results = self.client.search(
#                 collection_name=self.collection_name,
#                 query_vector=query_embedding.tolist(),
#                 limit=limit,
#                 score_threshold=score_threshold
#             )
            
#             # Format results
#             formatted = []
#             for result in results:
#                 formatted.append({
#                     'text': result.payload['text'],
#                     'source': result.payload['source'],
#                     'page': result.payload['page'],
#                     'score': result.score,
#                     'chunk_id': result.payload.get('chunk_id', 0)
#                 })
            
#             return formatted
        
#         except Exception as e:
#             logger.error(f"Search failed: {e}")
#             return []
    
#     def get_collection_info(self) -> Dict[str, Any]:
#         """Get information about the collection"""
#         try:
#             info = self.client.get_collection(self.collection_name)
#             return {
#                 'name': info.config.params.vectors.size if hasattr(info.config.params, 'vectors') else 'N/A',
#                 'vector_count': info.points_count,
#                 'status': info.status
#             }
#         except Exception as e:
#             logger.error(f"Failed to get collection info: {e}")
#             return {}
    
#     def clear_collection(self):
#         """Delete and recreate the collection"""
#         try:
#             logger.info(f"Clearing collection {self.collection_name}")
#             self.client.delete_collection(self.collection_name)
#             self._ensure_collection_exists()
#             logger.info("✓ Collection cleared")
#         except Exception as e:
#             logger.error(f"Failed to clear collection: {e}")

"""
Qdrant Knowledge Base for legal documents
Handles PDF ingestion and semantic search over legal knowledge
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import pypdf
import hashlib
import torch

logger = logging.getLogger(__name__)


class QdrantKnowledgeBase:
    """
    Qdrant vector database for legal PDF knowledge base
    """
    
    def __init__(
        self, 
        collection_name: str = "legal_knowledge",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        qdrant_path: str = "../qdrant_storage"
    ):
        """
        Initialize Qdrant knowledge base
        
        Args:
            collection_name: Name of the Qdrant collection
            embedding_model: SentenceTransformer model for embeddings
            qdrant_path: Path to store Qdrant data (local mode)
        """
        self.collection_name = collection_name
        self.qdrant_path = Path(qdrant_path)
        
        # Initialize Qdrant client (local mode)
        logger.info(f"Initializing Qdrant at {self.qdrant_path}")
        self.client = QdrantClient(path=str(self.qdrant_path))
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = SentenceTransformer(embedding_model, device=self.device)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        
        # Create collection if it doesn't exist
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"✓ Collection created")
            else:
                logger.info(f"✓ Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: Path) -> List[str]:
        """Extract text from PDF and split into pages"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                pages = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            pages.append(text.strip())
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num} from {pdf_path.name}: {e}")
                
                logger.info(f"Extracted {len(pages)} pages from {pdf_path.name}")
                return pages
        
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            return []
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence or word boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size * 0.5:
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return [c for c in chunks if len(c) > 50]
    
    def ingest_pdf(self, pdf_path: Path, chunk_size: int = 1000) -> int:
        """Ingest a PDF into the knowledge base"""
        logger.info(f"Ingesting {pdf_path.name}...")
        
        pages = self.extract_text_from_pdf(pdf_path)
        if not pages:
            logger.warning(f"No text extracted from {pdf_path.name}")
            return 0
        
        all_chunks = []
        for page_num, page_text in enumerate(pages):
            chunks = self.chunk_text(page_text, chunk_size=chunk_size)
            for chunk_num, chunk in enumerate(chunks):
                all_chunks.append({
                    'text': chunk,
                    'source': pdf_path.name,
                    'page': page_num + 1,
                    'chunk': chunk_num
                })
        
        if not all_chunks:
            logger.warning(f"No chunks created from {pdf_path.name}")
            return 0
        
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
        texts = [c['text'] for c in all_chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        
        points = []
        for idx, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
            content_hash = hashlib.md5(
                f"{chunk['source']}_{chunk['page']}_{chunk['chunk']}".encode()
            ).hexdigest()
            
            point = PointStruct(
                id=content_hash,
                vector=embedding.tolist(),
                payload={
                    'text': chunk['text'],
                    'source': chunk['source'],
                    'page': chunk['page'],
                    'chunk_id': chunk['chunk']
                }
            )
            points.append(point)
        
        logger.info(f"Uploading {len(points)} points to Qdrant...")
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        logger.info(f"✓ Ingested {pdf_path.name}: {len(points)} chunks")
        return len(points)
    
    def ingest_directory(self, pdf_dir: Path, chunk_size: int = 1000) -> Dict[str, int]:
        """Ingest all PDFs from a directory"""
        pdf_dir = Path(pdf_dir)
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
        
        results = {}
        for pdf_file in pdf_files:
            try:
                count = self.ingest_pdf(pdf_file, chunk_size=chunk_size)
                results[pdf_file.name] = count
            except Exception as e:
                logger.error(f"Failed to ingest {pdf_file.name}: {e}")
                results[pdf_file.name] = 0
        
        return results
    
    def search(
        self, 
        query: str, 
        limit: int = 5,
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search for relevant knowledge chunks using Qdrant v1.16+"""
        try:
            # Generate embedding
            query_embedding = self.embedder.encode(query).tolist()

            # NEW API (v1.16+)
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=limit,
                with_payload=True
            )

            formatted = []
            for point in search_result.points:
                if point.score >= score_threshold:
                    formatted.append({
                        "text": point.payload["text"],
                        "source": point.payload["source"],
                        "page": point.payload["page"],
                        "score": point.score,
                        "chunk_id": point.payload.get("chunk_id", 0)
                    })

            return formatted

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    
    # def search(
    #     self, 
    #     query: str, 
    #     limit: int = 5,
    #     score_threshold: float = 0.5
    # ) -> List[Dict[str, Any]]:
    #     """Search for relevant knowledge chunks using latest Qdrant client"""
    #     try:
    #         query_embedding = self.embedder.encode(query)

    #         results = self.client.search(
    #             collection_name=self.collection_name,
    #             query_vector=query_embedding.tolist(),
    #             limit=limit,
    #             with_payload=True
    #         )

    #         formatted = []
    #         for r in results:
    #             if r.score >= score_threshold:
    #                 formatted.append({
    #                     'text': r.payload['text'],
    #                     'source': r.payload['source'],
    #                     'page': r.payload['page'],
    #                     'score': r.score,
    #                     'chunk_id': r.payload.get('chunk_id', 0)
    #                 })

    #         return formatted

    #     except Exception as e:
    #         logger.error(f"Search failed: {e}")
    #         return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'name': info.config.params.vectors.size if hasattr(info.config.params, 'vectors') else 'N/A',
                'vector_count': info.points_count,
                'status': info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
    
    def clear_collection(self):
        """Delete and recreate the collection"""
        try:
            logger.info(f"Clearing collection {self.collection_name}")
            self.client.delete_collection(self.collection_name)
            self._ensure_collection_exists()
            logger.info("✓ Collection cleared")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
