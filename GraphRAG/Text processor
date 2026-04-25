Try AI directly in your favorite apps … Use Gemini to generate drafts and refine content, plus get Gemini Pro with access to Google's next-gen AI for $26.99 $0 for 1 month
text_processor.py
"""
Text processing utilities for GraphRAG pipeline.
"""
import logging
from typing import List, Optional
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import GraphRAGConfig

logger = logging.getLogger(__name__)


class TextProcessor:
    """Handles text processing operations for the GraphRAG pipeline."""
    
    def __init__(self, config: GraphRAGConfig):
        """Initialize TextProcessor.
        
        Args:
            config: GraphRAG configuration object
        """
        self.config = config
        self.embeddings = None
        self.splitter = None
        if config.use_semantic_chunking:
            self.embeddings = OpenAIEmbeddings(
                model=config.openai_model_embed,
                api_key=config.api_key
            )
            self.splitter = SemanticChunker(
                self.embeddings,
                buffer_size=config.buffer_size,
                breakpoint_threshold_type=config.breakpoint_threshold_type,
                breakpoint_threshold_amount=config.breakpoint_threshold_amount,
                add_start_index=config.add_start_index
            )

        # Always available local fallback splitter (no embeddings)
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=getattr(config, "fallback_chunk_size_chars", 2400),
            chunk_overlap=getattr(config, "fallback_chunk_overlap_chars", 200),
        )
    
    def load_text_from_file(self, file_path: str) -> str:
        """Load text content from a file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Text content as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file cannot be read
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                logger.info(f"Successfully loaded text from {file_path}")
                return content
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except IOError as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into semantic chunks.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        try:
            if not self.config.use_semantic_chunking or self.splitter is None:
                chunks = self.fallback_splitter.split_text(text)
                logger.info(f"Split text into {len(chunks)} chunks (fallback chunker; semantic disabled)")
                return chunks

            chunks = self.splitter.split_text(text)
            logger.info(f"Split text into {len(chunks)} chunks (semantic chunker)")
            return chunks
        except Exception as e:
            # If embeddings/quota/rate limit fails, continue with a local chunker.
            logger.warning(f"Semantic chunking failed; falling back to local chunker. Error: {e}")
            chunks = self.fallback_splitter.split_text(text)
            logger.info(f"Split text into {len(chunks)} chunks (fallback after semantic failure)")
            return chunks
    
    def save_text_to_file(self, text: str, file_path: str) -> None:
        """Save text content to a file.
        
        Args:
            text: Text content to save
            file_path: Path where to save the text
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(text)
                logger.info(f"Successfully saved text to {file_path}")
        except IOError as e:
            logger.error(f"Error saving text to {file_path}: {e}")
            raise

