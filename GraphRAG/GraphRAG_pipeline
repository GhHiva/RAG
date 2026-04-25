"""
Main GraphRAG pipeline for legal document processing.
"""
import logging
import time
from typing import Optional
from pathlib import Path

from config import GraphRAGConfig
from text_processor import TextProcessor
from llm_client import LLMClient

logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    """Main pipeline for GraphRAG legal document processing."""
    
    def __init__(self, config: Optional[GraphRAGConfig] = None):
        """Initialize GraphRAG pipeline.
        
        Args:
            config: Configuration object. If None, will load from environment.
        """
        self.config = config or GraphRAGConfig.from_env()
        # Ensure API key is hydrated before validation
        if not self.config.api_key:
            try:
                self.config.api_key = self.load_api_key()
            except Exception:
                # Defer error handling to validate() for clear messaging
                pass
        self.config.validate()
        
        # Initialize components
        self.text_processor = TextProcessor(self.config)
        self.llm_client = LLMClient(self.config)
        
        logger.info("GraphRAG pipeline initialized successfully")
    
    def load_api_key(self) -> str:
        """Load API key from file.
        
        Returns:
            API key as string
        """
        try:
            with open(self.config.api_key_file_path, 'r') as key_file:
                api_key = key_file.readline().strip()
                logger.info("API key loaded successfully")
                return api_key
        except FileNotFoundError:
            logger.error(f"API key file not found: {self.config.api_key_file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading API key: {e}")
            raise
    
    def run_entity_extraction(self, schema_key: Optional[str], save_to_file: bool = True) -> str:
        """Run the entity extraction pipeline.
        
        Args:
            save_to_file: Whether to save results to file
            schema_key: Schema key from response_schemas (e.g., 'incorporation_ab')
        Returns:
            Extracted entities as string
        """
        logger.info("Starting entity extraction pipeline")
        start_time = time.time()
        
        try:
            # Load API key if not already set
            if not self.config.api_key:
                self.config.api_key = self.load_api_key()
                # Update LLM client with the loaded API key
                self.llm_client = LLMClient(self.config)
            
            # Load and process text
            text = self.text_processor.load_text_from_file(self.config.input_file_path)
            chunks = self.text_processor.split_text_into_chunks(text)
            
            # Extract entities (schema_key can be None)
            extracted_entities = self.llm_client.extract_entities(chunks, schema_key)
            
            # Save to file if requested
            if save_to_file:
                self.text_processor.save_text_to_file(
                    extracted_entities, 
                    self.config.output_file_path
                )
            
            duration = time.time() - start_time
            logger.info(f"Entity extraction completed in {duration:.2f} seconds")
            
            return extracted_entities
            
        except Exception as e:
            logger.error(f"Error in entity extraction pipeline: {e}")
            raise
    
    # def run_entity_parsing(self, save_to_file: bool = True) -> str:
    #     """Run the entity parsing pipeline.
        
    #     Args:
    #         save_to_file: Whether to save results to file
            
    #     Returns:
    #         Parsed entities as string
    #     """
    #     logger.info("Starting entity parsing pipeline")
    #     start_time = time.time()
        
    #     try:
    #         # Load API key if not already set
    #         if not self.config.api_key:
    #             self.config.api_key = self.load_api_key()
    #             # Update LLM client with the loaded API key
    #             self.llm_client = LLMClient(self.config)
            
    #         # Load the extracted entities from previous step
    #         if not Path(self.config.output_file_path_indexing).exists():
    #             raise FileNotFoundError(
    #                 f"Extracted entities file not found: {self.config.output_file_path_indexing}."
    #                 "Please run entity extraction first."
    #             )
            
    #         extracted_text = self.text_processor.load_text_from_file(self.config.output_file_path_indexing)
    #         chunks = self.text_processor.split_text_into_chunks(extracted_text)
            
    #         # Parse entities
    #         parsed_entities = self.llm_client.parse_entities(chunks)
            
    #         # Save to file if requested
    #         if save_to_file:
    #             self.text_processor.save_text_to_file(
    #                 parsed_entities, 
    #                 self.config.output_file_path_parsing
    #             )
            
    #         duration = time.time() - start_time
    #         logger.info(f"Entity parsing completed in {duration:.2f} seconds")
            
    #         return parsed_entities
            
    #     except Exception as e:
    #         logger.error(f"Error in entity parsing pipeline: {e}")
    #         raise
    
    def run_full_pipeline(self, schema_key: Optional[str]) -> tuple[str, str]:
        """Run the complete GraphRAG pipeline.
        
        Returns:
            Tuple of (extracted_entities, parsed_entities)
        """
        logger.info("Starting full GraphRAG pipeline")
        
        # Run entity extraction
        extracted_entities = self.run_entity_extraction(schema_key)
        
        # Run entity parsing
        # parsed_entities = self.run_entity_parsing()
        
        logger.info("Full GraphRAG pipeline completed successfully")
        
        return extracted_entities
