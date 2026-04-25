Try AI directly in your favorite apps … Use Gemini to generate drafts and refine content, plus get Gemini Pro with access to Google's next-gen AI for $26.99 $0 for 1 month
llm_client.py
"""
LLM client for GraphRAG pipeline.
"""
import logging
from typing import List, Dict, Any, Tuple, Union, cast
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from text_processor import TextProcessor
from config import GraphRAGConfig
from prompts import LegalDocumentPrompts
from schema_loader import SchemaLoader

logger = logging.getLogger(__name__)


class LLMClient:
    """Handles LLM operations for the GraphRAG pipeline."""
    
    def __init__(self, config: GraphRAGConfig):
        """Initialize LLMClient.
        
        Args:
            config: GraphRAG configuration object
        """
        self.config = config
        # Initialize with temperature (it worked yesterday with temperature=0.3)
        # If the model doesn't support it, we'll catch the error during API calls and retry
        # model_name = config.openai_model_chat.lower()
        # if model_name in ['o1', 'o1-preview', 'o1-mini']:
        #     # These models definitely don't support custom temperature
        #     logger.info(f"Model {config.openai_model_chat} detected - using default temperature only")
        #     self.llm = ChatOpenAI(
        #         api_key=config.api_key,
        #         model=config.openai_model_chat,
        #     )
        #     self._use_custom_temperature = False
        # else:
        #     # Try with temperature first (as it worked yesterday)
        #     logger.info(f"Using model {config.openai_model_chat} with temperature={config.temperature}")
        self.llm = ChatOpenAI(
            temperature=config.temperature,
            api_key=config.api_key,
            model=config.openai_model_chat,
            seed=config.seed,
            top_p=config.top_p
        )
        self._use_custom_temperature = True
          # OpenAI changed API restrictions - some models (like gpt-5, o1) only support default temperature
        # Check model name and use appropriate initialization
        # model_name = config.openai_model_chat.lower()
        # models_without_custom_temp = ['gpt-5', 'o1', 'o1-preview', 'o1-mini']
        
        # if model_name in models_without_custom_temp:
        #     # These models only support default temperature (OpenAI changed this recently)
        #     logger.info(f"Model {config.openai_model_chat} detected - using default temperature only (OpenAI restriction)")
        #     self.llm = ChatOpenAI(
        #         api_key=config.api_key,
        #         model=config.openai_model_chat,
        #     )
        #     self._use_custom_temperature = False
        # else:
        #     # Other models support custom temperature
        #     logger.info(f"Using model {config.openai_model_chat} with temperature={config.temperature}")
        #     self.llm = ChatOpenAI(
        #         temperature=config.temperature,
        #         api_key=config.api_key,
        #         model=config.openai_model_chat,
        #         seed=config.seed,
        #         top_p=config.top_p
        #     )
        self._use_custom_temperature = True







        self.prompts = LegalDocumentPrompts()
    
    def extract_entities(self, text_chunks: Union[List[str], List[Tuple[int, str]]], schema_key: str) -> str:
        """Extract entities from text chunks.
        
        Args:
            text_chunks: List of text chunks to process
            schema_key: Schema key from response_schemas (e.g., 'incorporation_ab').
                       If None, will use a generic prompt without specific schema.
            
        Returns:
            Extracted entities from the text chunks as string
        """
        try:
            # Handle None schema_key - use generic prompt with JSON output
#             if schema_key is None:
#                 logger.warning("No schema_key provided, using generic entity extraction prompt")
#                 # Get all available entity types from schemas
#                 try:
#                     schema_loader = SchemaLoader()
#                     # Only allow Alberta schemas in generic mode to avoid outputs like
#                     # AmalgamationFD / AmalgamationBC, etc.
#                     entities_with_docs = schema_loader.get_entity_types_with_documents(schema_key_suffix="_ab")
#                     entity_types_list = sorted(
#                         [e.get("title") for e in entities_with_docs if e.get("title")]
#                     )
#                     entity_types_str = ", ".join(entity_types_list)
#                 except Exception as e:
#                     logger.warning(f"Could not load entity types: {e}")
#                     entity_types_str = "various entity types"
                
#                 prompt_template = """
# Goal:
# Your task is to carefully read the provided text chunks and extract all relevant entities in JSON format.

# Requirements:
# 1. Output ONLY valid JSON. Do not include any introductory text, explanations, or markdown formatting.
# 2. For each entity found in the text, identify its entity type (title) from the available types: {entity_types}
# 3. Extract all properties/attributes for each entity based on what information is available in the text chunks.
# 4. Structure the output as a JSON object where each key is an entity title, and the value is an object containing the properties found in the text.

# Output Format:
# {{
#   "Entity Title 1": {{
#     "property1": "value from text",
#     "property2": "value from text",
#     ...
#   }},
#   "Entity Title 2": {{
#     "property1": "value from text",
#     ...
#   }}
# }}

# Important:
# - Go through each text chunk and extract information for entities
# - Only include properties that are actually mentioned in the text
# - Use the exact entity title names from the schema types provided
# - Output must be valid JSON only, no additional text before or after

# Text chunks to process:
# {input_text}
# """
                # prompt = ChatPromptTemplate.from_template(prompt_template)
                # prompt = prompt.partial(entity_types=entity_types_str)
            if not schema_key:
                raise ValueError("schema_key is required; generic prompt is disabled.")
            else:
                prompt = ChatPromptTemplate.from_template(
                self.prompts.get_entity_types_prompt(schema_key)
                        )
                try:
                    entity_types_str = SchemaLoader().get_entity_types_string()
                    prompt = prompt.partial(entity_types=entity_types_str, language="English")
                except Exception as e:
                        logger.warning(f"Could not pre-fill entity_types: {e}")
                        prompt = prompt.partial(language="English")
                    
                # prompt = ChatPromptTemplate.from_template(
                #     self.prompts.get_entity_types_prompt(schema_key)
                # )
                # # Pre-fill defaults only for schema-specific prompts
                # try:
                #     # Use SchemaLoader directly; GraphRAGConfig does not expose this as a method
                #     entity_types_str = SchemaLoader().get_entity_types_string()
                #     prompt = prompt.partial(
                #         entity_types=entity_types_str,
                #         language="English"
                #     )
                # except Exception as e:
                #     logger.warning(f"Could not pre-fill entity_types: {e}")
                #     prompt = prompt.partial(language="English")
            
            parser = StrOutputParser()
            chain = prompt | self.llm | parser
            # Format text chunks as a string with clear chunk separators.
            # IMPORTANT: if caller passes (chunk_number, chunk_text), we preserve the provided chunk_number
            # (which should be the GLOBAL chunk number from the full document).
            formatted_text_parts: List[str] = []
            if text_chunks and isinstance(text_chunks[0], tuple):  # type: ignore[index]
                numbered = cast(List[Tuple[int, str]], text_chunks)
                for n, chunk in numbered:
                    formatted_text_parts.append(f"Chunk {n}:\n{chunk}")
            else:
                plain = cast(List[str], text_chunks)
                for i, chunk in enumerate(plain):
                    formatted_text_parts.append(f"Chunk {i+1}:\n{chunk}")

            formatted_text = "\n\n---CHUNK---\n\n".join(formatted_text_parts)
            
            try:
                Response = chain.invoke({"input_text": formatted_text})
            except Exception as api_error:
                # Check if it's a temperature-related error
                error_str = str(api_error).lower()
                if self._use_custom_temperature and ('temperature' in error_str or 'unsupported' in error_str):
                    # Model doesn't support custom temperature, retry without it
                    logger.warning(f"Model {self.config.openai_model_chat} doesn't support custom temperature, retrying with default temperature")
                    self.llm = ChatOpenAI(
                        api_key=self.config.api_key,
                        model=self.config.openai_model_chat,
                    )
                    self._use_custom_temperature = False
                    chain = prompt | self.llm | parser
                    Response = chain.invoke({"input_text": formatted_text})
                else:
                    # Re-raise if it's a different error
                    raise
            
            logger.info("Successfully extracted entities from all chunks")
            # return response_indexing
            
            return Response
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            raise
    
    # def parse_entities(self, text_chunks: List[str]) -> str:
    #     """Parse entities with detailed questions.
        
    #     Args:
    #         text_chunks: List of text chunks to process
            
    #     Returns:
    #         Parsed entities as string
    #     """
    #     try:
    #         prompt = ChatPromptTemplate.from_template(
    #             self.prompts.get_entity_parsing_prompt()
    #         )
            
    #         # Pre-fill defaults
    #         prompt = prompt.partial(
    #             entity_types=self.config.entity_types,
    #             entity_questions=self.prompts.get_entity_questions(),
    #             language="English"
    #         )
            
    #         parser = StrOutputParser()
    #         chain = prompt | self.llm | parser
    #         # Process each chunk
    #         # responses = []
    #         # for i, chunk in enumerate(text_chunks):
    #         #     logger.info(f"Parsing chunk {i+1}/{len(text_chunks)}")
    #         #     response = chain.invoke({"input_text": chunk})
    #         #     responses.append(response)
            
    #         # # Combine all responses
    #         # response_parsing = "\n\n".join(responses)
    #         Response_parsing = chain.invoke({"input_text": text_chunks})
    #         logger.info("Successfully parsed entities from all chunks")
    #         # return response_parsing
            
    #         return Response_parsing
            
    #     except Exception as e:
    #         logger.error(f"Error parsing entities: {e}")
    #         raise

