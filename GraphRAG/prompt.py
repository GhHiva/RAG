Try AI directly in your favorite apps … Use Gemini to generate drafts and refine content, plus get Gemini Pro with access to Google's next-gen AI for $26.99 $0 for 1 month
prompts.py
"""
Prompt templates for GraphRAG legal document processing.
"""
from typing import Dict, Any
from schema_loader import SchemaLoader


class LegalDocumentPrompts:
    """Collection of prompt templates for legal document processing."""
    
    @staticmethod
    def get_entity_types_prompt(schema_key: str) -> str:
        """Get the entity types prompt template."""
        schema_loader = SchemaLoader()
        # IMPORTANT:
        # LangChain's ChatPromptTemplate treats `{...}` as template variables.
        # Since our schema text + example JSON contain braces, we must escape them.
        def _escape_curly(text: str) -> str:
            return text.replace("{", "{{").replace("}", "}}")

        schema_info = _escape_curly(schema_loader.format_schema_for_prompt(schema_key))
        schema_json = schema_loader.get_schema_json(schema_key)
        title = schema_json.get('title', 'Entity')
        properties = schema_json.get('properties', {})
        property_names = ", ".join(properties.keys()) or "(no properties found in schema)"

        # Pull associated documents (from schema_loader.get_entity_types_with_documents)
        docs_block = ""
        docs = []
        try:
            for entry in schema_loader.get_entity_types_with_documents():
                if entry.get("schema_key") == schema_key:
                    docs = entry.get("documents", []) or []
                    break
            if docs:
                docs_lines = "\n".join([f"- {d}" for d in docs])
                docs_block = f"\nAssociated document types:\n{docs_lines}\n"
        except Exception:
            docs_block = ""

        title_with_docs = f"{title} ({', '.join(docs)})" if docs else title
        
        return f"""
Goal:
Read the provided text chunks and extract every {title_with_docs} entity as JSON.

Associated document types:
{docs_block or "- None provided"}

What to extract:
- Always set "document_name" using one of the associated document types ({', '.join(docs) if docs else 'if unknown, use "unknown"'})
- Always set "chunk_numbers" as a list of chunk indices (integers) where you found the information.
  The input is formatted like "Chunk N:" so use those N values (1-based).
- Always set "document_date" as the document's date in ISO format "YYYY-MM-DD" if you can find it, otherwise null.
  - Look for labels like Effective, Dated, Date, As of, On, or header/footer dates.
  - If information for this entity spans multiple chunks, include ALL relevant chunk numbers in "chunk_numbers".
  - If you see multiple distinct document dates that clearly correspond to different documents/entities, output multiple items (one per date/document).
- Always set "transaction_date" as the overall transaction-level date in ISO format "YYYY-MM-DD" if you can find it, otherwise null.
  - This is the date that applies to the transaction as a whole (e.g., filing date / effective date for the transaction), not necessarily a single document.
  - If the transaction date appears in a different chunk than the document details, include that chunk number in "chunk_numbers" as well.
  - If multiple transaction dates appear, choose the one explicitly tied to the transaction header/event; otherwise set null.
- Only use the properties defined in the schema; do not invent new fields
- If a property is not mentioned in the text, set it to null

Schema Structure (for guidance):
{schema_info}

Property keys you may emit:
{property_names}

Required output format (JSON only, no prose or markdown):
{{{{
  "{title}": [
    {{{{
      "chunk_numbers": [1],
      "document_name": "<one of: {', '.join(docs) if docs else 'use unknown if not provided'}>",
      "document_date": "YYYY-MM-DD or null",
      "transaction_date": "YYYY-MM-DD or null",
      "properties": {{{{
        "<schema_property_1>": "<value from text or null>",
        "<schema_property_2>": "<value from text or null>"
      }}}}
    }}}}
  ]
}}}}

Rules:
- Output MUST be valid JSON with a single top-level key "{title}"
- Include "chunk_numbers", "document_name", "document_date", "transaction_date", and a nested "properties" object populated only with schema properties
- Use null when a property exists in the schema but no value is found in the text
- Do not omit text chunks; extract from all provided chunks
- No explanations before or after the JSON

Text chunks to process:
{{input_text}}
"""

#     @staticmethod
#     def get_entity_extraction_prompt() -> str:
#         """Get the entity extraction prompt template."""
#         return """
# -Goal-
# You are given a legal or corporate text document along with a list of entity types.

# Your task is to carefully read the provided text and identify all entities that belong to the specified types. Note that each type may include several subtypes. First, determine the main type, and then go through the text to identify the specific subtype associated with each entity.

# * Do not skip or ignore any text chunks.
# * Extract every entity of the given types that appears in the text. In some form like letter maybe you can not specify the form, you have to go through the text to find out the type of the letter.
# * Ensure accuracy and completeness.

# -Steps-
# 1. Extract all entities of the specified types. For each entity, provide:
#    - entity_name: Capitalized name of the entity
#    - entity_type: One of the provided entity types with their subtypes: 

# {{
# }}

   
#     - entity_description: A clear, comprehensive description with its details of the entity, its attributes like address, names, sign, and dates.
   
# #    Format each entity as:
#     [<entity_name>, <entity_type>, <entity_description>]



#     @staticmethod
#     def get_entity_parsing_prompt() -> str:
#         """Get the entity parsing prompt template."""
#         return """
# -Goal-
# You are given a legal or corporate text document along with a list of entity types.

# Your task is to carefully read the provided text and identify all entities that belong to the specified types. 

# * Extract every entity of the specified types that appears in the text.
# * For each extracted entity, determine its subtype (if any) and then answer the questions listed in the entity description.
# * Only return the entity type (and subtype) plus answers to the questions that are relevant to that entity type.
# * Be thorough and accurate; include all occurrences and do not omit relevant entities.

# -Steps-
# 1. Extract all entities of the specified types. For each entity, provide:
#    - entity_name: Capitalized name of the entity
#    - entity_type: One of the provided entity types:

#    {{
# }}


#    - entity_description: Answer to each one of the questions which are specified to the entity type. Questions are: {entity_questions}

#    Format each entity as:
#    [<entity_name>, <entity_type>, <entity_description> ]

# ######################
# -Real Data-
# ######################
# text: {input_text}
# ######################
# output:
# """
#     # @staticmethod
#     # def get_entity_questions() -> Dict[str, list]:
#     #     """Get the entity-specific questions for parsing."""
#     #     return {  
#     #     }

