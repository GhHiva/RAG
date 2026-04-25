Try AI directly in your favorite apps … Use Gemini to generate drafts and refine content, plus get Gemini Pro with access to Google's next-gen AI for $26.99 $0 for 1 month
schema_loader.py
"""
Schema loader utility for extracting Pydantic model structures for GraphRAG prompts.
"""
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json
from importlib import import_module

# Add root directory to path so we can import schemas

ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Now import response_schemas
from schemas.schemasv2.schemas.response_schemas import response_schemas
# Default schemas path
SCHEMAS_PATH = ROOT_DIR / "schemas" / "schemasv2" / "schemas"
UNIVERSAL_DOC_LIST_PATH = (
    ROOT_DIR / "schemas" / "schemas" / "version2" / "universal_document_list" / "universal_document_list.json"
)

class SchemaLoader:
    """Utility class to load and format Pydantic schemas for prompts."""
    
    def __init__(self, schemas_base_path: Optional[Path] = None):
        """Initialize schema loader.
        
        Args:
            schemas_base_path: Base path to schemas directory. If None, uses default.
        """
        self.schemas_base_path = schemas_base_path or SCHEMAS_PATH
        self._schema_cache: Dict[str, Any] = {}
    def load_schema_model(self, schema_key: str) -> Any:
        """Load a Pydantic model by schema key.
        
        Args:
            schema_key: Key from response_schemas (e.g., 'agent_for_service_changes_and_updates_ab')
            
        Returns:
            Pydantic model class
        """
        if schema_key in self._schema_cache:
            return self._schema_cache[schema_key]
        
        if schema_key not in response_schemas:
            raise ValueError(f"Schema key '{schema_key}' not found in response_schemas")
        
        model = response_schemas[schema_key]
        self._schema_cache[schema_key] = model
        return model
    def get_schema_json(self, schema_key: str) -> Dict[str, Any]:
        """Get JSON schema from a Pydantic model.
        
        Args:
            schema_key: Key from response_schemas
            
        Returns:
            JSON schema dictionary
        """
        model = self.load_schema_model(schema_key)
        
        if not hasattr(model, "model_json_schema"):
            raise TypeError(f"Model for '{schema_key}' is not a Pydantic model")
        
        return model.model_json_schema() 
    def get_all_entity_types(self, schema_key_suffix: Optional[str] = None) -> set[str]:
        """Get all entity type titles from available schemas.
        
        Returns:
            List of entity type titles (e.g., ["Agent for Service Changes & Updates", ...])
        """
        entity_types = []
        for schema_key in response_schemas.keys():
            if schema_key_suffix and not schema_key.endswith(schema_key_suffix):
                continue
            try:
                schema = self.get_schema_json(schema_key)
                title = schema.get('title')
                if title:
                    entity_types.append(title)
            except Exception as e:
                # Skip schemas that can't be loaded
                print(f"Warning: Could not load schema '{schema_key}': {e}")
                continue
        return set(entity_types)  # Remove duplicates  
    def get_entity_types_string(
        self, separator: str = ", ", schema_key_suffix: Optional[str] = None
    ) -> str:
        """Get all entity types (titles) as a comma-separated string.

        Uses get_entity_types_with_documents so document context is available
        (even though we only return titles here).
        """
        entities = self.get_entity_types_with_documents(schema_key_suffix=schema_key_suffix)
        titles = [e.get("title") for e in entities if e.get("title")]
        return separator.join(set(titles))
    def get_entity_types_with_documents(
        self, schema_key_suffix: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """
        Return a list of dicts with schema_key, title, and document_names (if found).
        This merges response_schemas titles with universal_document_list.json documents.
        """
        # Load universal document list once
        doc_index: dict[str, dict[str, list[str]]] = {}
        try:
            data = json.loads(UNIVERSAL_DOC_LIST_PATH.read_text(encoding="utf-8"))
            tx_list = data.get("schemaObject", {}).get("transactionList", {})
            doc_list = data.get("schemaObject", {}).get("documentList", {})
            for tx_key, tx in tx_list.items():
                doc_keys = tx.get("documents", []) or []
                doc_names = []
                for dk in doc_keys:
                    doc_name = doc_list.get(dk, {}).get("document_name", dk)
                    doc_names.append(doc_name)
                doc_index[tx_key] = {
                    "doc_keys": doc_keys,
                    "doc_names": doc_names,
                }
        except Exception as e:
            print(f"Warning: could not load universal_document_list: {e}")
            doc_index = {}

        results: list[dict[str, Any]] = []
        for schema_key, _ in response_schemas.items():
            if schema_key_suffix and not schema_key.endswith(schema_key_suffix):
                continue
            try:
                schema_json = self.get_schema_json(schema_key)
                title = schema_json.get("title")
            except Exception as e:
                print(f"Warning: Could not load schema '{schema_key}': {e}")
                continue

            # Map schema_key to transaction_key (best guess: schema_key without trailing "_ab")
            tx_key_guess = schema_key
            if tx_key_guess.endswith("_ab"):
                tx_key_guess = tx_key_guess[:-3]

            results.append(
                {
                    "schema_key": schema_key,
                    "title": title,
                    "documents": doc_index.get(tx_key_guess, {}).get("doc_names", []),
                    "document_keys": doc_index.get(tx_key_guess, {}).get("doc_keys", []),
                }
            )
        return results

    def format_schema_for_prompt(self, schema_key: str) -> str:
        """Format schema structure for inclusion in prompts.
        
        Args:
            schema_key: Key from response_schemas
            
        Returns:
            Human-readable description of the schema structure
        """
        schema = self.get_schema_json(schema_key)
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        title = schema.get("title", schema_key)

        def describe_type(prop: dict[str, Any]) -> str:
            """Return a concise type description from common JSON schema shapes."""
            if "type" in prop:
                t = prop["type"]
                if isinstance(t, list):
                    return "/".join(str(x) for x in t)
                return str(t)
            # Fallbacks for schemas that use anyOf/oneOf without a top-level type
            if "anyOf" in prop:
                return " or ".join(
                    sorted({p.get("type", "object") for p in prop.get("anyOf", [])})
                )
            if "oneOf" in prop:
                return " or ".join(
                    sorted({p.get("type", "object") for p in prop.get("oneOf", [])})
                )
            return "object"

        lines: list[str] = [f"Entity Type: {title}", "Properties:"]
        for name, prop in properties.items():
            desc_parts: list[str] = []
            prop_type = describe_type(prop)
            if prop_type:
                desc_parts.append(f"type={prop_type}")

            if prop.get("description"):
                desc_parts.append(prop["description"])

            if enum_vals := prop.get("enum"):
                desc_parts.append(f"allowed values: {', '.join(map(str, enum_vals))}")

            if fmt := prop.get("format"):
                desc_parts.append(f"format={fmt}")

            required_flag = "required" if name in required else "optional"
            detail = " | ".join(desc_parts) if desc_parts else "no description provided"
            lines.append(f"- {name} ({required_flag}): {detail}")

        return "\n".join(lines)

    
        
   
        
    #     return entity_to_schema.get(entity_type)

