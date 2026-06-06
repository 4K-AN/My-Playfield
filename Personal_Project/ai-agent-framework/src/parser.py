"""
Output Parser untuk parse dan validate JSON dari LLM
Dengan error recovery dan schema enforcement
"""

import json
import re
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class ParsingError(Exception):
    """Error saat parsing output"""
    pass


class JsonExtractor:
    """Extract JSON dari text yang mungkin tidak pure JSON"""
    
    @staticmethod
    def extract_json_blocks(text: str) -> list[str]:
        """
        Extract semua JSON blocks dari text
        Mencoba multiple patterns:
        1. ```json ... ```
        2. ```{ ... }```
        3. Raw { ... }
        """
        json_blocks = []
        
        # Pattern 1: Fenced JSON blocks
        json_fenced = re.findall(r'```(?:json)?\s*([\s\S]*?)```', text)
        json_blocks.extend(json_fenced)
        
        # Pattern 2: Raw JSON objects (if not already captured)
        # Find balanced braces
        try:
            # Try to find JSON starting with {
            brace_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
            matches = re.finditer(brace_pattern, text)
            for match in matches:
                potential_json = match.group(0)
                if potential_json not in json_blocks:
                    json_blocks.append(potential_json)
        except Exception as e:
            logger.debug(f"Error in brace pattern matching: {e}")
        
        return json_blocks
    
    @staticmethod
    def extract_first_json(text: str) -> Optional[str]:
        """Extract JSON block pertama dari text"""
        blocks = JsonExtractor.extract_json_blocks(text)
        return blocks[0] if blocks else None


class OutputParser:
    """Parse dan validate LLM output"""
    
    def __init__(self, strict: bool = False):
        """
        Args:
            strict: Jika True, validation error langsung raise.
                   Jika False, coba recovery.
        """
        self.strict = strict
    
    def parse_json(self, text: str) -> Dict[str, Any]:
        """
        Parse JSON dari LLM output text
        
        Returns:
            Parsed JSON dictionary
        
        Raises:
            ParsingError: Jika parsing gagal
        """
        
        # Attempt 1: Direct JSON parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.debug("Direct JSON parsing failed, trying extraction")
        
        # Attempt 2: Extract JSON dari text
        extracted = JsonExtractor.extract_first_json(text)
        
        if extracted:
            try:
                return json.loads(extracted)
            except json.JSONDecodeError as e:
                if self.strict:
                    raise ParsingError(f"Failed to parse extracted JSON: {e}") from e
                logger.warning(f"Extracted JSON parsing failed: {e}")
        
        # Attempt 3: Common JSON fixing (missing quotes, etc)
        if not self.strict:
            try:
                fixed = self._attempt_json_repair(text)
                if fixed:
                    return json.loads(fixed)
            except Exception as e:
                logger.debug(f"JSON repair attempt failed: {e}")
        
        raise ParsingError(f"Could not parse JSON from text: {text[:100]}...")
    
    def _attempt_json_repair(self, text: str) -> Optional[str]:
        """
        Attempt untuk fix common JSON issues
        """
        
        # Extract potential JSON
        extracted = JsonExtractor.extract_first_json(text)
        if not extracted:
            return None
        
        # Try common fixes
        attempts = [
            extracted,
            # Add missing closing braces
            extracted + "}" if extracted.count('{') > extracted.count('}') else None,
            # Remove trailing comma before }
            re.sub(r',(\s*})' , r'\1', extracted),
            # Add quotes around unquoted keys (simple version)
            re.sub(r'(\w+):', r'"\1":', extracted),
        ]
        
        for attempt in attempts:
            if attempt:
                try:
                    json.loads(attempt)
                    return attempt
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def parse_to_model(
        self,
        text: str,
        model: Type[T],
        fallback: Optional[T] = None
    ) -> T:
        """
        Parse JSON dari text dan validate ke Pydantic model
        
        Args:
            text: Text yang di-parse
            model: Pydantic model class
            fallback: Fallback value jika parsing gagal
        
        Returns:
            Instance dari model class
        """
        
        try:
            json_data = self.parse_json(text)
            return model(**json_data)
        
        except ParsingError as e:
            if fallback:
                logger.warning(f"Parsing failed, using fallback: {e}")
                return fallback
            if self.strict:
                raise
            # Return empty model
            logger.error(f"Critical parsing error: {e}")
            raise
        
        except ValidationError as e:
            if fallback:
                logger.warning(f"Validation failed, using fallback: {e}")
                return fallback
            if self.strict:
                raise ParsingError(f"Validation error: {e}") from e
            raise ParsingError(f"Validation error: {e}") from e
    
    def parse_multiple(
        self,
        text: str,
        model: Type[T]
    ) -> list[T]:
        """
        Parse multiple JSON objects dari text
        Berguna untuk batch processing
        """
        
        blocks = JsonExtractor.extract_json_blocks(text)
        results = []
        
        for block in blocks:
            try:
                json_data = json.loads(block)
                if isinstance(json_data, list):
                    results.extend([model(**item) for item in json_data])
                else:
                    results.append(model(**json_data))
            except (json.JSONDecodeError, ValidationError) as e:
                if self.strict:
                    raise ParsingError(f"Error parsing block: {e}") from e
                logger.warning(f"Skipping invalid block: {e}")
        
        return results


class StructuredOutputSchema:
    """Helper untuk define output schema dan parsing instructions"""
    
    def __init__(self, model: Type[BaseModel]):
        self.model = model
        self.schema = model.model_json_schema()
    
    def get_parsing_instructions(self) -> str:
        """Get instruction untuk LLM tentang output format"""
        
        return f"""
Return response as valid JSON matching this schema:

{json.dumps(self.schema, indent=2)}

IMPORTANT:
- Return ONLY valid JSON, no additional text
- All required fields must be present
- Wrap response in JSON code block if needed:
  \`\`\`json
  {{...}}
  \`\`\`
"""
    
    def parse_response(
        self,
        response_text: str,
        parser: OutputParser = None,
        strict: bool = False
    ) -> BaseModel:
        """
        Parse response menggunakan schema
        """
        
        if parser is None:
            parser = OutputParser(strict=strict)
        
        try:
            return parser.parse_to_model(response_text, self.model)
        except ParsingError as e:
            logger.error(
                f"Failed to parse response to {self.model.__name__}: {e}"
            )
            raise


# Example usage
"""
from pydantic import BaseModel, Field

class PlanSchema(BaseModel):
    reasoning: str
    steps: list[str]
    confidence: float = Field(ge=0, le=1)

schema = StructuredOutputSchema(PlanSchema)

# Get instructions untuk LLM prompt
instructions = schema.get_parsing_instructions()
prompt = f"Generate a plan...\\n{instructions}"

# Parse response
response = llm.complete(prompt)
plan = schema.parse_response(response)
"""
