from typing import Any, Dict, List, Tuple
from .base import BaseParser


class DefaultParser(BaseParser):
    """Default parser that returns responses as-is."""

    def __init__(self):
        self.type = "default"

    def parse_response(self, response: str or Tuple[str, str]) -> str or Tuple[str, str]:
        """Return the response without modification. If response is a tuple (content, reasoning), return both."""
        return response

    def extract_code(self, response: str) -> List[Dict[str, str]]:
        """No code extraction in default parser."""
        return []
