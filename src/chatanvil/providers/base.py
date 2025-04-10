from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple


class ChatProvider(ABC):
    """Base class for all chat providers."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.system_prompt: Optional[str] = None
        self._initialize()

    @abstractmethod
    def _initialize(self) -> None:
        """Initialize the provider-specific client and settings."""
        pass

    @abstractmethod
    def get_response(
        self,
        message: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[str, Tuple[str, str]]:
        """Get a response from the chat provider.

        Args:
            message: The user's message
            model: Optional model override
            system_prompt: Optional system prompt override
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Provider-specific parameters

        Returns:
            The model's response as a string or tuple of (response, reasoning)
        """
        pass

    @abstractmethod
    def get_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any], Tuple[str, str]]:
        """Get a chat completion from the provider.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Optional model override
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Provider-specific parameters

        Returns:
            The model's response, either as a string, structured data, or tuple of (response, reasoning)
        """
        pass

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt for future conversations."""
        self.system_prompt = prompt

    @abstractmethod
    def validate_api_key(self) -> bool:
        """Validate that the API key is correct and working.

        Returns:
            True if the API key is valid, False otherwise
        """
        pass
