from typing import Any, Dict, List, Optional, Type, Union, Tuple
from ..providers.base import ChatProvider
from .config import Config
from ..parsers.factory import ParserFactory


class Chat:
    """Main interface for interacting with chat providers."""

    def __init__(
        self,
        service_provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        parser_type: str = "default",
        reasoning: bool = False,
        **kwargs: Any,
    ):
        self.config = Config(service_provider=service_provider)
        self.provider_name = service_provider.lower()
        self.reasoning = reasoning

        # Initialize the parser
        self.parser = ParserFactory.get_parser(parser_type)

        # Get provider configuration
        provider_config = self.config

        # Update with any provided overrides
        if api_key:
            provider_config.api_key = api_key
        if model:
            provider_config.model = model

        # Initialize the appropriate provider
        self.provider = self._get_provider_instance(provider_config, **kwargs)

    def _get_provider_instance(self, config: Config, **kwargs: Any) -> ChatProvider:
        """Get the appropriate provider instance based on the service name."""
        # Import providers here to avoid circular imports
        from ..providers.openai import OpenAIChat
        from ..providers.claude import ClaudeChat
        from ..providers.groq import GroqChat
        from ..providers.ollama import OllamaChat
        from ..providers.openrouter import OpenRouterChat

        providers: Dict[str, Type[ChatProvider]] = {
            "openai": OpenAIChat,
            "claude": ClaudeChat,
            "groq": GroqChat,
            "ollama": OllamaChat,
            "openrouter": OpenRouterChat
        }

        if self.provider_name not in providers:
            raise ValueError(f"Unsupported provider: {self.provider_name}")

        provider_class = providers[self.provider_name]
        return provider_class(api_key=config.api_key, model=config.model, **kwargs)

    def get_response(
        self,
        message: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        reasoning: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[str, Tuple[str, str]]:
        """Get a response from the chat provider with parser."""
        # Use instance reasoning setting if not overridden in method call
        if reasoning is None:
            reasoning = self.reasoning

        # Get the raw response
        raw_response = self.provider.get_response(
            message=message,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning=reasoning,
            **kwargs,
        )

        # Use the parser to process the response
        return self.parser.parse_response(raw_response)

    def get_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any]]:
        """Get a chat completion from the provider with parser."""
        raw_response = self.provider.get_chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # If the response is a string, parse it
        if isinstance(raw_response, str):
            return self.parser.parse_response(raw_response)

        # If the response is a dictionary, return it as is
        return raw_response

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt for future conversations."""
        self.provider.set_system_prompt(prompt)

    def extract_code(self, response: str) -> List[Dict[str, str]]:
        """Extract code blocks from the response."""
        if self.parser.type == "default":
            raise ValueError("Default parser does not support code extraction.")
        return self.parser.extract_code(response)

    def set_parser(self, parser_type: str) -> None:
        """Change the current parser."""
        self.parser = ParserFactory.get_parser(parser_type)

    def get_current_parser(self) -> str:
        """Get the type of the current parser."""
        return self.parser.__class__.__name__.lower().replace("parser", "")
