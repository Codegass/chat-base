from typing import Any, Dict, List, Optional, Union
import os
import openai
from ...providers.base import ChatProvider
from ...utils.logging import ChatLogger
from ...utils.retry import retry_on_rate_limit


class OpenAIChat(ChatProvider):
    """OpenAI chat provider implementation."""

    def __init__(
        self, api_key: Optional[str] = None, model: Optional[str] = None, **kwargs: Any
    ):
        super().__init__(api_key, model)
        self.logger = ChatLogger("openai")

    def _initialize(self):
        """Initialize the OpenAI client."""
        import openai

        # Use API key from environment or passed during initialization
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key."
            )

        self.client = openai.OpenAI(api_key=api_key)

    @retry_on_rate_limit
    def get_response(
        self,
        message: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Get a response from OpenAI."""
        self.logger.log_request(message, model, system_prompt)

        try:
            # Ensure client is initialized
            if not hasattr(self, "client"):
                self._initialize()

            messages = []
            if system_prompt or self.system_prompt:
                messages.append(
                    {"role": "system", "content": system_prompt or self.system_prompt}
                )

            messages.append({"role": "user", "content": message})

            response = self.client.chat.completions.create(
                model=model or self.model or "gpt-4",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **{k: v for k, v in kwargs.items() if k != 'reasoning'},
            )

            result = response.choices[0].message.content
            self.logger.log_response(result)
            return result

        except Exception as e:
            self.logger.log_response("", error=e)
            raise

    @retry_on_rate_limit
    def get_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any]]:
        """Get a chat completion from OpenAI."""
        try:
            # Ensure client is initialized
            if not hasattr(self, "client"):
                self._initialize()

            response = self.client.chat.completions.create(
                model=model or self.model or "gpt-4",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **{k: v for k, v in kwargs.items() if k != 'reasoning'},
            )

            return response.choices[0].message.content

        except Exception as e:
            self.logger.log_error(e, "Chat completion failed")
            raise

    def validate_api_key(self) -> bool:
        """Validate the OpenAI API key."""
        try:
            # Make a minimal API call to validate the key
            openai.models.list()
            return True
        except Exception as e:
            self.logger.log_error(e, "API key validation failed")
            return False
