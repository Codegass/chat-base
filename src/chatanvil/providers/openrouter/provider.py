from typing import Dict, List, Optional, Any, Union, Tuple
from openai import OpenAI
import os
from ...providers.base import ChatProvider
from ...utils.logging import ChatLogger
from ...utils.retry import retry_on_rate_limit

class OpenRouterChat(ChatProvider):
    """
    OpenRouter chat provider implementation.
    """

    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        referer: Optional[str] = None,
        title: Optional[str] = None,
        reasoning: bool = True,
        **kwargs: Any
    ):
        self.base_url = base_url
        self.referer = referer
        self.title = title
        self.reasoning = reasoning
        super().__init__(api_key, model)
        self.logger = ChatLogger("openrouter")
        # self._initialize()

    def _initialize(self):
        """Initialize the OpenRouter client with custom base URL."""
        api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass api_key."
            )

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
        )

    @retry_on_rate_limit
    def get_response(
        self,
        message: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.5,
        reasoning: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[str, Tuple[str, str]]:
        """
        Get a response from OpenRouter. For single message response.
        if the model support reasoning, it will be returned as the thinking token in the reasoning field.
        the response will be as a tuple of (response, reasoning) if reasoning is True.
        """
        self.logger.log_request(message, model, system_prompt)

        try:
            messages = []
            if system_prompt or self.system_prompt:
                messages.append(
                    {"role": "system", "content": system_prompt or self.system_prompt}
                )

            messages.append({"role": "user", "content": message})

            # Use instance reasoning if not overridden
            if reasoning is None:
                reasoning = self.reasoning

            response = self.get_chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                reasoning=reasoning,
                max_tokens=max_tokens,
                **kwargs
            )

            self.logger.log_response(response)
            return response

        except Exception as e:
            self.logger.log_response("", error=e)
            raise

    @retry_on_rate_limit
    def get_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.5,
        reasoning: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[str, Tuple[str, str]]:
        """Get a chat completion from OpenRouter with custom headers. Support multiple messages."""
        try:
            extra_headers = {}
            if self.referer:
                extra_headers["HTTP-Referer"] = self.referer
            if self.title:
                extra_headers["X-Title"] = self.title

            # Use instance reasoning if not overridden
            if reasoning is None:
                reasoning = self.reasoning

            # Create extra_body for include_reasoning
            extra_body = {}
            if reasoning:
                extra_body["include_reasoning"] = True

            response = self.client.chat.completions.create(
                model=model or self.model or "microsoft/phi-3-medium-128k-instruct:free",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers=extra_headers,
                extra_body=extra_body,
                **kwargs,
            )
            if reasoning:
                return (response.choices[0].message.content, response.choices[0].message.reasoning)
            else:
                return response.choices[0].message.content

        except Exception as e:
            self.logger.log_error(e, "Chat completion failed")
            raise

    def validate_api_key(self) -> bool:
        """Validate the OpenRouter API key."""
        try:
            # Make a minimal API call to validate the key
            self.client.models.list()
            return True
        except Exception as e:
            self.logger.log_error(e, "API key validation failed")
            return False
