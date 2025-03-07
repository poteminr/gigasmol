import logging
from typing import List, Dict, Optional, Any, Tuple, Union

from smolagents.tools import Tool
from smolagents.models import Model, MessageRole
from huggingface_hub import ChatCompletionOutputMessage

from .gigachat_api.api_model import DialogRole, GigaChat
from .gigachat_api.auth import APIAuthorize, LLMAuthorizeEnablers


tool_role_conversions = {
    MessageRole.TOOL_CALL: DialogRole.ASSISTANT,
    MessageRole.TOOL_RESPONSE: DialogRole.USER,
    MessageRole.ASSISTANT: DialogRole.ASSISTANT,
    MessageRole.USER: DialogRole.USER,
    MessageRole.SYSTEM: DialogRole.SYSTEM,
}


class GigaChatSmolModel(Model):
    """A wrapper for the GigaChat model that implements the smolagents Model interface.
    
    This class handles communication with the GigaChat API, including authentication,
    message formatting, and response processing.
    
    Attributes:
        model_name: The name of the GigaChat model to use.
        temperature: Controls randomness in generation (0.0-1.0).
        top_p: Controls diversity via nucleus sampling (0.0-1.0).
        repetition_penalty: Penalizes repetition in generated text (>= 1.0).
        max_tokens: Maximum number of tokens to generate.
        profanity_check: Whether to enable profanity filtering.
        auth: Authentication handler for the GigaChat API.
        gigachat_instance: The underlying GigaChat client.
    """

    def __init__(
        self,
        model_name: str = "GigaChat",
        api_endpoint: str = "https://gigachat.devices.sberbank.ru/api/v1/",
        temperature: float = 0.1,
        top_p: float = 0.1,
        repetition_penalty: float = 1.0,
        max_tokens: int = 1500,
        profanity_check: bool = True,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        auth_endpoint: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        auth_scope: str = "GIGACHAT_API_CORP",
        cert_path: str = '',
    ) -> None:
        """Initialize a new GigaChatModel instance.
        
        Args:
            model_name: The name of the GigaChat model to use.
            temperature: Controls randomness in generation (0.0-1.0).
            top_p: Controls diversity via nucleus sampling (0.0-1.0).
            repetition_penalty: Penalizes repetition in generated text (>= 1.0).
            max_tokens: Maximum number of tokens to generate.
            profanity_check: Whether to enable profanity filtering.
            api_endpoint: The GigaChat API endpoint URL.
            cert_path: Path to the certificate file for API authentication.
            client_id: The client ID for API authentication.
            client_secret: The client secret for API authentication.
            auth_endpoint: The authentication endpoint URL.
            auth_scope: The authentication scope.
        """
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_tokens = max_tokens
        self.profanity_check = profanity_check
        
        self.gigachat_instance = GigaChat(
            model_name=self.model_name,
            api_endpoint=api_endpoint,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_tokens,
            profanity_check=self.profanity_check,
            client_id=client_id,
            client_secret=client_secret,
            auth_endpoint=auth_endpoint,
            auth_scope=auth_scope,
            cert_path=cert_path
        )

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
    ) -> ChatCompletionOutputMessage:
        try:
            messages = self.map_message_roles_to_api_format(messages)
            response = self.gigachat_instance.chat(messages=messages)
            answer = response['answer']
            if stop_sequences and isinstance(stop_sequences, list):
                answer = self.remove_stop_sequences(answer, stop_sequences)
                        
            return ChatCompletionOutputMessage(role="assistant", content=answer)
            
        except Exception as e:
            logging.error(f"Critical error in __call__: {str(e)}", exc_info=True)
            return ChatCompletionOutputMessage(
                role="assistant",
                content=f"Error in model execution: {str(e)}"
            )
            
    @staticmethod
    def map_message_roles_to_api_format(
        messages: List[Dict[str, str]],
    ) -> List[Tuple[DialogRole, str]]:
        converted_messages = []
        for message in messages:
            message_role = tool_role_conversions[message['role']]
            message_content = message['content'][0]['text']

            converted_messages.append((message_role, message_content))                  
        return converted_messages
    
    @staticmethod
    def remove_stop_sequences(content: str, stop_sequences: List[str]) -> str:
        for stop_seq in stop_sequences:
            if content[-len(stop_seq) :] == stop_seq:
                content = content[: -len(stop_seq)]
        return content
    
    def chat(
        self, 
        messages: Union[list[dict[str, str]], list[tuple[DialogRole, str]]], 
        params: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        return self.gigachat_instance.chat(messages, params)
    
    def get_available_models(self) -> list[str]:
        return self.gigachat_instance._get_list_model()