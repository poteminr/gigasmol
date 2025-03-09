import uuid
import json
import logging
from copy import deepcopy
from typing import List, Dict, Optional, Any, Tuple, Union, Literal

from smolagents.tools import Tool
from smolagents.models import Model, MessageRole, parse_tool_args_if_needed, remove_stop_sequences
from huggingface_hub import ChatCompletionOutputMessage, ChatCompletionOutputToolCall

from .gigachat_api.api_model import DialogRole, GigaChat


TOOL_ROLE_CONVERSIONS = {
    MessageRole.TOOL_CALL: DialogRole.ASSISTANT,
    MessageRole.TOOL_RESPONSE: DialogRole.USER,
    MessageRole.ASSISTANT: DialogRole.ASSISTANT,
    MessageRole.USER: DialogRole.USER,
    MessageRole.SYSTEM: DialogRole.SYSTEM,
}


def get_tool_json_schema_gigachat(tool: Tool) -> Dict:
    """Convert a Tool object to a GigaChat-compatible function schema.
    
    This function transforms a smolagents Tool object into a JSON schema format
    that is compatible with GigaChat's function calling API. It handles type
    conversions and determines required parameters.
    
    Args:
        tool: A Tool object containing name, description, and input specifications.
        
    Returns:
        Dict: A dictionary representing the function schema in GigaChat's expected format
    """
    properties = deepcopy(tool.inputs)
    required = []
    for key, value in properties.items():
        if value["type"] == "any":
            value["type"] = "string"
        if not ("nullable" in value and value["nullable"]):
            required.append(key)
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def map_message_roles_to_api_format(messages: List[Dict[str, str]]) -> List[Tuple[DialogRole, str]]:
    """Convert smolagents message format to GigaChat API format.
    
    This function transforms messages from the smolagents format to the format
    expected by the GigaChat API. It converts message roles using the TOOL_ROLE_CONVERSIONS
    mapping and extracts the text content from each message.
    
    Args:
        messages: A list of message dictionaries in smolagents format, each containing
                 'role' and 'content' keys.
                 
    Returns:
        List[Tuple[DialogRole, str]]: A list of tuples containing the converted DialogRole
                                      and the message text content.
    """
    converted_messages = []
    for message in messages:
        message_role = TOOL_ROLE_CONVERSIONS[message['role']]
        message_content = message['content'][0]['text']
        converted_messages.append((message_role, message_content))                  
    return converted_messages


def extract_tool_calls(response: Dict[str, Any]) -> Optional[ChatCompletionOutputToolCall]:
    """Extract and format tool calls from a raw GigaChat API response.
    
    This utility function processes a raw GigaChat response and extracts any function
    calls, formatting them into the standardized structure with unique IDs.
    
    Args:
        response: The raw response from GigaChat API
        
    Returns:
        ChatCompletionOutputToolCall
    """    
    tool_calls = []        
    for choice in response['response']['choices']:
        if 'message' in choice and 'function_call' in choice['message']:
            func_call = choice['message']['function_call']
            call_id = f"call_{str(uuid.uuid4())[:8]}"
            arguments = func_call['arguments']
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments)
            
            formatted_call = {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": func_call['name'],
                    "arguments": arguments
                }
            }
            tool_calls.append(formatted_call)
    return ChatCompletionOutputToolCall.parse_obj(tool_calls) if tool_calls else None


def create_final_answer_tool_call(answer: str) -> ChatCompletionOutputToolCall:
    """Create a FinalAnswerTool call with the given answer.
    
    This helper method creates a properly formatted tool call for the FinalAnswerTool
    using the provided answer as the argument.
    
    Args:
        answer: The text answer to include in the tool call.
        
    Returns:
        ChatCompletionOutputToolCall: A formatted tool call for FinalAnswerTool.
    """
    call_id = f"call_{str(uuid.uuid4())[:8]}"
    final_answer_call = [{
        "id": call_id,
        "type": "function",
        "function": {
            "name": "final_answer",
            "arguments": json.dumps({"answer": answer})
        }
    }]
    return ChatCompletionOutputToolCall.parse_obj(final_answer_call)


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
        auth_scope: Literal["GIGACHAT_API_PERS", "GIGACHAT_API_CORP", "GIGACHAT_API_B2B"] = "GIGACHAT_API_CORP",
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
            messages = map_message_roles_to_api_format(messages)
            functions = [get_tool_json_schema_gigachat(tool) for tool in tools_to_call_from] if tools_to_call_from else None
            response = self.chat(messages=messages, functions=functions)
            answer = response.get('answer', '')
            tool_calls = extract_tool_calls(response)

            if tool_calls is None and tools_to_call_from is not None:
                tool_calls = create_final_answer_tool_call(answer)
            
            if stop_sequences and isinstance(stop_sequences, list):
                answer = remove_stop_sequences(answer, stop_sequences)
                
            return parse_tool_args_if_needed(
                ChatCompletionOutputMessage(
                    role="assistant",
                    content=answer,
                    tool_calls=tool_calls
                )
            )
        except Exception as e:
            logging.error(f"Critical error in __call__: {str(e)}", exc_info=True)
            return ChatCompletionOutputMessage(
                role="assistant",
                content=f"Error in model execution: {str(e)}"
            )  

    def chat(
        self, 
        messages: Union[list[dict[str, str]], list[tuple[DialogRole, str]]], 
        params: Optional[dict[str, Any]] = None,
        functions: Optional[list[dict[str, Any]]] = None,
        function_call: Optional[Union[str, dict[str, str]]] = None,
    ) -> dict[str, Any]:
        return self.gigachat_instance.chat(messages, params, functions, function_call)
    
    def get_available_models(self) -> list[str]:
        return self.gigachat_instance._get_list_model()