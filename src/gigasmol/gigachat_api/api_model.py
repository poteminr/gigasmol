import json
import logging
from datetime import datetime
from typing import Any, Iterator, Optional, Union, Literal, Dict, List, Tuple

import requests
from sseclient import SSEClient

from .auth import APIAuthorize, LLMAuthorizeEnablers, SberDSAuthorize

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        pass


class DialogRole(StrEnum):
    """Roles for GigaChat conversations."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
         

MessageList = Union[List[Tuple[DialogRole, str]], List[Dict[str, str]]]

         
class GigaChat:
    """Access to LLM based on GigaChat API."""

    def __init__(
        self,
        auth_data: Optional[str] = None,
        model_name: str = 'GigaChat',
        api_endpoint: str = "https://gigachat.devices.sberbank.ru/api/v1/",
        sber_ds: bool = False,
        authorize: Optional[APIAuthorize] = None,
        temperature: float = 0.1,
        top_p: float = 0.1,
        repetition_penalty: float = 1.0,
        max_tokens: int = 5000,
        n: int = 1,
        n_stream: int = 1,
        profanity_check: bool = True,
        client_id: Optional[str] = None,
        auth_endpoint: Optional[str] = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        auth_scope: Literal["GIGACHAT_API_PERS", "GIGACHAT_API_CORP", "GIGACHAT_API_B2B"] = "GIGACHAT_API_CORP",
        cert_path: Optional[str] = None,
    ) -> None:
        """Initialize with GigaChat API access parameters and response generation settings.

        Args:
            auth_data: Authorization key for exchanging messages with GigaChat API
            model_name: Model name of the GigaChat to use.
            api_endpoint: GigaChat API URL
            sber_ds: Using on sber-ds platform
            authorize: Authorization method for GigaChat API. If not provided, LLMAuthorizeEnablers will be used.
            temperature: Temperature parameter; higher values produce more diverse outputs (typical value 0.7).
            top_p: Another parameter for output diversity (typical value 0.1).
            repetition_penalty: Controls word repetition. Value 1.0 is neutral, 0-1 increases repetition, >1 decreases repetition.
            max_tokens: Maximum number of tokens to generate in the response.
            n: Number of completions to generate (non-streaming mode).
            n_stream: Number of completions to generate (streaming mode).
            profanity_check: Whether to enable profanity checking.
            client_id: GigaChat API client ID (used as RqUID).
            auth_endpoint: The authentication endpoint URL.
            auth_scope: The authentication scope. Contains information about the API version being accessed. 
            cert_path: Path to the certificate for GigaChat API access.
        """
        self.api_endpoint = api_endpoint
        self.sber_ds = sber_ds
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.n = n
        self.n_stream = n_stream
        self.profanity_check = profanity_check

        assert auth_data is not None or sber_ds is True, "auth_data is required for non-sber_ds mode"
        
        if authorize is not None:
            self.__authorize = authorize
        elif not sber_ds:
            self.__authorize = LLMAuthorizeEnablers(
                auth_data=auth_data,
                client_id=client_id,
                auth_endpoint=auth_endpoint,
                auth_scope=auth_scope,
                cert_path=cert_path
            )
        else:
            self.__authorize = SberDSAuthorize()
        self.__token_expiration_time = datetime.min

    def _get_list_model(self) -> List[str]:
        """Get a list of available models.

        The model list is useful when multiple models are available on the inference endpoint
        and a specific large or small model is needed.

        Returns:
            list[str]: List of available model names.

        Raises:
            RuntimeError: If the API response is invalid or an error occurs.
        """
        try:
            url = f"{self.api_endpoint}models"
            headers = {}
            
            if not self.sber_ds:
                headers["Authorization"] = f"Bearer {self.__authorize.token}"
            else:
                headers["Accept"] = "application/json"
            
            response = requests.get(url, headers=headers, verify=self.__authorize.cert_path)
            if response.status_code == 200:
                models = json.loads(response.content)
                if "data" not in models:
                    raise RuntimeError("Incorrect response from GigaChat API")
                logging.info(f"Available models: {models['data']}")
                return models["data"]
            else:
                raise RuntimeError(f"Error in getting list models: {response.status_code} - {response.content}")
        except Exception as e:
            logging.error(f"{str(e)}")
            raise e

    def _prepare_request(
        self,
        params: Dict[str, Any],
        messages: List[Dict[str, Any]],
        stream: bool,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
    ) -> Tuple[str, Dict[str, Any], str, str]:
        """Prepare and return all necessary parameters for a Gigachat API request.

        Args:
            params: Additional parameters for the request.
            messages: List of formatted messages.
            stream: Whether to use streaming mode.
            functions: Optional list of function definitions.
            function_call: Optional function call specification.

        Returns:
            tuple: URL, headers, JSON query, and model name.
        """
        temperature = params.get("temperature", self.temperature)
        top_p = params.get("top_p", self.top_p)
        repetition_penalty = params.get("repetition_penalty", self.repetition_penalty)
        model_name: str = params.get("model_name", self.model_name)
        max_tokens: int = params.get("max_tokens", self.max_tokens)
        profanity_check: bool = params.get("profanity_check", self.profanity_check)
        n = self.n_stream if stream else self.n

        url = f"{self.api_endpoint}chat/completions"
        headers = { "Content-Type": "application/json"}
        
        if not self.sber_ds:
            headers["Authorization"] = f"Bearer {self.__authorize.token}"
        else:
            headers["Accept"] = "application/json"

        params = {
            "model": f"{model_name}",
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "repetition_penalty": repetition_penalty,
            "profanity_check": bool(profanity_check),
            "stream": bool(stream),
            "max_tokens": int(max_tokens),
        }
        if functions is not None:
            params["functions"] = functions
            if function_call is not None:
                params["function_call"] = function_call
            else:
                params["function_call"] = "auto"

        if stream:
            params["update_interval"] = 0
        query = json.dumps(params)
        return url, headers, query, model_name
    
    def get_model_name(self) -> str:
        """Return the name of the default selected model.

        Returns:
            str: The model name.
        """
        return self.model_name

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Send a prompt to the model and return the model's response.

        Args:
            prompt: The user's prompt text.
            system_prompt: Optional system instructions.
            params: Additional parameters for controlling the response generation.
            functions: Optional list of function definitions for function calling.
            function_call: Optional function call specification.

        Returns:
            dict: A dictionary containing the LLM response and metadata.
        """
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages=messages, params=params, functions=functions, function_call=function_call)
                                     
    def complete_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
    ) -> Iterator[str]:
        """Send a prompt to the model and return the model's response as a stream.

        Args:
            prompt: The user's prompt text.
            system_prompt: Optional system instructions.
            params: Additional parameters for controlling the response generation.
            functions: Optional list of function definitions for function calling.
            function_call: Optional function call specification.

        Returns:
            Iterator[str]: An iterator that yields response tokens as they're generated.
        """
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat_stream(messages=messages, params=params, functions=functions, function_call=function_call)

    def tokens(self, text: str, params: Dict[str, Any]) -> int:
        """Count the number of tokens in the given text string.

        Args:
            text: The text string to count tokens for.
            params: Additional parameters as a dictionary.

        Returns:
            int: The number of tokens in the provided text.

        Raises:
            RuntimeError: If the API response is invalid or an error occurs.
        """
        try:
            model_name = params.get("model_name", self.model_name)
            headers = {"Content-Type": "application/json"}
            
            if not self.sber_ds:
                headers["Authorization"] = f"Bearer {self.__authorize.token}"
            else:
                headers["Accept"] = "application/json"
            
            query = json.dumps({"model": f"{model_name}", "input": [text]})
            url = f"{self.api_endpoint}tokens/count"
            logging.debug(f"Count tokens on LLM '{model_name}' and prompt: {text}...")
            response = requests.post(url, data=query, headers=headers, verify=self.__authorize.cert_path)
            if response.status_code == 200:
                answer = json.loads(response.content)
                if "tokens" not in answer[0] or "characters" not in answer[0]:
                    raise RuntimeError("Incorrect response from GigaChat API")
                return answer[0]["tokens"]
            else:
                raise RuntimeError(f"Error in running tokens method: {response.status_code} - {response.content}")
        except Exception as e:
            logging.error(f"{str(e)}")
            raise e

    def chat(
        self,
        messages: MessageList,
        params: Optional[Dict[str, Any]] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Send a list of messages to the model and return the model's response.

        Args:
            messages: Messages either as a list of (role, message) tuples or 
                    a list of dictionaries with 'role' and 'content' keys.
            params: Additional parameters for controlling the response generation.
            functions: Optional list of function definitions for function calling.
            function_call: Optional function call specification.

        Returns:
            Dict: A dictionary containing the LLM response and metadata.

        Raises:
            RuntimeError: If the API response is invalid or an error occurs.
        """
        try:
            chat_messages = self.validate_message_roles(messages)
            params = params if params is not None else {}
            url, headers, query, model_name = self._prepare_request(
                params=params,
                messages=chat_messages,
                stream=False,
                functions=functions,
                function_call=function_call,
            )
            logging.debug(f"Run chat on LLM '{model_name}' and chat: {chat_messages}...")
            response = requests.post(url, data=query, headers=headers, verify=self.__authorize.cert_path)
            if response.status_code == 200:
                answer = json.loads(response.content)
                if "choices" not in answer or "usage" not in answer:
                    raise RuntimeError("Incorrect response from GigaChat API")
                complete = answer["choices"][0]["message"]["content"]
                prompt_tokens = answer["usage"]["prompt_tokens"]
                completion_tokens = answer["usage"]["completion_tokens"]
                finish_reason = answer["choices"][0]["finish_reason"]
                logging.debug(f"answer: {complete}")
                logging.debug(f"prompt tokens number: {prompt_tokens} complete tokens number: {completion_tokens}")
                return {
                    "answer": complete,
                    "response": answer,
                    "prompt_tokens": prompt_tokens,
                    "answer_tokens": completion_tokens,
                    "finish_reason": finish_reason,
                    "info": json.dumps({'model':response.json()['model'],
                                           'x-request-id':response.headers.get('x-request-id'),
                                           'x-session-id':response.headers.get('x-session-id')})
                }
            else:
                raise RuntimeError(f"Error in running complete method: {response.status_code} - {response.content}")
        except Exception as e:
            logging.error(f"{str(e)}")
            raise e

    def chat_stream(
        self,
        messages: MessageList,
        params: Optional[Dict[str, Any]] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
    ) -> Iterator[str]:
        """Send a list of messages to the model and return the model's response as a stream.

        Args:
            messages: Messages either as a list of (role, message) tuples or 
                    a list of dictionaries with 'role' and 'content' keys.
            params: Additional parameters for controlling the response generation.
            functions: Optional list of function definitions for function calling.
            function_call: Optional function call specification.

        Returns:
            Iterator[str]: An iterator that yields response tokens as they're generated.

        Raises:
            RuntimeError: If the API response is invalid or an error occurs.
        """
        try:
            chat_messages = self.validate_message_roles(messages)
            params = params if params is not None else {}
            url, headers, query, model_name = self._prepare_request(
                params=params,
                messages=chat_messages,
                stream=True,
                functions=functions,
                function_call=function_call,
            )
            logging.debug(f"Run chat on LLM '{model_name}' and chat: {chat_messages}...")
            response = requests.post(url, data=query, headers=headers, verify=self.__authorize.cert_path, stream=True)
            if response.status_code == 200:
                finish_reason = str()
                client = SSEClient(response)
                for event in client.events():
                    if event.data == "[DONE]":
                        return {"finish_reason": finish_reason}
                    event_data = json.loads(event.data)
                    if "choices" not in event_data:
                        raise RuntimeError("No choices field in response from GigaChat API")
                    if "finish_reason" in event_data["choices"][0]:
                        finish_reason = event_data["choices"][0]["finish_reason"]
                    token = event_data["choices"][0]["delta"]["content"]
                    logging.debug(f"token: {token}")
                    yield token
            else:
                raise RuntimeError(f"Error in running complete method: {response.status_code} - {response.content}")
        except Exception as e:
            logging.error(f"{str(e)}")
            raise e
        
    @staticmethod
    def validate_message_roles(messages: MessageList) -> List[Dict[str, str]]:
        """Validate message roles and convert to the required format.

        Args:
            messages: Messages either as a list of (role, message) tuples or 
                    a list of dictionaries with 'role' and 'content' keys.

        Returns:
            List[Dict[str, str]]: A list of properly formatted message dictionaries.

        Raises:
            RuntimeError: If an unknown role is encountered.
        """
        chat_messages = []
        allowed_roles = {role for role in DialogRole}
        if isinstance(messages[0], tuple):
            for role, message in messages:
                if role in allowed_roles:
                    chat_messages.append({"role": role, "content": message})
                else:
                    raise RuntimeError(f"Passed unknown role: '{role}' in message: '{message}'. Available roles: '{str(list(allowed_roles))}'")
        else:
            for message in messages:
                if message['role'] in allowed_roles:
                    chat_messages.append({"role": message['role'], "content": message['content']})
                else:
                    raise RuntimeError(f"Passed unknown role: '{message['role']}' in message: '{message['content']}'. Available roles: '{str(list(allowed_roles))}'")
        return chat_messages
    
    def check_chat_profanity(self, messages: MessageList) -> bool:
        """Check if the given message history contains prohibited content.

        Args:
            messages: The message history as a list of (role, message) tuples.

        Returns:
            bool: True if the history contains prohibited content, False otherwise.
        """
        try:
            check_result =  self.chat(messages=messages, params={"profanity_check": True})
            return True if check_result["finish_reason"] == "blacklist" else False
        except Exception as err:
            logging.error(f"Error in checking censorship for chat: '{messages}', error: '{err}'")
            raise err
    
    def check_question_profanity(self, question: str) -> bool:
        """Check if the given question contains prohibited content.

        Args:
            question: The question text.    

        Returns:
            bool: True if the question contains prohibited content, False otherwise.
        """
        try:
            check_result = self.complete(prompt=question, params={"profanity_check": True})
            return True if check_result["finish_reason"] == "blacklist" else False
        except Exception as err:
            logging.error(f"Error in checking censorship for question: '{question}', error: '{err}'")
            raise err


class GigaFilter:
    """Class for direct interaction with GigaFilter for profanity checking."""

    def __init__(self, api_endpoint: str, authorize: APIAuthorize) -> None:
        """Initialize GigaFilter.

        Args:
            api_endpoint: The API endpoint URL.
            authorize: Authorization method for the API.
        """
        self.api_endpoint = api_endpoint
        self.__authorize = authorize
        
    def check_profanity(self, question: str, return_json: bool = False) -> Union[bool, Dict[str, Any]]:
        """Check if the given text contains profanity.

        Args:
            question: The text to check.
            return_json: If True, return the full JSON response; if False, return only a boolean result.

        Returns:
            Union[bool, dict]: Either a boolean indicating whether profanity was detected,
                              or the complete JSON response if return_json is True.
        """
        url = f"{self.api_endpoint}filter/check"
        headers = {
            "Authorization": f"Bearer {self.__authorize.token}",
            "Content-Type": "application/json",
        }
        query = {
            "model": "GigaFilter",
            "messages": [{
                "content": question,
                "role": "user"
            }]
        }
        query = json.dumps(query)
        response = requests.post(url, data=query, headers=headers, verify=self.__authorize.cert_path)
        if return_json:
            return response.json()
        else:
            return response.json()['is_profane']
