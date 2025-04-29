from abc import ABC, abstractmethod
import datetime
import logging
from typing import Literal, Optional
import uuid

import requests
import urllib3
from urllib3.exceptions import InsecureRequestWarning


class APIAuthorize(ABC):
    """Interface for authorization and obtaining access to various APIs.

    Each concrete authorization class receives user data and issues an API token.
    """

    @property
    @abstractmethod
    def token(self) -> str:
        """Returns the token for API access.

        Returns:
            str: Valid authentication token.
        """
        pass

    @property
    @abstractmethod
    def cert_path(self) -> str:
        """Returns the path to the certificate for API access.

        Returns:
            str: Path to the certificate file.
        """
        pass


class LLMAuthorizeAdvanced(APIAuthorize):
    """Access to GigaChat via username/password authentication."""
    def __init__(
        self,
        username: str,
        password: str,
        auth_endpoint: str = "https://beta.saluteai.sberdevices.ru/v1/"
    ) -> None:
        """Initialize with GigaChat API access parameters.

        Args:
            username: GigaChat API username
            password: GigaChat API password
        """
        self.auth_endpoint = auth_endpoint
        self.__username = username
        self.__password = password
        self.__token_expiration_time = datetime.datetime.min
        self._token = ""
        self._set_token()

        urllib3.disable_warnings(
            urllib3.exceptions.InsecureRequestWarning
        )

    @property
    def token(self) -> str:
        """Get the current valid token, refreshing if necessary.

        Returns:
            str: Valid authentication token.
        """
        self._check_token()
        return self._token

    @token.setter
    def token(self, value: str) -> None:
        """Set the token value.

        Args:
            value: Token value to set.
        """
        self._token = value

    @property
    def cert_path(self) -> str:
        """Get the certificate path (not used in this implementation).

        Returns:
            str: Empty string as this implementation doesn't use certificates.
        """
        return ""

    @cert_path.setter
    def cert_path(self, value: str) -> None:
        """Certificate path setter (not used in this implementation).

        Args:
            value: Certificate path.
        """
        pass

    def _check_token(self) -> None:
        """Check if the current token is expired and refresh if needed."""
        if datetime.datetime.now() > self.__token_expiration_time:
            self._set_token()

    def _set_token(self) -> None:
        """Obtain a fresh access token from the API."""
        logger = logging.getLogger(__name__)
        token_url = f"{self.API_ENDPOINT}token"
        try:
            logger.info(
                "Getting token from GigaChat API",
                extra={"url": token_url, "username": self.__username}
            )
            response = requests.post(token_url, auth=(self.__username, self.__password))
            if response.status_code == 200:
                data = response.json()
                if "tok" not in data or "exp" not in data:
                    raise ValueError("Incorrect response format from GigaChat API")
                
                self.__token_expiration_time = datetime.datetime.utcfromtimestamp(data["exp"])
                self._token = data["tok"]
                logger.info("Successfully obtained authentication token")
            else:
                raise ValueError(
                    f"Failed to get token: HTTP {response.status_code} - {response.text}"
                )
        except requests.RequestException as e:
            logger.error(f"Network error during token retrieval: {str(e)}")
            raise
        except ValueError as e:
            logger.error(f"Error processing token response: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during token retrieval: {str(e)}")
            raise


class LLMAuthorizeEnablers(APIAuthorize):
    """Access to GigaChat via client_id/client_secret with certificate."""

    def __init__(
        self,
        auth_data: str,
        client_id: Optional[str] = None,
        auth_endpoint: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        auth_scope: str = Literal["GIGACHAT_API_PERS", "GIGACHAT_API_CORP", "GIGACHAT_API_B2B"],
        cert_path: Optional[str] = None
    ) -> None:
        """Initialize with GigaChat API access parameters.
        Args:
            auth_data: Authorization key for exchanging messages with GigaChat API
            client_id: GigaChat API client ID (used as RqUID)
            auth_endpoint: The authentication endpoint URL
            auth_scope: The authentication scope. Contains information about the API version 
                        being accessed. If you are using the API version for individual 
                        entrepreneurs or legal entities, specify this explicitly in the scope 
                        parameter (GIGACHAT_API_PERS, GIGACHAT_API_CORP, or GIGACHAT_API_B2B)
            cert_path: Path to the certificate for GigaChat API access
        """
        self.auth_endpoint = auth_endpoint
        self.auth_scope = auth_scope
        self.__client_id = client_id if client_id is not None else str(uuid.uuid4())
        self.__auth_data = auth_data
        self.__token_expiration_time = datetime.datetime.min
        self._cert_path = cert_path if cert_path is not None else ""
        self._token = ""
        self._set_token()

    @property
    def token(self) -> str:
        """Get the current valid token, refreshing if necessary.

        Returns:
            str: Valid authentication token.
        """
        self._check_token()
        return self._token

    @token.setter
    def token(self, value: str) -> None:
        """Set the token value.

        Args:
            value: Token value to set.
        """
        self._token = value

    @property
    def cert_path(self) -> str:
        """Get the path to the certificate.

        Returns:
            str: Path to the certificate file.
        """
        return self._cert_path

    @cert_path.setter
    def cert_path(self, value: str) -> None:
        """Set the certificate path.

        Args:
            value: Certificate path to set.
        """
        self._cert_path = value

    def _check_token(self) -> None:
        """Check if the current token is expired and refresh if needed."""
        if datetime.datetime.now() > self.__token_expiration_time:
            self._set_token()

    def _set_token(self) -> None:
        """Obtain a fresh access token from the API."""
        logger = logging.getLogger(__name__)
        headers = {
            "Authorization": f"Bearer {self.__auth_data}",
            "RqUID": self.__client_id,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        try:
            logger.info(
                "Getting token from GigaChat Auth API",
                extra={"url": self.auth_endpoint, "client_id": self.__client_id}
            )
            response = requests.post(
                self.auth_endpoint, 
                data={"scope": self.auth_scope}, 
                headers=headers, 
                verify=self._cert_path
            )
            if response.status_code == 200:
                data = response.json()
                if "access_token" not in data or "expires_at" not in data:
                    raise ValueError("Incorrect response format from GigaChat Auth API")
                
                self._token = data["access_token"]
                expiry_timestamp = int(data["expires_at"]) / 1000
                self.__token_expiration_time = datetime.datetime.fromtimestamp(expiry_timestamp)
                
                logger.info("Successfully obtained authentication token")
            else:
                raise ValueError(
                    f"Failed to get token: HTTP {response.status_code} - {response.text}"
                )
        except requests.RequestException as e:
            logger.error(f"Network error during token retrieval: {str(e)}")
            raise
        except ValueError as e:
            logger.error(f"Error processing token response: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during token retrieval: {str(e)}")
            raise
        

class SberDSAuthorize(APIAuthorize):
    @property
    def cert_path(self) -> bool:
        return False
    
    @property
    def token(self) -> str:
        return ""