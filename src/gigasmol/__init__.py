"""GigaSmol - Lightweight GigaChat API wrapper for smolagents"""

from gigasmol.gigachat_api.api_model import GigaChat

__version__ = "0.0.6"

__all__ = ["GigaChat"]

try:
    from gigasmol.models import GigaChatSmolModel
    __all__.append("GigaChatSmolModel")
except ImportError:
    # For API-only installations where smolagents is not installed
    class GigaChatSmolModel:
        """This class requires the smolagents package.
        
        Install with: pip install gigasmol
        """
        def __init__(self, *args, **kwargs):
            raise ImportError(
                'The smolagents package is required to use GigaChatSmolModel. '
                'Please install the full package with `pip install "gigasmol[agent]"`.'
            )
    
    __all__.append("GigaChatSmolModel")
