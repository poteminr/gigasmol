
<div align="center">
  
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/🤗-smolagents-orange.svg)](https://github.com/huggingface/smolagents)
[![GigaChat](https://img.shields.io/badge/GigaChat-API-green.svg)](https://gigachat.ru/)

</div>

<div align="center">
  <img src="./assets/logo.png" alt="GigaSmol Logo" width="400"/>
  <p><i>lightweight gigachat api wrapper for smolagents</i></p>
</div>

## Overview

GigaSmol serves two primary purposes:

1. Provides **direct, lightweight access** to GigaChat models through GigaChat API without unnecessary abstractions
2. Creates a **smolagents-compatible wrapper** that lets you use GigaChat within agent systems

No complex abstractions — just clean, straightforward access to GigaChat's capabilities through smolagents.

```
GigaChat API + smolagents = gigasmol 💀
```

## Quick Start
```bash
pip install gigasmol
```
### Basic Usage with smolagents

```python
from gigasmol import GigaChatSmolModel
from smolagents import CodeAgent, DuckDuckGoSearchTool

# Initialize the GigaChat model with your credentials
model = GigaChatSmolModel(
    model_name="GigaChat-Max",  
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
)

# Create an agent with the model
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model
)

# Run the agent
agent.run("What are the main tourist attractions in Moscow?")
```

### Using Raw GigaChat API

```python
import json
from gigasmol import GigaChat

# Load credentials from file
credentials = json.load(open('credentials.json'))

# Direct access to GigaChat API
gigachat = GigaChat(
    model_name="GigaChat-Max",
    client_id=credentials['client_id'],
    client_secret=credentials['client_secret'],
)

# Generate a response
response = gigachat.chat([
    {"role": "user", "content": "What is the capital of Russia?"}
])
print(response.choices[0].message.content)
```

## 🔍 How It Works

GigaSmol provides two layers of functionality:

```
┌───────────────────────────────────────────────────┐
│                    gigasmol                       │
├───────────────────────────────────────────────────┤
│ ┌───────────────┐          ┌───────────────────┐  │
│ │    Direct     │          │   smolagents      │  │
│ │ GigaChat API  │          │  compatibility    │  │
│ │    access     │          │      layer        │  │
│ └───────────────┘          └───────────────────┘  │
└───────────────────────────────────────────────────┘
    │                             │
    ▼                             ▼
┌─────────────┐           ┌────────────────┐
│ GigaChat API│           │ Agent systems  │
└─────────────┘           └────────────────┘
```

1. **Direct API Access**: Use `GigaChat` for clean, direct access to the API
2. **smolagents Integration**: Use `GigaChatSmolModel` to plug GigaChat into smolagents


## Examples

Check the `examples` directory:
- `structured_output.ipynb`: Using GigaChat for structured output
- `code_agents.ipynb`: Building code agents with GigaChat and smolagents

## Acknowledgements

- [SberDevices](https://gigachat.ru/) for creating the GigaChat API
- [Hugging Face](https://huggingface.co/) for the smolagents framework
