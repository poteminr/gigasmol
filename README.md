<div align="center">
  
<!-- [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/🤗-smolagents-orange.svg)](https://github.com/huggingface/smolagents)
[![GigaChat](https://img.shields.io/badge/GigaChat-API-green.svg)](https://gigachat.ru/) -->

</div>

<div align="center">
  <img src="./assets/logo.png" alt="GigaSmol Logo" width="500"/>
  <p><i>lightweight gigachat api wrapper for <a href="https://github.com/huggingface/smolagents">smolagents</a></i></p>
</div>

## Overview

gigasmol serves two primary purposes:

1. Provides **direct, lightweight access** to GigaChat models through GigaChat API without unnecessary abstractions
2. Creates a **smolagents-compatible wrapper** that lets you use GigaChat within agent systems

No complex abstractions — just clean, straightforward access to GigaChat's capabilities through smolagents.

```
GigaChat API + smolagents = gigasmol 💀
```

## Why gigasmol 💀?

- **Tiny Footprint**: Less than 1K lines of code total
- **Simple Structure**: Just 4 core files
- **Zero Bloat**: Only essential dependencies
- **Easy to Understand**: Read and comprehend the entire codebase in minutes
- **Maintainable**: Small, focused codebase means fewer bugs and easier updates
## Installation
### API-Only Installation (Default)
```bash
pip install gigasmol
```

### Full Installation with Agent Support
```bash
pip install "gigasmol[agent]"
```


## Quick Start
### Basic Usage with smolagents

```python
from gigasmol import GigaChatSmolModel
from smolagents import CodeAgent, ToolCallingAgent, DuckDuckGoSearchTool

# Initialize the GigaChat model with your credentials
model = GigaChatSmolModel(
    model_name="GigaChat-Max",  
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
)

# Create an CodeAgent with the model
code_agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model
)

# Run the code_agent
code_agent.run("What are the main tourist attractions in Moscow?")

# Create an CodeAgent with the model
tool_calling_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model
)

# Run the tool_calling_agent
tool_calling_agent.run("What are the main tourist attractions in Moscow?")
```


### Using Raw GigaChat API

```python
import json
from gigasmol import GigaChat

# Direct access to GigaChat API
gigachat = GigaChat(
    model_name="GigaChat-Max",
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
)

# Generate a response
response = gigachat.chat([
    {"role": "user", "content": "What is the capital of Russia?"}
])
print(response['answer']) # or print(response['response']['choices'][0]['message']['content'])
```

## How It Works

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
- `agents.ipynb`: Building code and tool agents with GigaChat and smolagents

## Acknowledgements

- [SberDevices](https://gigachat.ru/) for creating the GigaChat API
- [Hugging Face](https://huggingface.co/) for the smolagents framework
