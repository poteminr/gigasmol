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
### Full Installation (recommended)
```bash
pip install gigasmol
```

### API-Only Installation
If you only need direct access to the GigaChat API without the smolagents integration:
```bash
pip install "gigasmol[api]"
```

## Quick Start
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
```python
>>> "The main tourist attractions in Moscow are: Red Square, St. Basil's Cathedral, Kremlin, Bolshoi Theatre, Gorky Park, Tretyakov Gallery, Novodevichy Convent, and Moscow Metro."
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
