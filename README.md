# open-webui-functions
A collection of my own open-webui functions.

## What is Open WebUI?
Open WebUI is an extensible, feature-rich, and user-friendly self-hosted AI interface designed to operate entirely offline. It supports various LLM runners, including Ollama and OpenAI-compatible APIs.

## What are Open WebUI functions?
Functions are scripts, written in python, that are provided to an LLM at the time of the request. 
Funtions and Tools allow LLMs to perform actions and receive additional context as a result. Generally speaking, your LLM of choice will need to support function calling for tools to be reliably utilized.

[more info](https://docs.openwebui.com/tutorials/plugin/functions/#what-are-functions)


## What’s the difference between Functions and Pipelines?

The main difference between Functions and Pipelines are that Functions
are executed directly on the Open WebUI server, while Pipelines are
executed on a separate server. Functions are not capable of downloading
new packages in Open WebUI, meaning that you are only able to import
libraries into Functions that are packaged into Open WebUI. Pipelines,
on the other hand, are more extensible, enabling you to install any
Python dependencies your filter or pipe could need. Pipelines also
execute on a separate server, potentially reducing the load on your Open
WebUI instance.
