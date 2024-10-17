"""
title: AI21 Studio Manifold Pipe
authors: bgeneto
author_url: https://github.com/bgeneto/open-webui-ai21
funding_url: https://github.com/open-webui
version: 0.1.3
required_open_webui_version: 0.3.17
license: MIT
requirements: pydantic, ai21
environment_variables: AI21_API_KEY
"""

import os
from typing import Generator, List, Union

from ai21 import AI21Client
from ai21.models.chat.chat_message import AssistantMessage, SystemMessage, UserMessage
from pydantic import BaseModel, Field


class AI21Service:
    def __init__(self, pipe_id: str, valves: BaseModel):
        self.pipe_id = pipe_id
        self.valves = valves
        if not self.valves.AI21_API_KEY:
            raise Exception("Error: AI21_API_KEY is not set!")
        self.client = AI21Client(api_key=self.valves.AI21_API_KEY)
        pass

    def update_log_level(self, debug: bool) -> None:
        self.valves.AI21_DEBUG = debug
        pass

    def get_available_models(self) -> List[dict]:
        """
        Returns a list of available AI21 models.
        """
        return [
            {"id": "jamba-1.5-large", "name": "Jamba 1.5 Large"},
            {"id": "jamba-1.5-mini", "name": "Jamba 1.5 Mini"},
        ]

    def validate_model_id(self, model_id: str) -> None:
        """
        Validates the model ID.
        """
        available_models = [model["id"] for model in self.get_available_models()]
        if model_id not in available_models:
            raise ValueError(
                f"Invalid model ID: {model_id}. Available models: {', '.join(available_models)}"
            )
        pass

    def validate_messages(self, messages: List[dict]) -> None:
        """
        Validates the messages.
        """
        if not isinstance(messages, list) or not all(
            isinstance(message, dict) for message in messages
        ):
            raise ValueError("Messages must be a list of dictionaries")
        pass

    def convert_messages(
        self, messages: List[dict]
    ) -> List[Union[SystemMessage, UserMessage, AssistantMessage]]:
        """
        Converts the messages to the ai21 format.
        """
        ai21_messages = []
        for message in messages:
            if message["role"] == "system":
                ai21_messages.append(
                    SystemMessage(content=message["content"], role="system")
                )
            elif message["role"] == "user":
                ai21_messages.append(
                    UserMessage(content=message["content"], role="user")
                )
            elif message["role"] == "assistant":
                ai21_messages.append(
                    AssistantMessage(content=message["content"], role="assistant")
                )
            else:
                raise ValueError(f"Invalid role '{message['role']}'")
        return ai21_messages

    def process_request(self, body: dict) -> Union[str, Generator]:
        """
        Handles the pipe request.
        """
        try:
            model_id = body["model"].split(f"{self.pipe_id}.")[1]
            self.validate_model_id(model_id)

            messages = body["messages"]
            self.validate_messages(messages)

            ai21_messages = self.convert_messages(messages)
            stream = body.get("stream", False)

            response = self.client.chat.completions.create(
                messages=ai21_messages,
                model=model_id,
                max_tokens=body.get("max_tokens", 4096),
                temperature=body.get("temperature", 0.8),
                top_p=body.get("top_p", 0.9),
                stop=body.get("stop", []),
                stream=stream,
            )

            if stream:

                def stream_generator():
                    if self.valves.AI21_DEBUG:
                        print("**DEBUG: Entering stream_generator")
                    for chunk in response:
                        if self.valves.AI21_DEBUG:
                            print(f"**DEBUG: Received chunk: {chunk}")
                        if (
                            chunk.choices
                            and chunk.choices[0].delta
                            and chunk.choices[0].delta.content
                        ):
                            content = chunk.choices[0].delta.content
                            if self.valves.AI21_DEBUG:
                                print(f"**DEBUG: Yielding content: {content}")
                            yield content
                        else:
                            if self.valves.AI21_DEBUG:
                                print("**DEBUG: Skipping chunk due to missing content")

                return stream_generator()

            return response

        except Exception as e:
            raise Exception(f"Error: {e}")


class Pipe:
    class Valves(BaseModel):
        AI21_API_KEY: str = Field(default="")
        AI21_DEBUG: bool = Field(
            default=False, description="Turn debugging messages on/off"
        )
        pass

    def __init__(self):
        self.type = "manifold"
        self.id = "ai21_studio"
        self.name = "AI21 Studio: "
        self.config = {"AI21_API_KEY": os.getenv("AI21_API_KEY", "")}
        self.valves = self.Valves(**self.config)
        self.service = AI21Service(self.id, self.valves)
        pass

    def pipes(self) -> List[dict]:
        """
        Returns a list of available pipes.
        """
        return self.service.get_available_models()

    def pipe(self, body: dict) -> Union[str, Generator]:
        """
        Handles the pipe request.
        """
        self.service.update_log_level(self.valves.AI21_DEBUG)
        return self.service.process_request(body)
