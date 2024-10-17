"""
title: DALL·E Manifold
description: A manifold function to integrate OpenAI's DALL-E models into Open WebUI.
author: bgeneto (based on Marc Lopez pipeline)
funding_url: https://github.com/open-webui
version: 0.1.4
license: MIT
requirements: pydantic
environment_variables: OPENAI_API_BASE_URL, OPENAI_IMG_API_KEY
"""

import os
from typing import Iterator, List, Union

from open_webui.utils.misc import get_last_user_message
from openai import OpenAI
from pydantic import BaseModel, Field


class Pipe:
    """OpenAI ImageGen pipeline"""

    class Valves(BaseModel):

        OPENAI_API_BASE_URL: str = Field(
            default="https://api.openai.com/v1", description="OpenAI API Base URL"
        )
        OPENAI_IMG_API_KEY: str = Field(default="", description="your OpenAI API key")
        IMAGE_SIZE: str = Field(default="1024x1024", description="Generated image size")
        NUM_IMAGES: int = Field(default=1, description="Number of images to generate")

    def __init__(self):
        self.type = "manifold"
        self.id = "DALL_E"
        self.name = "DALL-E"
        self.valves = self.Valves(
            **{
                "OPENAI_IMG_API_KEY": os.getenv("OPENAI_IMG_API_KEY", ""),
                "OPENAI_API_BASE_URL": os.getenv(
                    "OPENAI_API_BASE_URL", "https://api.openai.com/v1"
                ),
            }
        )

        self.client = OpenAI(
            base_url=self.valves.OPENAI_API_BASE_URL,
            api_key=self.valves.OPENAI_IMG_API_KEY,
        )

    def get_openai_assistants(self) -> List[dict]:
        """Get the available ImageGen models from OpenAI

        Returns:
            List[dict]: The list of ImageGen models
        """

        if self.valves.OPENAI_IMG_API_KEY:
            self.client = OpenAI(
                base_url=self.valves.OPENAI_API_BASE_URL,
                api_key=self.valves.OPENAI_IMG_API_KEY,
            )

            models = self.client.models.list()
            return [
                {
                    "id": model.id,
                    "name": model.id,
                }
                for model in models
                if "dall-e" in model.id
            ]

        return []

    def pipes(self) -> List[dict]:
        return self.get_openai_assistants()

    def pipe(self, body: dict) -> Union[str, Iterator[str]]:
        if not self.valves.OPENAI_IMG_API_KEY:
            return "Error: OPENAI_IMG_API_KEY is not set"

        self.client = OpenAI(
            base_url=self.valves.OPENAI_API_BASE_URL,
            api_key=self.valves.OPENAI_IMG_API_KEY,
        )

        model_id = body["model"]
        model_id = model_id.split(".")[1]
        user_message = get_last_user_message(body["messages"])

        response = self.client.images.generate(
            model=model_id,
            prompt=user_message,
            size=self.valves.IMAGE_SIZE,
            n=self.valves.NUM_IMAGES,
        )

        message = ""
        for image in response.data:
            if image.url:
                message += "![image](" + image.url + ")\n"

        yield message
