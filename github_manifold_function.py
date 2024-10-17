"""
title: GitHub Models Manifold Function
authors: cheahjs and bgeneto (improvements)
funding_url: https://github.com/open-webui
version: 0.1.2
license: MIT
requirements: pydantic, requests
environment_variables:
"""

from typing import Generator, Iterator, List, Union

import requests
from pydantic import BaseModel, Field


class Pipe:
    class Valves(BaseModel):
        GITHUB_PAT: str = Field(default="", description="GitHub Personal Access Token")
        GITHUB_MODELS_BASE_URL: str = Field(
            default="https://models.inference.ai.azure.com",
            description="GitHub Models Base URL",
        )
        MODELS_WHITELIST: str = Field(
            default="", description="Comma-separated list of model names to retrieve"
        )
        pass

    def __init__(self):
        self.type = "manifold"
        self.id = "github_models"
        self.name = "GitHub: "
        self.valves = self.Valves()
        pass

    def get_github_models(self, whitelist: str = "") -> List[dict]:
        if self.valves.GITHUB_PAT:
            try:
                headers = {
                    "Authorization": f"Bearer {self.valves.GITHUB_PAT}",
                    "Content-Type": "application/json",
                }

                r = requests.get(
                    f"{self.valves.GITHUB_MODELS_BASE_URL}/models", headers=headers
                )

                # ensure whitelist is a list
                whitelist = whitelist.strip()
                if whitelist == "*" or whitelist == "":
                    whitelist = []
                else:
                    whitelist = [item.strip().lower() for item in whitelist.split(",")]

                models = r.json()
                return [
                    {
                        "id": model["name"],
                        "name": (
                            model["friendly_name"]
                            if "friendly_name" in model
                            else model["name"]
                        ),
                        "description": (model["summary"] if "summary" in model else ""),
                    }
                    for model in models
                    if model["task"] == "chat-completion"
                    if not whitelist
                    or model["name"].lower() in whitelist
                    or model["friendly_name"].lower() in whitelist
                ]

            except Exception as e:

                print(f"Error: {e}")
                return [
                    {
                        "id": "error",
                        "name": "Could not fetch models from GitHub Models, please update the PAT in the valves.",
                    },
                ]
        else:
            return []

    def pipes(self) -> List[dict]:
        return self.get_github_models(self.valves.MODELS_WHITELIST)

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:

        headers = {
            "Authorization": f"Bearer {self.valves.GITHUB_PAT}",
            "Content-Type": "application/json",
        }

        allowed_params = {
            "messages",
            "temperature",
            "top_p",
            "stream",
            "stop",
            "model",
            "max_tokens",
            "stream_options",
        }

        # Remap the model name to the model id
        body["model"] = ".".join(body["model"].split(".")[1:])

        # Filter out any parameters that are not allowed
        filtered_body = {k: v for k, v in body.items() if k in allowed_params}

        # log fields that were filtered out as a single line
        if len(body) != len(filtered_body):
            print(
                f"Dropped params: {', '.join(set(body.keys()) - set(filtered_body.keys()))}"
            )

        try:
            r = requests.post(
                url=f"{self.valves.GITHUB_MODELS_BASE_URL}/chat/completions",
                json=filtered_body,
                headers=headers,
                stream=True,
            )

            r.raise_for_status()

            if body["stream"]:
                return r.iter_lines()
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e} {r.text}"
