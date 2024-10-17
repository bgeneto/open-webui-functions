"""
title: FLUX.1 Schnell Manifold Function for Black Forest Lab Image Generation Models
author: bgeneto
author_url: https://github.com/bgeneto/open-webui-flux-image-gen
funding_url: https://github.com/open-webui
version: 0.1.5
license: MIT
requirements: pydantic, requests
environment_variables: FLUX_SCHNELL_API_BASE_URL, FLUX_SCHNELL_API_KEY
supported providers: huggingface.co, replicate.com, together.xyz
https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell
https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions
https://api.together.xyz/v1/images/generations
"""

import base64
import os
from typing import Any, Dict, Generator, Iterator, List, Union

import requests
from open_webui.utils.misc import get_last_user_message
from pydantic import BaseModel, Field


class Pipe:
    """
    Class representing the FLUX.1 Schnell Manifold Function.
    """

    class Valves(BaseModel):
        """
        Pydantic model for storing API keys and base URLs.
        """

        FLUX_SCHNELL_API_KEY: str = Field(
            default="", description="Your API Key for Flux.1 Schnell"
        )
        FLUX_SCHNELL_API_BASE_URL: str = Field(
            default="https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions",
            description="Base URL for the API",
        )

    def __init__(self):
        """
        Initialize the Pipe class with default values and environment variables.
        """
        self.type = "manifold"
        self.id = "FLUX_Schnell"
        self.name = "FLUX.1: "
        self.valves = self.Valves(
            FLUX_SCHNELL_API_KEY=os.getenv("FLUX_SCHNELL_API_KEY", ""),
            FLUX_SCHNELL_API_BASE_URL=os.getenv(
                "FLUX_SCHNELL_API_BASE_URL",
                "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell",
            ),
        )

    def url_to_img_data(self, url: str) -> str:
        """
        Convert a URL to base64-encoded image data.

        Args:
            url (str): The URL of the image.

        Returns:
            str: Base64-encoded image data.
        """
        headers = {"Authorization": f"Bearer {self.valves.FLUX_SCHNELL_API_KEY}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "application/octet-stream")
        encoded_content = base64.b64encode(response.content).decode("utf-8")
        return f"data:{content_type};base64,{encoded_content}"

    def stream_response(
        self, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> Generator[str, None, None]:

        yield self.non_stream_response(headers, payload)

    def get_img_extension(self, img_data: str) -> Union[str, None]:
        """
        Get the image extension based on the base64-encoded data.

        Args:
            img_data (str): Base64-encoded image data.

        Returns:
            Union[str, None]: The image extension or None if unsupported.
        """
        if img_data.startswith("/9j/"):
            return "jpeg"
        elif img_data.startswith("iVBOR"):
            return "png"
        elif img_data.startswith("R0lG"):
            return "gif"
        elif img_data.startswith("UklGR"):
            return "webp"
        return None

    def handle_json_response(self, response: requests.Response) -> str:
        """
        Handle JSON response from the API.

        Args:
            response (requests.Response): The response object.

        Returns:
            str: The formatted image data or an error message.
        """
        resp = response.json()
        if "output" in resp:
            img_data = resp["output"][0]
        elif "data" in resp and "b64_json" in resp["data"][0]:
            img_data = resp["data"][0]["b64_json"]
        else:
            return "Error: Unexpected response format for the image provider!"

        # split ;base64, from img_data
        try:
            img_data = img_data.split(";base64,")[1]
        except IndexError:
            pass

        img_ext = self.get_img_extension(img_data[:9])
        if not img_ext:
            return "Error: Unsupported image format!"

        # rebuild img_data with proper format
        img_data = f"data:image/{img_ext};base64,{img_data}"
        return f"![Image]({img_data})\n`GeneratedImage.{img_ext}`"

    def handle_image_response(self, response: requests.Response) -> str:
        """
        Handle image response from the API.

        Args:
            response (requests.Response): The response object.

        Returns:
            str: The formatted image data.
        """
        content_type = response.headers.get("Content-Type", "")
        # check image type in the content type
        img_ext = "png"
        if "image/" in content_type:
            img_ext = content_type.split("/")[-1]
        image_base64 = base64.b64encode(response.content).decode("utf-8")
        return f"![Image](data:{content_type};base64,{image_base64})\n`GeneratedImage.{img_ext}`"

    def non_stream_response(
        self, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> str:
        """
        Get a non-streaming response from the API.

        Args:
            headers (Dict[str, str]): The headers for the request.
            payload (Dict[str, Any]): The payload for the request.

        Returns:
            str: The response from the API.
        """
        try:
            response = requests.post(
                url=self.valves.FLUX_SCHNELL_API_BASE_URL,
                headers=headers,
                json=payload,
                stream=False,
                timeout=(3.05, 60),
            )
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                return self.handle_json_response(response)
            elif "image/" in content_type:
                return self.handle_image_response(response)
            else:
                return f"Error: Unsupported content type {content_type}"

        except requests.exceptions.RequestException as e:
            return f"Error: Request failed: {e}"
        except Exception as e:
            return f"Error: {e}"

    def pipes(self) -> List[Dict[str, str]]:
        """
        Get the list of available pipes.

        Returns:
            List[Dict[str, str]]: The list of pipes.
        """
        return [{"id": "flux_schnell", "name": "Schnell"}]

    def pipe(
        self, body: Dict[str, Any]
    ) -> Union[str, Generator[str, None, None], Iterator[str]]:
        """
        Process the pipe request.

        Args:
            body (Dict[str, Any]): The request body.

        Returns:
            Union[str, Generator[str, None, None], Iterator[str]]: The response from the API.
        """
        headers = {
            "Authorization": f"Bearer {self.valves.FLUX_SCHNELL_API_KEY}",
            "Content-Type": "application/json",
        }

        body["stream"] = False
        prompt = get_last_user_message(body["messages"])

        # Define headers and payload for different API providers
        headers_map = {
            "huggingface.co": {"x-wait-for-model": "true"},
            "replicate.com": {"Prefer": "wait=10"},
            "together.xyz": {},
        }

        # Define payload for different API providers
        payload_map = {
            "huggingface.co": {"inputs": prompt},
            "replicate.com": {
                "input": {
                    "prompt": prompt,
                    "go_fast": True,
                    "num_outputs": 1,
                    "aspect_ratio": "1:1",
                    "output_format": "webp",
                    "output_quality": 90,
                }
            },
            "together.xyz": {
                "model": "black-forest-labs/FLUX.1-schnell-Free",
                "prompt": prompt,
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "n": 1,
                "response_format": "b64_json",
            },
        }

        payload = None
        for key in payload_map:
            if key in self.valves.FLUX_SCHNELL_API_BASE_URL:
                payload = payload_map[key]
                headers.update(headers_map.get(key, {}))
                break

        if payload is None:
            return "Error: Unsupported API base URL! Remember, that's the beauty of open-source: you can add your own..."

        try:
            if body.get("stream", False):
                return self.stream_response(headers, payload)
            else:
                return self.non_stream_response(headers, payload)
        except requests.exceptions.RequestException as e:
            return f"Error: Request failed: {e}"
        except Exception as e:
            return f"Error: {e}"
