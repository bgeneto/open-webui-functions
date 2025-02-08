"""
title: FLUX 1.1 Pro Manifold Function for Black Forest Lab Image Generation Model
author: bgeneto
author_url: https://github.com/bgeneto/open-webui-functions
funding_url: https://github.com/open-webui
created: 2024/10/30
modified: 2025/02/08
version: 0.2.7
license: MIT
requirements: pydantic, aiohttp
environment_variables: REPLICATE_API_KEY, TOGETHER_API_KEY, DEEPINFRA_API_KEY
supported providers: replicate.com, together.xyz
    replicate:  https://api.replicate.com/v1/models/black-forest-labs/flux-1.1-pro/predictions
    togetherai: https://api.together.xyz/v1/images/generations
"""

import asyncio
import base64
import logging
import os
from typing import Any, Dict, Generator, List, Optional, Union

import aiohttp
from aiohttp import ClientResponseError, ClientSession
from open_webui.utils.misc import get_last_user_message
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Pipe:
    """
    Class representing the FLUX1.1 Pro Manifold Function with asynchronous support.
    """

    class Valves(BaseModel):
        """
        Pydantic model for storing API keys.
        """

        DEEPINFRA_API_KEY: str = Field(
            default="", description="Your deepinfra.com API Key"
        )
        REPLICATE_API_KEY: str = Field(
            default="", description="Your replicate.com API Key"
        )
        TOGETHER_API_KEY: str = Field(
            default="", description="Your together.ai API Key"
        )

    def __init__(self):
        """
        Initialize the Pipe class with default values and environment variables.
        """
        self.type = "manifold"
        self.id = "FLUX11_Pro"
        self.name = "FLUX1.1: "
        self.valves = self.Valves(
            DEEPINFRA_API_KEY=os.getenv("DEEPINFRA_API_KEY", ""),
            REPLICATE_API_KEY=os.getenv("REPLICATE_API_KEY", ""),
            TOGETHER_API_KEY=os.getenv("TOGETHER_API_KEY", ""),
        )

        # Define provider configurations
        self.providers = [
            {
                "name": "deepinfra.com",
                "base_url": "https://api.deepinfra.com/v1/inference/black-forest-labs/FLUX-1.1-pro",
                "api_key": self.valves.DEEPINFRA_API_KEY,
                "headers_map": {},
                "payload_map": {
                    "prompt": "",  # To be filled dynamically
                    "width": 1440,
                    "height": 1440,
                    "steps": 10,
                    "n": 1,
                },
                "response_type": "json",
            },
            {
                "name": "together.xyz",
                "base_url": "https://api.together.xyz/v1/images/generations",
                "api_key": self.valves.TOGETHER_API_KEY,
                "headers_map": {},
                "payload_map": {
                    "model": "black-forest-labs/FLUX.1.1-pro",
                    "prompt": "",  # To be filled dynamically
                    "width": 1440,
                    "height": 1440,
                    "steps": 10,
                    "n": 1,
                    "response_format": "b64_json",
                },
                "response_type": "json",
            },
            {
                "name": "replicate.com",
                "base_url": "https://api.replicate.com/v1/models/black-forest-labs/flux-1.1-pro/predictions",
                "api_key": self.valves.REPLICATE_API_KEY,
                "headers_map": {"Prefer": "wait=40"},
                "payload_map": {
                    "input": {
                        "prompt": "",  # To be filled dynamically
                        "prompt_upsampling": False,
                        "aspect_ratio": "1:1",
                        "output_format": "webp",
                        "output_quality": 100,
                        "num_inference_steps": 10,
                        "safety_tolerance": 5,
                        "height": 1440,
                        "width": 1440,
                    }
                },
                "response_type": "json",
            },
        ]

    async def stream_response(
        self,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        session: ClientSession,
        response_type: str,
    ) -> Optional[str]:
        """
        Asynchronously handle streaming responses (if applicable in the future).
        Currently returns the non-streaming response.
        Args:
            headers (Dict[str, str]): The headers for the request.
            payload (Dict[str, Any]): The payload for the request.
            session (ClientSession): The aiohttp client session.
            response_type (str): Type of response ('json' or 'image').
        Returns:
            Optional[str]: The processed response or None.
        """
        return await self.non_stream_response(headers, payload, session, response_type)

    async def image_url_to_markdown_b64(self, image_url):
        """
        Convert an image URL to base64-encoded image data.
        """
        try:
            # Download the image
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    response.raise_for_status()
                    # return the response content
                    return base64.b64encode(await response.read()).decode("utf-8")

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise

    def is_valid_b64(self, resp) -> bool:
        """
        Validate the base64-encoded data in the response.
        """
        if (
            not isinstance(resp, dict)
            or "data" not in resp
            or not isinstance(resp["data"], list)
        ):
            logger.error(
                "Invalid response structure"
            )  # Early return for invalid top-level structure
            return False

        for item in resp["data"]:
            if not isinstance(item, dict):
                return False
            if "b64_json" not in item or not isinstance(item["b64_json"], str):
                return False

            try:
                base64.b64decode(item["b64_json"])
            except BaseException:
                return False

        return True

    async def get_img_extension(self, img_data: str) -> Union[str, None]:
        """
        Get the image extension based on the base64-encoded data.
        Args:
            img_data (str): Base64-encoded image data.
        Returns:
            Union[str, None]: The image extension or None if unsupported.
        """
        header = img_data[:10]  # Increased to capture longer headers
        if header.startswith("/9j/"):
            return "jpeg"
        elif header.startswith("iVBOR"):
            return "png"
        elif header.startswith("R0lGOD"):
            return "gif"
        elif header.startswith("UklGR"):
            return "webp"
        return None

    async def handle_json_response(
        self, resp_json: Dict[str, Any], response_type: str
    ) -> str:
        """
        Handle JSON responses from the API.
        Args:
            resp_json (Dict[str, Any]): The JSON response from the API.
            response_type (str): Type of response ('json' or 'image').
        Returns:
            str: The formatted image data or an error message.
        """
        img_data = ""

        # Handle DeepInfra response format
        if "image_url" in resp_json and resp_json["status"] == "ok":
            img_data = resp_json["image_url"]
        # Handle regular DeepInfra response format
        elif "images" in resp_json and isinstance(resp_json["images"], list):
            img_data = resp_json["images"][0]
        # Handle Replicate response format
        elif "output" in resp_json:
            img_data = resp_json["output"]
        # Handle Together.ai response format
        elif self.is_valid_b64(resp_json):
            img_data = resp_json["data"][0]["b64_json"]
        else:
            logger.error("Unexpected response format for the image provider!")
            raise ValueError("Unexpected response format for the image provider!")

        if img_data.startswith(("http://", "https://")):
            img_data = await self.image_url_to_markdown_b64(img_data)

        # Split ;base64, from img_data only after converting to markdown base64
        if ";base64," in img_data:
            img_data = img_data.split(";base64,")[1]

        img_ext = await self.get_img_extension(img_data)

        if not img_ext:
            logger.error("Unsupported image format returned!")
            raise ValueError("Unsupported image format returned!")

        # Rebuild img_data with proper format
        img_data = f"data:image/{img_ext};base64,{img_data}"

        return f"![Image]({img_data})\n📸 {img_ext} image should appear above."

    async def handle_image_response(self, content: bytes, content_type: str) -> str:
        """
        Handle raw image responses from the API.
        Args:
            content (bytes): The raw image bytes.
            content_type (str): The content type of the image.
        Returns:
            str: The formatted image data.
        """
        img_ext = "png"  # Default extension
        if "image/" in content_type:
            img_ext = content_type.split("/")[-1]
        image_base64 = base64.b64encode(content).decode("utf-8")
        return f"![Image](data:{content_type};base64,{image_base64})\nGeneratedImage.{img_ext}"

    async def non_stream_response(
        self,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        session: ClientSession,
        response_type: str,
    ) -> str:
        """
        Asynchronously get a non-streaming response from the API.
        Args:
            headers (Dict[str, str]): The headers for the request.
            payload (Dict[str, Any]): The payload for the request.
            session (ClientSession): The aiohttp client session.
            response_type (str): Type of response ('json' or 'image').
        Returns:
            str: The response from the API.
        """
        try:
            async with session.post(
                url=headers.pop("url"),
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status in {200, 201}:
                    content_type = response.headers.get("Content-Type", "")
                    if "application/json" in content_type:
                        resp_json = await response.json()
                        return await self.handle_json_response(resp_json, response_type)
                    elif "image/" in content_type:
                        content = await response.read()
                        return await self.handle_image_response(content, content_type)
                    else:
                        return f"Error: Unsupported content type {content_type}"
                elif response.status in {
                    400,
                    401,
                    403,
                    404,
                    405,
                    408,
                    429,
                    500,
                    502,
                    503,
                    504,
                }:
                    raise ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=response.reason,
                        headers=response.headers,
                    )
                else:
                    return f"Error: Received unexpected status code {response.status}"
        except ClientResponseError as e:
            return f"Error: HTTP {e.status} - {e.message}"
        except asyncio.TimeoutError:
            return "Error: Request timed out."
        except aiohttp.ClientError as e:
            return f"Error: Client error occurred: {str(e)}"
        except Exception as e:
            return f"Error: An unexpected error occurred: {str(e)}"

    def pipes(self) -> List[Dict[str, str]]:
        """
        Get the list of available pipes.
        Returns:
            List[Dict[str, str]]: The list of pipes.
        """

        api_keys = [
            self.valves.DEEPINFRA_API_KEY,
            self.valves.TOGETHER_API_KEY,
            self.valves.REPLICATE_API_KEY,
        ]
        if not any(key.strip() for key in api_keys):
            return [{"id": "flux1.1_pro", "name": "Pro (API key not set!)"}]

        return [{"id": "flux1.1_pro", "name": "Pro"}]

    async def pipe(
        self, body: Dict[str, Any]
    ) -> Union[str, Generator[str, None, None], asyncio.Task]:
        """
        Asynchronously process the pipe request with provider fallback.
        Args:
            body (Dict[str, Any]): The request body.
        Returns:
            Union[str, Generator[str, None, None], asyncio.Task]: The response from the API.
        """
        headers_common = {
            "Content-Type": "application/json",
        }
        body["stream"] = False
        prompt = get_last_user_message(body["messages"])

        # Initialize aiohttp session
        async with aiohttp.ClientSession() as session:
            last_error = None
            # Iterate over providers and attempt each
            for provider in self.providers:
                try:
                    logger.error(f"Trying text-to-image provider {provider['name']}...")
                    # Prepare headers and payload
                    headers = headers_common.copy()
                    headers.update(provider["headers_map"])
                    headers["Authorization"] = f"Bearer {provider['api_key']}"

                    payload = provider["payload_map"].copy()

                    # Insert dynamic prompt
                    if provider["name"] == "replicate.com":
                        payload["input"]["prompt"] = prompt
                    elif provider["name"] == "together.xyz":
                        payload["prompt"] = prompt
                    elif provider["name"] == "deepinfra.com":
                        payload["prompt"] = prompt

                    # Add the URL to headers for reference in non_stream_response
                    headers["url"] = provider["base_url"]

                    response = await self.non_stream_response(
                        headers, payload, session, provider["response_type"]
                    )

                    if not response.startswith("Error:"):
                        # Successful response
                        return response
                        # Store the error and continue with next provider
                    last_error = response
                    logger.error(
                        f"Provider {provider['name']} failed with error: {response}"
                    )
                    continue

                except Exception as e:
                    # Catch any exception that might occur
                    error_msg = (
                        f"Provider {provider['name']} failed with exception: {str(e)}"
                    )
                    logger.error(error_msg)
                    last_error = f"Error: {str(e)}"
                    continue

            # If all providers failed, return the last error
            return f"Error: All providers failed. Last error: {last_error}"
