"""
title: FLUX.1 Schnell Manifold Function for Black Forest Lab Image Generation Model from several providers.
author: bgeneto
author_url: https://github.com/bgeneto/open-webui-flux-image-gen
funding_url: https://github.com/open-webui
modified: 2025-02-08
version: 0.3.5
license: MIT
requirements: pydantic, aiohttp
environment_variables: TOGETHER_API_KEY, HUGGINGFACE_API_KEY, REPLICATE_API_KEY
supported providers: huggingface.co, replicate.com, together.xyz
providers urls:
https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell
https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions
https://api.together.xyz/v1/images/generations
"""

import base64
import logging
import os
from typing import Any, Dict, Generator, List, Optional, Union

import aiohttp
from open_webui.utils.misc import get_last_user_message
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Provider:
    def __init__(
        self, name: str, url: str, headers: Dict[str, str], payload: Dict[str, Any]
    ):
        self.name = name
        self.url = url
        self.headers = headers
        self.payload = payload


class Pipe:
    """
    Class representing the FLUX.1 Schnell Manifold Function.
    """

    class Valves(BaseModel):
        """
        Pydantic model for storing API keys and base URLs.
        """

        TOGETHER_API_KEY: str = Field(
            default="", description="Your API Key for Together.xyz"
        )
        HUGGINGFACE_API_KEY: str = Field(
            default="", description="Your API Key for Huggingface.co"
        )
        REPLICATE_API_KEY: str = Field(
            default="", description="Your API Key for Replicate.com"
        )

    def __init__(self):
        """
        Initialize the Pipe class with default values and environment variables.
        """
        self.type = "manifold"
        self.id = "FLUX11_Schnell"
        self.name = "FLUX1.1: "
        self.valves = self.Valves(
            TOGETHER_API_KEY=os.getenv("TOGETHER_API_KEY", ""),
            HUGGINGFACE_API_KEY=os.getenv("HUGGINGFACE_API_KEY", ""),
            REPLICATE_API_KEY=os.getenv("REPLICATE_API_KEY", ""),
        )

        self.providers = [
            Provider(
                name="together.xyz",
                url="https://api.together.xyz/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {self.valves.TOGETHER_API_KEY}",
                    "Content-Type": "application/json",
                },
                payload={
                    "model": "black-forest-labs/FLUX.1-schnell-Free",
                    "width": 1024,
                    "height": 1024,
                    "steps": 4,
                    "n": 1,
                    "response_format": "b64_json",
                },
            ),
            Provider(
                name="huggingface.co",
                url="https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell",
                headers={
                    "Authorization": f"Bearer {self.valves.HUGGINGFACE_API_KEY}",
                    "Content-Type": "application/json",
                    "x-wait-for-model": "true",
                },
                payload={},
            ),
            Provider(
                name="replicate.com",
                url="https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions",
                headers={
                    "Authorization": f"Bearer {self.valves.REPLICATE_API_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "wait",
                },
                payload={
                    "input": {
                        "num_outputs": 1,
                        "aspect_ratio": "1:1",
                        "output_quality": 100,
                        "output_format": "webp",
                        "num_inference_steps": 4,
                        "disable_safety_checker": True,
                    }
                },
            ),
        ]

    async def url_to_img_data(self, API_KEY: str, url: str) -> str:
        """
        Convert a URL to base64-encoded image data.

        Args:
            API_KEY (str): The API key for authorization.
            url (str): The URL of the image.

        Returns:
            str: Base64-encoded image data.
        """
        headers = {"Authorization": f"Bearer {API_KEY}"}
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=headers, timeout=20) as response:
                    response.raise_for_status()
                    content_type = response.headers.get(
                        "Content-Type", "application/octet-stream"
                    )
                    encoded_content = base64.b64encode(await response.read()).decode(
                        "utf-8"
                    )
                    return f"data:{content_type};base64,{encoded_content}"
            except aiohttp.ClientError as e:
                logger.error(f"URL to Image Data conversion failed: {e}")
                raise

    async def non_stream_response(self, provider: Provider) -> str:
        """
        Get a non-streaming response from the API with enhanced error handling.

        Args:
            provider (Provider): The API provider to use.

        Returns:
            str: The formatted image data.

        Raises:
            aiohttp.ClientError: If a network-related error occurs.
            ValueError: If the response format is unexpected or unsupported.
        """
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url=provider.url,
                    headers=provider.headers,
                    json=provider.payload,
                    timeout=20,
                ) as response:
                    await self.verify_auth_response(response, provider)
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "")
                    if "application/json" in content_type:
                        return await self.handle_json_response(response)
                    elif "image/" in content_type:
                        return await self.handle_image_response(response)
                    else:
                        logger.error(
                            f"Unsupported content type from {provider.name}: {content_type}"
                        )
                        raise ValueError(f"Unsupported content type {content_type}")
            except aiohttp.ClientError as e:
                logger.error(f"Request failed for {provider.name}: {str(e)}")
                raise

    async def verify_auth_response(
        self, response: aiohttp.ClientResponse, provider: Provider
    ) -> None:
        """Verify authentication response and raise exception if unauthorized"""
        if response.status == 401:
            logger.warning(f"Authentication failed for {provider.name}")
            raise aiohttp.ClientResponseError(
                request_info=response.request_info,
                history=response.history,
                status=response.status,
                message=f"Authentication failed for {provider.name}",
            )

    async def stream_response(self, provider: Provider) -> Generator[str, None, None]:
        """
        Get a streaming response from the API.

        Args:
            provider (Provider): The API provider to use.

        Yields:
            Generator[str]: The formatted image data.
        """
        try:
            response = await self.non_stream_response(provider)
        except Exception as e:
            logger.error(f"Error with {provider.name}: {str(e)}")
            raise
        yield response

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

    def get_img_extension(self, img_data: str) -> Optional[str]:
        """
        Get the image extension based on the base64-encoded data.

        Args:
            img_data (str): Base64-encoded image data.

        Returns:
            Optional[str]: The image extension or None if unsupported.
        """
        img_header = img_data[:10]

        if img_header.startswith("/9j/"):
            return "jpeg"
        elif img_header.startswith("iVBOR"):
            return "png"
        elif img_header.startswith("R0lG"):
            return "gif"
        elif img_header.startswith("UklGR"):
            return "webp"

        return None

    async def handle_json_response(self, response: aiohttp.ClientResponse) -> str:
        """
        Handle JSON response from the API.

        Args:
            response (aiohttp.ClientResponse): The response object.

        Returns:
            str: The formatted image data.

        Raises:
            ValueError: If the response format is unexpected or unsupported.
        """
        resp = await response.json()

        img_data = ""

        if "output" in resp:
            img_data = resp["output"][0]
        elif self.is_valid_b64(resp):
            img_data = resp["data"][0]["b64_json"]
        else:
            logger.error("Unexpected response format for the image provider!")
            raise ValueError("Unexpected response format for the image provider!")

        if img_data.startswith(("http://", "https://")):
            img_data = await self.image_url_to_markdown_b64(img_data)

        # Split ;base64, from img_data only after converting to markdown base64 (see above)
        if ";base64," in img_data:
            img_data = img_data.split(";base64,")[1]

        img_ext = self.get_img_extension(img_data)
        if not img_ext:
            logger.error("Unsupported image format returned!")
            raise ValueError("Unsupported image format returned!")

        # Rebuild img_data with proper format
        img_data = f"data:image/{img_ext};base64,{img_data}"

        return f"![Image]({img_data})\n📸 {img_ext} image generated above."

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

    async def handle_image_response(self, response: aiohttp.ClientResponse) -> str:
        """
        Handle image response from the API.

        Args:
            response (aiohttp.ClientResponse): The response object.

        Returns:
            str: The formatted image data.
        """
        content_type = response.headers.get("Content-Type", "")
        img_ext = content_type.split("/")[-1] if "image/" in content_type else None
        image_base64 = base64.b64encode(await response.read()).decode("utf-8")
        img_ext = self.get_img_extension(image_base64) or "png"
        return f"![Image](data:{content_type};base64,{image_base64})\n📸 {img_ext} image generated above."

    def provider_prompt_payload(
        self, provider: Provider, payload: Dict[str, Any], prompt: str
    ) -> Dict[str, Any]:
        """
        Set the payload prompt for the given provider.

        Args:
            provider (Provider): The provider object.
            prompt (str): The prompt to use.

        Returns:
            Dict[str, Any]: The provider prompt.
        """
        if provider.name == "together.xyz":
            payload["prompt"] = prompt
        elif provider.name == "huggingface.co":
            payload["inputs"] = prompt
        elif provider.name == "replicate.com":
            provider.payload["input"]["prompt"] = prompt

        return payload

    def pipes(self) -> List[Dict[str, str]]:
        """
        Get the list of available pipes.

        Returns:
            List[Dict[str, str]]: The list of pipes.
        """
        api_keys = [
            self.valves.TOGETHER_API_KEY,
            self.valves.REPLICATE_API_KEY,
            self.valves.HUGGINGFACE_API_KEY,
        ]
        if not any(key.strip() for key in api_keys):
            return [{"id": "flux1.1_dev", "name": "Schnell (API key not set!)"}]

        return [{"id": "flux1.1_schnell", "name": "Schnell"}]

    async def pipe(
        self, body: Dict[str, Any]
    ) -> Union[str, Generator[str, None, None], List[Dict[str, str]]]:
        """
        Main method to process the pipe request.

        Args:
            body (Dict[str, Any]): The request body containing messages and options.

        Returns:
            Union[str, Generator[str, None, None], List[Dict[str, str]]]: The API response or pipes list.
        """
        if "id" in body and body["id"] == "flux_schnell":
            return self.pipes()

        prompt = get_last_user_message(body.get("messages", []))
        if not prompt:
            logger.error("No prompt found in the request body.")
            return "Error: No prompt provided."

        last_error = None
        for provider in self.providers:
            provider.payload = self.provider_prompt_payload(
                provider, provider.payload, prompt
            )
            logger.info(f"Attempting to generate image using {provider.name}...")

            try:
                return await anext(self.stream_response(provider))

            except aiohttp.ClientResponseError as e:
                last_error = e
                logger.warning(
                    f"Provider {provider.name} failed with {str(e)}, trying next provider..."
                )
                continue
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error with {provider.name}: {str(e)}")
                continue

        error_msg = f"Failed to generate image from all providers."
        logger.error(error_msg + f" Last error: {str(last_error)}")
        return f"Error: {error_msg}"
