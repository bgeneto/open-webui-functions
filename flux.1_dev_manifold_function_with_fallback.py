"""
title: FLUX 1.1 Dev Manifold Function for Black Forest Lab Image Generation Model from Multi Providers with Fallback.
author: bgeneto
author_url: https://github.com/bgeneto/open-webui-functions
funding_url: https://github.com/open-webui
created: 2025/02/08
modified: 2025/02/08
version: 0.1.3
license: MIT
requirements: pydantic, aiohttp, gradio_client
environment_variables: HUGGINGFACE_API_KEY, DEEPINFRA_API_KEY, REPLICATE_API_KEY, TOGETHER_API_KEY
supported providers: huggingface.co, deepinfra.com, replicate.com, together.xyz
notes: huggingface.co is used via ZeroGPU.
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

        HUGGINGFACE_API_KEY: str = Field(
            default=os.getenv("HUGGINGFACE_API_KEY", ""),
            description="Your huggingface.co API Key",
        )
        DEEPINFRA_API_KEY: str = Field(
            default=os.getenv("DEEPINFRA_API_KEY", ""),
            description="Your deepinfra.com API Key",
        )
        TOGETHER_API_KEY: str = Field(
            default=os.getenv("TOGETHER_API_KEY", ""),
            description="Your together.ai API Key",
        )
        REPLICATE_API_KEY: str = Field(
            default=os.getenv("REPLICATE_API_KEY", ""),
            description="Your replicate.com API Key",
        )
        WIDTH: int = Field(default=1024, description="Min./Max. width: 256/2048")
        HEIGHT: int = Field(default=1024, description="Min./Max. height: 256/2048")
        pass

    class UserValves(BaseModel):
        WIDTH: int = Field(default=1024, description="Min./Max. width: 256/2048")
        HEIGHT: int = Field(default=1024, description="Min./Max. height: 256/2048")
        pass

    def __init__(self):
        """
        Initialize the Pipe class with default values and environment variables.
        """
        self.type = "manifold"
        self.id = "FLUX11_Dev"
        self.name = "FLUX1.1: "
        self.user_valves = self.UserValves()
        self.valves = self.Valves(
            **{
                "HUGGINGFACE_API_KEY": os.getenv("HUGGINGFACE_API_KEY", ""),
                "DEEPINFRA_API_KEY": os.getenv("DEEPINFRA_API_KEY", ""),
                "TOGETHER_API_KEY": os.getenv("TOGETHER_API_KEY", ""),
                "REPLICATE_API_KEY": os.getenv("REPLICATE_API_KEY", ""),
            }
        )

        width = int(
            self.user_valves.WIDTH if self.user_valves.WIDTH else self.valves.WIDTH
        )
        height = int(
            self.user_valves.HEIGHT if self.user_valves.HEIGHT else self.valves.HEIGHT
        )

        # Define provider configurations
        self.providers = [
            {
                "name": "huggingface.co",
                "base_url": "https://black-forest-labs-flux-1-dev.hf.space/gradio_api/call/infer",
                "api_key": self.valves.HUGGINGFACE_API_KEY,
                "headers_map": {},
                "payload_map": {
                    "data": [
                        "",  # prompt - will be filled dynamically
                        0,  # seed
                        1,  # randomize_seed
                        width,  # image_width
                        height,  # image_height - will be filled from self.valves
                        12.5,  # guidance_scale
                        25,  # num_inference_steps
                    ]
                },
                "response_type": "json",
            },
            {
                "name": "deepinfra.com",
                "base_url": "https://api.deepinfra.com/v1/inference/black-forest-labs/FLUX-1-dev",
                "api_key": self.valves.DEEPINFRA_API_KEY,
                "headers_map": {},
                "payload_map": {
                    "prompt": "",  # To be filled dynamically
                    "width": width,
                    "height": height,
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
                    "model": "black-forest-labs/FLUX.1-dev",
                    "prompt": "",  # To be filled dynamically
                    "width": width,
                    "height": height,
                    "steps": 10,
                    "n": 1,
                    "response_format": "b64_json",
                },
                "response_type": "json",
            },
            {
                "name": "replicate.com",
                "base_url": "https://api.replicate.com/v1/models/black-forest-labs/flux-dev/predictions",
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
                        "width": width,
                        "height": height,
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

    async def handle_huggingface_response(
        self,
        session: ClientSession,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
    ) -> str:
        """
        Handle Hugging Face Space API calls using Gradio client.
        """
        try:
            from gradio_client import Client
            import base64

            # Extract parameters from payload
            data = payload.get("data", [])
            if len(data) < 7:
                raise ValueError("Invalid payload data structure")

            prompt = data[0]
            seed = data[1]
            randomize_seed = data[2]
            width = data[3]
            height = data[4]
            guidance_scale = data[5]
            num_inference_steps = data[6]

            # Create Gradio client and make prediction
            client = Client(
                "black-forest-labs/FLUX.1-dev",
                hf_token=self.valves.HUGGINGFACE_API_KEY,
            )

            result = await asyncio.to_thread(
                client.predict,
                prompt,
                seed,
                randomize_seed,
                width,
                height,
                guidance_scale,
                num_inference_steps,
                api_name="/infer",
            )

            # Process the result - expecting (file_path, number) tuple
            if isinstance(result, tuple) and len(result) >= 1:
                file_path = result[0]

                # Read the image file and convert to base64
                try:
                    with open(file_path, "rb") as image_file:
                        img_data = base64.b64encode(image_file.read()).decode("utf-8")

                    # Get image extension from file path
                    img_ext = file_path.split(".")[-1].lower()
                    if img_ext in ["jpg", "jpeg", "png", "webp", "gif"]:
                        return f"![Image](data:image/{img_ext};base64,{img_data})\n📸 {img_ext} image should appear above."
                    else:
                        raise ValueError(f"Unsupported image format: {img_ext}")
                except FileNotFoundError:
                    raise ValueError(f"Image file not found at {file_path}")
                except Exception as e:
                    raise ValueError(f"Error processing image file: {str(e)}")

            raise ValueError("Invalid response format from Gradio API")

        except Exception as e:
            logger.error(f"Hugging Face API error: {str(e)}")
            raise

    async def handle_huggingface_response_aiohttp(
        self,
        session: ClientSession,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
    ) -> str:
        """
        Handle the two-step process for Hugging Face Space API calls.
        First gets event_id, then fetches result using that ID.
        """
        try:
            # Step 1: Get the event_id
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    raise ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                    )

                result = await response.json()

                if not isinstance(result, dict) or "event_id" not in result:
                    raise ValueError("Expected event_id in response")

                event_id = result["event_id"]

                # Step 2: Fetch the result using event_id
                result_url = f"{url}/{event_id}"
                max_retries = 3
                retry_delay = 2

                for retry in range(max_retries):
                    async with session.get(result_url) as result_response:
                        if result_response.status != 200:
                            if retry < max_retries - 1:
                                await asyncio.sleep(retry_delay)
                                continue
                            raise ClientResponseError(
                                request_info=result_response.request_info,
                                history=result_response.history,
                                status=result_response.status,
                            )

                        error_received = False
                        # Read the event stream content
                        async for line in result_response.content:
                            line_str = line.decode("utf-8").strip()

                            # Check for error event
                            if line_str == "event: error":
                                error_received = True
                                continue

                            if error_received and line_str.startswith("data: "):
                                error_data = line_str[6:]  # Skip 'data: ' prefix
                                raise RuntimeError(
                                    f"Hugging Face API error: {error_data}"
                                )

                            if line_str.startswith("data: "):
                                try:
                                    import json

                                    data = json.loads(
                                        line_str[6:]
                                    )  # Skip 'data: ' prefix
                                    if isinstance(data, dict) and "data" in data:
                                        for item in data["data"]:
                                            if isinstance(item, list) and len(item) > 0:
                                                img_data = item[0]
                                                if (
                                                    isinstance(img_data, str)
                                                    and ";base64," in img_data
                                                ):
                                                    img_base64 = img_data.split(
                                                        ";base64,"
                                                    )[1]
                                                    img_ext = (
                                                        await self.get_img_extension(
                                                            img_base64
                                                        )
                                                    )
                                                    if img_ext:
                                                        return f"![Image](data:image/{img_ext};base64,{img_base64})\n📸 {img_ext} image should appear above."
                                except json.JSONDecodeError as e:
                                    logger.error(f"Failed to parse JSON: {e}")
                                    continue

                        if retry < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            continue

                raise TimeoutError("Timeout waiting for image generation")

        except Exception as e:
            logger.error(f"Hugging Face API error: {str(e)}")
            raise

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
        if "images" in resp_json and isinstance(resp_json["images"], list):
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
        if any(domain in headers["url"] for domain in ["huggingface.co", "hf.space"]):
            return await self.handle_huggingface_response(
                session, headers["url"], headers, payload
            )
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
            self.valves.HUGGINGFACE_API_KEY,
            self.valves.DEEPINFRA_API_KEY,
            self.valves.TOGETHER_API_KEY,
            self.valves.REPLICATE_API_KEY,
        ]
        if not any(key.strip() for key in api_keys):
            return [{"id": "flux1.1_dev", "name": "Dev (API key not set!)"}]

        return [{"id": "flux1.1_dev", "name": "Dev"}]

    async def pipe(
        self, body: Dict[str, Any], __user__: dict
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
                    logger.error(f"Trying text-to-image provider: {provider['name']}")
                    # Prepare headers and payload
                    headers = headers_common.copy()
                    headers.update(provider["headers_map"])
                    if provider["api_key"] and provider["api_key"].strip():
                        api_key = provider["api_key"].strip()
                        headers["Authorization"] = f"Bearer {api_key}"
                    else:
                        logger.error(
                            f"Skipping provider {provider['name']}, API key not set!"
                        )
                        continue

                    payload = provider["payload_map"].copy()

                    width = int(
                        __user__["valves"].WIDTH
                        if __user__["valves"].WIDTH
                        else self.valves.WIDTH
                    )
                    height = int(
                        __user__["valves"].HEIGHT
                        if __user__["valves"].HEIGHT
                        else self.valves.HEIGHT
                    )

                    # Insert dynamic prompt
                    if provider["name"] == "replicate.com":
                        payload["input"]["prompt"] = prompt
                        payload["input"]["width"] = width
                        payload["input"]["height"] = height
                    elif provider["name"] == "together.xyz":
                        payload["prompt"] = prompt
                        payload["width"] = width
                        payload["height"] = height
                    elif provider["name"] == "deepinfra.com":
                        payload["prompt"] = prompt
                        payload["width"] = width
                        payload["height"] = height
                    elif provider["name"] == "huggingface.co":
                        payload["data"][0] = prompt
                        payload["data"][3] = width
                        payload["data"][4] = height

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
