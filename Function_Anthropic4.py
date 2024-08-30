import os
import requests
import json
import base64
import logging
from typing import List, Union, Generator, Iterator, Optional, Tuple
from pydantic import BaseModel, Field

# Constants
MAX_IMAGES_PER_CALL = 5
MAX_TOTAL_IMAGE_SIZE = 100 * 1024 * 1024  # 100 MB
API_VERSION = "2023-06-01"


class Pipe:
    class Valves(BaseModel):
        ANTHROPIC_API_KEY: Optional[str] = Field(
            default="",
            description="Your Anthropic API Key.",
        )
        MAX_TOTAL_TOKENS: Optional[int] = Field(
            default="4000",
            description="The MAX input tokens sent per request.",
        )

    def __init__(self):
        self.type = "manifold"
        self.id = "anthropic"
        self.name = "anthropic/"
        self.logger = logging.getLogger(__name__)

        self.valves = self.Valves(
            **{"ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", "")}
        )

        if not self.valves.ANTHROPIC_API_KEY:
            self.logger.warning(
                "ANTHROPIC_API_KEY is not set. Please set it in your environment variables before making API calls."
            )

    def get_anthropic_models(self) -> List[dict]:
        return [
            {"id": "claude-3-haiku-20240307", "name": "claude-3-haiku"},
            {"id": "claude-3-opus-20240229", "name": "claude-3-opus"},
            {"id": "claude-3-sonnet-20240229", "name": "claude-3-sonnet"},
            {"id": "claude-3-5-sonnet-20240620", "name": "claude-3.5-sonnet"},
        ]

    def pipes(self) -> List[dict]:
        return self.get_anthropic_models()

    def process_image(self, image_data: dict) -> dict:
        if image_data["image_url"]["url"].startswith("data:image"):
            mime_type, base64_data = image_data["image_url"]["url"].split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                },
            }
        else:
            return {
                "type": "image",
                "source": {"type": "url", "url": image_data["image_url"]["url"]},
            }

    def extract_system_message(self, messages: List[dict]) -> Tuple[Optional[str], List[dict]]:
        system_message = None
        other_messages = []
        for message in messages:
            if message["role"] == "system":
                system_message = message["content"]
            else:
                other_messages.append(message)
        return system_message, other_messages

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        try:
            if not self.valves.ANTHROPIC_API_KEY:
                raise ValueError(
                    "ANTHROPIC_API_KEY is not set. Please set it in your environment variables before making API calls."
                )

            system_message, messages = self.extract_system_message(body["messages"])

            processed_messages = []
            image_count = 0
            total_image_size = 0

            for message in messages:
                processed_content = []
                if isinstance(message.get("content"), list):
                    for item in message["content"]:
                        if item["type"] == "text":
                            processed_content.append(
                                {"type": "text", "text": item["text"]}
                            )
                        elif item["type"] == "image_url":
                            if image_count >= MAX_IMAGES_PER_CALL:
                                raise ValueError(
                                    f"Maximum of {MAX_IMAGES_PER_CALL} images per API call exceeded"
                                )

                            processed_image = self.process_image(item)
                            processed_content.append(processed_image)

                            if processed_image["source"]["type"] == "base64":
                                image_size = (
                                    len(processed_image["source"]["data"]) * 3 / 4
                                )
                                if (
                                    image_size
                                    > MAX_TOTAL_IMAGE_SIZE / MAX_IMAGES_PER_CALL
                                ):
                                    raise ValueError(
                                        f"Image size exceeds maximum allowed size per image"
                                    )
                            else:
                                image_size = 0

                            total_image_size += image_size
                            if total_image_size > MAX_TOTAL_IMAGE_SIZE:
                                raise ValueError(
                                    f"Total size of images exceeds {MAX_TOTAL_IMAGE_SIZE / (1024 * 1024)} MB limit"
                                )

                            image_count += 1
                else:
                    processed_content = [
                        {"type": "text", "text": message.get("content", "")}
                    ]

                processed_messages.append(
                    {"role": message["role"], "content": processed_content}
                )

            payload = {
                "model": body["model"][body["model"].find(".") + 1 :],
                "messages": processed_messages,
                "max_tokens": body.get("max_tokens", int(self.valves.MAX_TOTAL_TOKENS)),
                "temperature": body.get("temperature", 0.8),
                "top_k": body.get("top_k", 40),
                "top_p": body.get("top_p", 0.9),
                "stop_sequences": body.get("stop", []),
                **({"system": system_message} if system_message else {}),
                "stream": body.get("stream", False),
            }

            headers = {
                "x-api-key": self.valves.ANTHROPIC_API_KEY,
                "anthropic-version": API_VERSION,
                "content-type": "application/json",
            }

            url = "https://api.anthropic.com/v1/messages"

            if body.get("stream", False):
                return self.stream_response(url, headers, payload)
            else:
                return self.non_stream_response(url, headers, payload)

        except requests.RequestException as e:
            self.logger.error(f"Request error: {e}")
            return f"Error: {e}"
        except ValueError as e:
            self.logger.error(f"Value error: {e}")
            return f"Error: {e}"
        except Exception as e:
            self.logger.error(f"Unexpected error in pipe method: {e}")
            return f"Unexpected error: {e}"

    def stream_response(self, url: str, headers: dict, payload: dict ) -> Generator[str, None, None]:
        with requests.post(url, headers=headers, json=payload, stream=True) as response:
            if response.status_code != 200:
                raise Exception(f"HTTP Error {response.status_code}: {response.text}")

            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if data["type"] == "content_block_start":
                                yield data["content_block"]["text"]
                            elif data["type"] == "content_block_delta":
                                yield data["delta"]["text"]
                            elif data["type"] == "message_stop":
                                break
                            elif data["type"] == "message":
                                for content in data.get("content", []):
                                    if content["type"] == "text":
                                        yield content["text"]
                        except json.JSONDecodeError:
                            self.logger.error(f"Failed to parse JSON: {line}")
                        except KeyError as e:
                            self.logger.error(f"Unexpected data structure: {e}")
                            self.logger.error(f"Full data: {data}")

    def non_stream_response(self, url: str, headers: dict, payload: dict) -> str:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"HTTP Error {response.status_code}: {response.text}")

        res = response.json()
        return res["content"][0]["text"] if "content" in res and res["content"] else ""
