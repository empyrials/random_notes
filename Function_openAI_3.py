from pydantic import BaseModel, Field
from typing import Optional, Union, Generator, Iterator
from utils.misc import get_last_user_message

import os
import requests


class Pipe:
    class Valves(BaseModel):
        OPENAI_API_BASE_URL: str = Field(
            default="https://api.openai.com/v1",
            description="The base URL for OpenAI API endpoints.",
        )
        OPENAI_API_KEY: str = Field(
            default="",
            description="Required API key to retrieve the model list.",
        )
        MODEL_FILTER: str = Field(
            default="gpt-4o,dall-e",
            description="Only models starting with these prefixes will show (comma-separated).",
        )
        NAME_PREFIX: str = Field(
            default="OPENAI/",
            description="The prefix applied before the model names.",
        )

    class UserValves(BaseModel):
        OPENAI_API_KEY: str = Field(
            default="",
            description="User-specific API key for accessing OpenAI services.",
        )

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()

    def pipes(self):
        if self.valves.OPENAI_API_KEY:
            try:
                headers = {
                    "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                }

                response = requests.get(
                    f"{self.valves.OPENAI_API_BASE_URL}/models", headers=headers
                )
                response.raise_for_status()
                models = response.json()

                # Split the MODEL_FILTER into individual prefixes
                filter_prefixes = self.valves.MODEL_FILTER.split(",")

                filtered_models = [
                    {
                        "id": model["id"],
                        "name": f'{self.valves.NAME_PREFIX}{model["name"] if "name" in model else model["id"]}',
                    }
                    for model in models["data"]
                    if any(model["id"].startswith(prefix) for prefix in filter_prefixes)
                ]

                return filtered_models

            except Exception as e:
                print(f"Error: {e}")
                return [
                    {
                        "id": "error",
                        "name": "Could not fetch models from OpenAI, please update the API Key in the valves.",
                    },
                ]
        else:
            return [
                {
                    "id": "error",
                    "name": "Global API Key not provided.",
                },
            ]

    def pipe(
        self, body: dict, __user__: dict
    ) -> Union[str, Generator[str, None, None], Iterator[str]]:
        print(f"pipe:{__name__}")
        print(__user__)

        if not self.valves.OPENAI_API_KEY:
            raise Exception("OPENAI_API_KEY not provided by the user.")

        headers = {
            "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        # Remove the model prefix before sending the request
        model_id = body["model"][body["model"].find(".") + 1 :]
        payload = {**body, "model": model_id}
        print(payload)

        try:
            response = requests.post(
                url=f"{self.valves.OPENAI_API_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                stream=True,  # Enables streaming
            )

            response.raise_for_status()

            if body.get("stream", False):
                # Generate a streaming response by yielding each line of the response
                return self._streaming_response(response)
            else:
                return response.json()

        except Exception as e:
            return f"Error: {e}"

    def _streaming_response(
        self, response: requests.Response
    ) -> Generator[str, None, None]:
        for line in response.iter_lines():
            if line:
                # Decode and yield lines as strings
                decoded_line = line.decode("utf-8")
                yield decoded_line  # You can also yield a more structured response, e.g., json.loads(decoded_line)
