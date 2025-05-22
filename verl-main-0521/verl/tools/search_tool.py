# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
import os
import time
from typing import Any, Optional, Tuple
from uuid import uuid4

import aiohttp
import requests
from verl.utils.reward_score import gsm8k

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SearchTool(BaseTool):
    """A demo tool for search.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "search",
                "description": "Searches the web for relevant information based on the given query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_list": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "A list of fully-formed semantic queries. The tool will return search results for each query."
                        }
                    },
                    "required": ["query_list"]
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
        """
        if instance_id is None:
            return str(uuid4())
        else:
            return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute the tool.

        Args:
            instance_id: The instance id of the tool.
            parameters: The json string of the parameters of the tool.

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        url = "http://127.0.0.1:8862/tool_call"
        payload = {"name": "search", "arguments": parameters}
        func_args_str = f'<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>'
        input_data = {
            "func_args_str": func_args_str,
            "level": 0
        }

        debug_save_path = os.path.join(os.getcwd(), "debug", "tool_call")

        for step in range(10):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=input_data, timeout=80) as resp:
                        response_json = await resp.json()
                resp_text = str(response_json.get("response", "search server timeout"))
                break
            except Exception as e:
                os.makedirs(debug_save_path, exist_ok=True)

                files = os.listdir(debug_save_path)
                if len(files) > 1000:
                    for fn in files:
                        os.remove(os.path.join(debug_save_path, fn))

                ts = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                log_file = os.path.join(debug_save_path, f"{len(files)}_{ts}.txt")
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write("--- func_args_str ---\n")
                    f.write(func_args_str + "\n")
                    f.write("--- exception ---\n")
                    f.write(str(e))

                print(f"[AgentTool] search tool step {step} failed: {e}. Retrying in 10s...")
                await asyncio.sleep(10)
                resp_text = '{"result": "search server request failed"}'

        return resp_text, 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
        
