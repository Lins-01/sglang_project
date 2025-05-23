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
        retrieval_service_url = "http://127.0.0.1:8000/retrieve"
        query_list_from_params = parameters.get("query_list")
        if not query_list_from_params or not isinstance(query_list_from_params, list):
            error_msg = "Error: 'query_list' is missing, empty, or not a list in parameters."
            logger.error(f"[SearchTool] {error_msg} Received parameters: {parameters}")
            return json.dumps({"result": error_msg}), 0.0, {}
        
        
        payload = {
            "queries": query_list_from_params,
            "topk": 1,
            "return_scores": True
        }
        
        resp_text_str = json.dumps({"result": "Search server request failed or timed out after retries."}) # 默认失败信息
        debug_save_path = os.path.join(os.getcwd(), "debug", "tool_call_searchtool")
        os.makedirs(debug_save_path, exist_ok=True)

        for step in range(10): # 最多重试10次
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(retrieval_service_url, json=payload, timeout=80) as resp:
                        response_data = await resp.json() # 直接获取JSON响应
                        if resp.status == 200:
                            # 服务端返回的已经是 {"result": ...} 格式的JSON
                            # Tool 的 execute 通常返回字符串，所以我们将服务端JSON转为字符串
                            resp_text_str = json.dumps(response_data)
                            logger.debug(f"[SearchTool] Success. Response: {resp_text_str}")
                        else:
                            error_detail = response_data.get("detail", await resp.text())
                            logger.error(f"[SearchTool] Error from retrieval server. Status: {resp.status}, Detail: {error_detail}")
                            resp_text_str = json.dumps({"result": f"Error from retrieval server: Status {resp.status} - {error_detail}"})
                        break  # 成功或有明确错误响应后跳出重试
            except aiohttp.ClientConnectorError as e: # 网络连接错误
                logger.error(f"[SearchTool] Connection error on step {step}: {e}. Retrying in 10s...")
                if step == 9: # 最后一次尝试仍然失败
                    resp_text_str = json.dumps({"result": f"Search server connection error after multiple retries: {e}"})
            except asyncio.TimeoutError: # aiohttp 的超时 (如果 ClientSession 或请求级别设置了总超时)
                 logger.error(f"[SearchTool] Timeout error on step {step}. Retrying in 10s...")
                 if step == 9:
                    resp_text_str = json.dumps({"result": "Search server request timed out after multiple retries."})
            except Exception as e: # 其他所有异常
                # 保存调试信息
                current_files = os.listdir(debug_save_path)
                if len(current_files) > 1000: # 限制日志文件数量
                    oldest_file = min([os.path.join(debug_save_path, f) for f in current_files], key=os.path.getctime, default=None)
                    if oldest_file: os.remove(oldest_file)
                
                ts = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                # 使用 instance_id 或其他唯一标识避免文件名冲突（如果并行处理）
                log_file = os.path.join(debug_save_path, f"error_{instance_id}_{ts}_{step}.txt")
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write("--- Attempted Payload to Server ---\n")
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n\n")
                    f.write(f"--- Exception (Attempt {step + 1}) ---\n")
                    f.write(str(e) + "\n")
                    


                logger.error(f"[SearchTool] Exception on step {step}: {e}. Retrying in 10s...")
                if step == 9: # 最后一次尝试仍然失败
                    resp_text_str = json.dumps({"result": f"Search server request failed after multiple retries due to: {e}"})
            
            if step < 9: # 如果不是最后一次尝试，则等待后重试
                await asyncio.sleep(10)

        return resp_text_str, 0.0, {} # tool_reward_score 和 tool_metrics 暂时为默认值

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        # del self._instance_dict[instance_id]
        pass

        
