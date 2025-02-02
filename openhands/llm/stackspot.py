import copy
import time
from typing import Any, Callable

import requests
from litellm import Message as LiteLLMMessage
from litellm import ModelResponse
from litellm.utils import Messages

from openhands.core.config import LLMConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message
from openhands.llm.debug_mixin import DebugMixin
from openhands.llm.metrics import Metrics
from openhands.llm.retry_mixin import RetryMixin


class StackSpotLLM(RetryMixin, DebugMixin):
    """Implementação da integração com a StackSpot AI."""

    def __init__(
        self,
        config: LLMConfig,
        metrics: Metrics | None = None,
        retry_listener: Callable[[int, int], None] | None = None,
    ):
        """Inicializa a integração com a StackSpot AI.

        Args:
            config: Configuração do LLM
            metrics: Métricas para monitoramento
            retry_listener: Listener para tentativas de retry
        """
        self._tried_model_info = False
        self.metrics = (
            metrics if metrics is not None else Metrics(model_name=config.model)
        )
        self.cost_metric_supported = True
        self.config = copy.deepcopy(config)
        self.retry_listener = retry_listener

        # URL base da API da StackSpot
        self.base_url = self.config.base_url or "https://api.stackspot.com"

        if not self.config.api_key:
            raise ValueError("API key is required for StackSpot AI integration")

    def completion(
        self, messages: list[dict[str, Any]] | dict[str, Any], **kwargs
    ) -> ModelResponse:
        """Realiza uma chamada de completion para a StackSpot AI.

        Args:
            messages: Lista de mensagens ou mensagem única
            **kwargs: Argumentos adicionais

        Returns:
            ModelResponse: Resposta do modelo
        """
        # Garante que messages seja uma lista
        if not isinstance(messages, list):
            messages = [messages]

        # Prepara o payload para a API da StackSpot
        payload = {
            "messages": messages,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
        }

        # Adiciona parâmetros opcionais se fornecidos
        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p

        headers = {
            "Authorization": f"Bearer {self.config.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

        start_time = time.time()

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.config.timeout,
            )
            response.raise_for_status()

            # Calcula e registra a latência
            latency = time.time() - start_time
            response_data = response.json()
            response_id = response_data.get("id", "unknown")
            self.metrics.add_response_latency(latency, response_id)

            # Converte a resposta para o formato ModelResponse
            model_response = ModelResponse(
                id=response_data["id"],
                choices=[
                    {
                        "message": {
                            "role": "assistant",
                            "content": response_data["choices"][0]["message"][
                                "content"
                            ],
                        },
                        "finish_reason": response_data["choices"][0].get(
                            "finish_reason", "stop"
                        ),
                    }
                ],
                model=self.config.model,
                usage=response_data.get("usage", {}),
                created=int(time.time()),
            )

            return model_response

        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to StackSpot AI: {str(e)}")
            raise

    def get_token_count(self, messages: list[dict] | list[Message]) -> int:
        """Retorna a contagem de tokens para as mensagens.

        Args:
            messages: Lista de mensagens

        Returns:
            int: Número de tokens
        """
        # Implementar lógica de contagem de tokens específica para StackSpot AI
        # Por enquanto, retorna uma estimativa básica
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        return total_chars // 4  # Estimativa básica de tokens

    def format_messages_for_llm(self, messages: Message | list[Message]) -> list[dict]:
        """Formata as mensagens para o formato esperado pela StackSpot AI.

        Args:
            messages: Mensagem única ou lista de mensagens

        Returns:
            list[dict]: Lista de mensagens formatadas
        """
        if not isinstance(messages, list):
            messages = [messages]

        formatted_messages = []
        for message in messages:
            if isinstance(message, Message):
                formatted_message = {
                    "role": message.role,
                    "content": message.content,
                }
                formatted_messages.append(formatted_message)
            else:
                formatted_messages.append(message)

        return formatted_messages

    def reset(self) -> None:
        """Reseta o estado do LLM."""
        pass  # Não há estado para resetar nesta implementação

    def __str__(self) -> str:
        return f"StackSpotLLM(model={self.config.model})"

    def __repr__(self) -> str:
        return self.__str__()
