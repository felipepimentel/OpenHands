# Integração StackSpot AI no OpenHands

## Visão Geral

A integração StackSpot AI no OpenHands permite utilizar os modelos de linguagem da StackSpot através de uma interface consistente com o restante do framework. Esta implementação segue os padrões do OpenHands e fornece todas as funcionalidades necessárias para interagir com a API da StackSpot.

## Instalação

A integração já está incluída no pacote principal do OpenHands. Não é necessária nenhuma instalação adicional além das dependências padrão do projeto.

## Configuração

### Requisitos Básicos

1. API Key da StackSpot
2. Nome do modelo a ser utilizado
3. Configurações opcionais de geração

### Exemplo de Configuração

```python
from openhands.core.config import LLMConfig
from openhands.llm import StackSpotLLM

config = LLMConfig(
    model="stackspot-model",           # Nome do modelo desejado
    api_key="sua-api-key",            # Sua API key da StackSpot
    base_url="https://api.stackspot.com",  # URL base (opcional)
    temperature=0.7,                   # Temperatura para geração (0.0 a 1.0)
    max_output_tokens=1000,            # Limite de tokens na saída
    timeout=30,                        # Timeout em segundos
    top_p=0.95                         # Top P (opcional)
)

llm = StackSpotLLM(config)
```

## Uso Básico

### Exemplo Simples

```python
# Criando uma mensagem simples
messages = [
    {
        "role": "user",
        "content": "Olá, como você está?"
    }
]

# Obtendo uma resposta
response = llm.completion(messages)
print(response.choices[0].message.content)
```

### Exemplo com Múltiplas Mensagens

```python
messages = [
    {
        "role": "system",
        "content": "Você é um assistente prestativo e amigável."
    },
    {
        "role": "user",
        "content": "Pode me ajudar com uma dúvida?"
    },
    {
        "role": "assistant",
        "content": "Claro! Estou aqui para ajudar. Qual é a sua dúvida?"
    },
    {
        "role": "user",
        "content": "Como faço para usar o OpenHands?"
    }
]

response = llm.completion(messages)
```

## Funcionalidades

### Completion

O método principal para interação com o modelo:

```python
response = llm.completion(messages, **kwargs)
```

#### Parâmetros:
- `messages`: Lista de mensagens ou mensagem única
- `**kwargs`: Argumentos adicionais para a chamada

#### Retorno:
- `ModelResponse`: Objeto contendo a resposta do modelo

### Formatação de Mensagens

```python
from openhands.core.message import Message

message = Message(role="user", content="Olá!")
formatted = llm.format_messages_for_llm(message)
```

### Contagem de Tokens

```python
token_count = llm.get_token_count(messages)
```

## Métricas e Monitoramento

A integração inclui suporte para métricas básicas:

```python
# Acessando métricas
latency = llm.metrics.get_average_latency()
```

## Tratamento de Erros

A integração inclui tratamento robusto de erros:

```python
try:
    response = llm.completion(messages)
except requests.exceptions.RequestException as e:
    print(f"Erro na chamada da API: {e}")
```

## Recursos Avançados

### Retry Mechanism

A integração herda do `RetryMixin`, fornecendo capacidades de retry automático:

```python
# O retry é configurado através do LLMConfig
config = LLMConfig(
    # ... outras configurações ...
    num_retries=3,
    retry_min_wait=1,
    retry_max_wait=10,
    retry_multiplier=2
)
```

### Debug Mode

Através do `DebugMixin`, é possível habilitar logs detalhados:

```python
# Habilitando debug
config = LLMConfig(
    # ... outras configurações ...
    debug=True
)
```

## Limitações Conhecidas

1. **Contagem de Tokens**: A implementação atual usa uma estimativa básica de tokens (caracteres/4)
2. **Streaming**: Não há suporte para respostas em streaming nesta versão
3. **Métricas**: Apenas métricas básicas de latência estão implementadas

## Boas Práticas

1. **Segurança**:
   - Nunca exponha sua API key no código
   - Use variáveis de ambiente ou gerenciadores de segredos

2. **Performance**:
   - Configure timeouts apropriados
   - Utilize o mecanismo de retry para maior resiliência

3. **Monitoramento**:
   - Monitore as métricas de latência
   - Configure logs apropriados para debug quando necessário

## Troubleshooting

### Problemas Comuns

1. **API Key Inválida**:
```python
ValueError: API key is required for StackSpot AI integration
```
Solução: Verifique se a API key foi configurada corretamente.

2. **Timeout na Requisição**:
```python
requests.exceptions.Timeout: Request timed out
```
Solução: Aumente o valor do timeout na configuração.

## Contribuindo

Para contribuir com melhorias na integração:

1. Fork o repositório
2. Crie uma branch para sua feature
3. Implemente suas mudanças
4. Adicione testes
5. Submeta um Pull Request

## Recursos Adicionais

- [Documentação da API StackSpot](https://ai.stackspot.com/docs)
- [OpenHands Documentation](https://docs.all-hands.dev)
- [Exemplos de Uso](https://docs.all-hands.dev/examples)

## Suporte

Para suporte:
- Abra uma issue no GitHub
- Entre no canal Slack da comunidade
- Consulte a documentação oficial 