# Instruções para GitHub Copilot - Projeto WSI RVC

## Estrutura do Projeto
```
wsi-rvc/
├── project/
│   ├── conversor/          # Lógica de conversão de voz
│   │   ├── controller.py   # Endpoints da API
│   │   ├── service.py      # Serviço de conversão
│   │   ├── core_conversion_service.py # Serviço core de conversão
│   │   ├── processor.py    # Processamento de áudio
│   │   └── wrapper/        # Wrappers para modelos
│   ├── core/              # Configurações e aplicação
    # código que pode falhar
# Instruções do repositório para o GitHub Copilot (agente)

Objetivo: fornecer regras claras e contexto para que o Copilot gere sugestões e edições consistentes com a arquitetura, padrões e expectativas do projeto WSI RVC.

Regras gerais de comportamento
- Priorize mudanças pequenas e reversíveis (pequenos commits/patches).
- Sempre favor clareza e legibilidade: nomes descritivos, docstrings e type hints.
- Não execute comandos no terminal, não faça chamadas de rede externas.
- Não crie ou exfiltre segredos, tokens ou dados sensíveis.

Guia de estilo e padrões esperados
- Tipos e anotações: use type hints em funções públicas e retornos.
- Nomenclatura: classes em PascalCase; funções/variáveis em snake_case; constantes em UPPER_CASE.
- Arquitetura: respeite os patterns existentes (Factory, Manager, Service, DTO).
- Isolamento: mantenha a lógica de I/O (arquivos, rede) separada da lógica pura.

Como propor mudanças de código
- Ao sugerir uma alteração, produza a menor mudança possível que resolva o problema.
- Se modificar comportamento público, atualize/adicione um teste unitário em `tests/` cobrindo o caso.
- Ao criar novos arquivos, adicione explicações curtas no topo (docstring) e exporte/registre no lugar apropriado (factory/manager) se necessário.

Testes e verificação
- Sempre adicione ou atualizar um teste unitário quando alterar lógica relevante.
- Prefira testes pequenos e determinísticos; use `pytest` e `pytest-asyncio` para rotinas assíncronas.

Interações com modelos e grandes dependências
- Evite carregar modelos pesados em sugestões de snippet; use fakes/mocks nos testes.
- Quando necessário, ofereça uma implementação que use injeção de dependência (p.ex. passar um manager/factory) para facilitar testes.

Erros e logs
- Use o logger central (`logging_config.py`) para mensagens de erro ou info relevantes.
- Retorne erros claros com exceções e mensagens curtas; para handlers HTTP prefira `HTTPException` do FastAPI.

Limitações e proibições
- Não modifique arquivos fora do escopo do bug/feature sem motivo claro.
- Não execute ou sugera execução de scripts que possam alterar o ambiente do usuário.
- Não adicione credenciais, senhas ou URLs privadas ao repositório.

Contexto rápido do código (apenas o essencial)
- A API principal está em `app.py` e `main.py` expõe a aplicação.
- Conversão de voz: `project/conversor/` (controller, service, processor, wrapper).
- Model adapters e factory: `project/model/factory.py`, `project/model/manager.py`.
- Extração de embeddings: `project/embedding/`.

Se não houver informação suficiente
- Faça uma sugestão conservadora e peça ao autor mais contexto no PR (pequeno comentário explicando suposições).

Formato de sugestões
- Forneça patches pequenos aplicáveis (diffs) ou snippets prontos para copiar.
- Inclua um breve resumo (1-2 linhas) do que o patch faz e por que é seguro.

FIM
## Dependências Principais
