# WSI RVC
Repositório minimal para integrar o modelo OpenVoiceV2 (RVC-style) e utilitários para extração de embeddings e inferência.
## Visão geral
Este projeto fornece uma pequena estrutura (factories/managers) para carregar modelos de voz (adaptadores para OpenVoice), extrair embeddings e executar inferência.
Objetivos principais:
- Facilitar o carregamento de checkpoints do OpenVoiceV2.
- Fornecer uma API simples para extrair embeddings (`project/embedding/factory.py`) e gerenciar o modelo (`project/model/manager.py`).
## Links importantes
- Página do modelo OpenVoiceV2 (Hugging Face):
	https://huggingface.co/myshell-ai/OpenVoiceV2#openvoice-v2
- Download do checkpoint (fornecido):
	https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip
Coloque o conteúdo do arquivo descompactado em uma pasta que terá pelo menos `config.json` e `model.pth` (ou nomes equivalentes conforme o checkpoint). Ex.: `models/openvoice_v2/`.
## Instalação
Este projeto usa Python 3.12+ (verifique sua versão). Instale dependências com pip:
```bash
python -m pip install -r requirements.txt
# ou, se preferir o arquivo alternativo:
python -m pip install -r requirements_new.txt
```
Notas:
- Se for usar GPU, assegure que o PyTorch com suporte CUDA esteja instalado compatível com sua placa.
- Dependências relacionadas ao pacote `TTS.vc` são referenciadas no código (`TTS.vc.models.openvoice`). Assegure que o submódulo/lib necessária esteja disponível no ambiente (pode vir de um pacote externo ou do diretório `rvc-cli/rvc`).
## Preparar o checkpoint do OpenVoiceV2
1. Baixe e extraia o arquivo ZIP fornecido:
```bash
# exemplo
wget -O checkpoints_v2_0417.zip "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip"
unzip checkpoints_v2_0417.zip -d models/openvoice_v2
```
2. Confirme que dentro de `models/openvoice_v2` existem os arquivos de configuração (por exemplo `config.json`) e o peso do modelo (`model.pth`). Ajuste caminhos conforme necessário.
## Uso rápido (exemplo)
Exemplo mínimo para carregar o modelo usando o `ModelManager` e extrair um embedding com `EmbeddingFactory`:
```python
from project.model.manager import ModelManager
from project.embedding.factory import EmbeddingFactory

# Caminho para a pasta onde colocou config.json e model.pth
MODEL_DIR = "/path/to/models/openvoice_v2"

# Inicializa gerenciador e carrega o modelo (lazy load)
manager = ModelManager(MODEL_DIR)
model = manager.get_model()

# Extrai embedding de um arquivo wav
factory = EmbeddingFactory(model)
se = factory.create_embedding("/path/to/audio.wav")
print(se.shape)
```
Se o seu modelo suportar inferência diretamente, você pode usar os métodos do adaptador (`extract_se`, `inference`) conforme as assinaturas em `project/model/factory.py`.
## Estrutura relevante
- `project/model/factory.py` — adaptadores e factory para criar modelos de voz.
- `project/model/manager.py` — gerenciador que retorna uma instância carregada do modelo.
- `project/embedding/factory.py` — utilitário para criar embeddings a partir de arquivos WAV.
## Execução de testes
Há alguns testes em `tests/` (pytest). Execute:
```bash
python -m pytest -q
```
## Dicas e resolução de problemas
- Se ocorrerem erros relacionados a `TTS.vc` ou importações, verifique se as dependências do repositório `rvc-cli` (anexado neste workspace) estão instaladas ou disponíveis no PYTHONPATH.
- Para usar GPU, assegure que o PyTorch detecte sua GPU (`torch.cuda.is_available()`), caso contrário o carregamento seguirá em CPU.
- Erros de versão de pacote podem ser resolvidos criando um ambiente virtual e instalando as versões listadas em `requirements.txt`.

## TTS (Kokoro) — wrapper

O projeto inclui um wrapper assíncrono para a API Kokoro TTS, implementado em `project/synthesizer/tts_provider.py` e consumido por `project/synthesizer/synthesizer_service.py`.

Como funciona (visão rápida):
- `TtsProvider.synthesize(text: str, options: dict = None)` — realiza uma chamada HTTP POST assíncrona para o endpoint `/audio/speech` da API Kokoro. Lida tanto com respostas JSON quanto com respostas binárias (áudio). Retorna um dicionário com `success` e, em caso de áudio, os bytes ficam em `audio` e `content_type` descreve o formato.
- `SynthesizerService.synthesize_audio(dto: RvcTtsDTO)` — serviço de alto nível que recebe um `RvcTtsDTO` (ou `KokoroTtsDto`) e delega ao `TtsProvider`. É assíncrono e retorna os bytes do áudio sintetizado.

DTOs relevantes (arquivo `project/dto/tts_dto.py`):
- `RvcTtsDTO`: campos { voice, target_voice, text }
- `KokoroTtsDto`: campos { voice, text, response_format, download_format, speed, stream, return_download_link, lang_code, volume_multiplier }

Exemplo de uso (assíncrono):

```python
import asyncio
from project.synthesizer.synthesizer_service import SynthesizerService
from project.dto.tts_dto import RvcTtsDTO

async def main():
	svc = SynthesizerService()
	dto = RvcTtsDTO(text="Olá mundo", voice="am_adam")
	audio_bytes = await svc.synthesize_audio(dto)
	with open("out.mp3", "wb") as f:
		f.write(audio_bytes)

asyncio.run(main())
```

Notas de integração:
- O `TtsProvider` usa por padrão a URL `http://localhost:8880/v1` — ajuste o `url` no construtor se a API Kokoro estiver em outro host/porta.
- Há um `#TODO` em `tts_provider.py` para mapear vozes; atualize conforme suas vozes disponíveis (ex.: `am_adam`, `af_alloy`).
- `TtsProvider` utiliza `aiohttp` e retorna exceções `Exception` em falhas de comunicação; capture-as no código chamador se necessário.

## Licença e contribuição
Verifique `LICENSE` e `PKG-INFO` para detalhes; contribuições são bem-vindas.
---

Se quiser, posso:
- adicionar exemplos práticos de inferência (conversão de voz) usando os controllers em `project/conversor/`;
- criar um script de instalação que baixa e extrai automaticamente o checkpoint para `models/openvoice_v2`.
# WSI RVC

