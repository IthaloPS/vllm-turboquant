## vLLM + TurboQuant (systemd)

Esse diretório (`vllm/`) foi organizado pra você conseguir criar um repositório só dele e subir um serviço com `systemctl` usando o script `start_vllm.sh`.

**Autonomia:** o TurboQuant necessário ao runtime vem em `third_party/turboquant/` (cópia empacotada). Você não precisa ter outro repositório ao lado (`../turboquant`), a menos que queira sobrescrever com `TURBOQUANT_ROOT`.

### Estrutura

- `start_vllm.sh`: ponto de entrada (configura CUDA + TurboQuant + sobe o server)
- `run_api_server_turboquant.py`: wrapper que ativa TurboQuant antes do vLLM inicializar
- `config.yaml`: config do vLLM (model/tokenizer/etc.)
- `third_party/turboquant/`: cópia do TurboQuant usada pelo `bootstrap.sh` e pelo `start_vllm.sh` (`PYTHONPATH` + instalação editável)
- `requirements.txt`: instala TurboQuant (editável) com extra `vllm` a partir de `third_party/`
- `systemd/`: unit do systemd + instalador

### Pré-requisitos

- Linux com systemd
- CUDA instalado (se for usar GPU). Por padrão `CUDA_HOME=/usr/local/cuda`
- Python 3
- Dependências Python instaladas (vLLM + turboquant etc.). Se você usa o venv local: `vllm/venv/`

### Instalação rápida (recomendado)

1) Instale as dependências Python no interpretador que vai rodar o serviço (ex.: venv local ou conda):

```bash
cd vllm
python3 -m venv venv
./venv/bin/python -m pip install -U pip setuptools wheel
./venv/bin/pip install -r requirements.txt
```

(O `requirements.txt` instala o TurboQuant em modo editável a partir de `third_party/turboquant[vllm]`.)

2) Alternativa ao passo 1 — só garantir o TurboQuant no Python escolhido:

```bash
VLLM_PYTHON=/root/anaconda3/bin/python3 ./bootstrap.sh
```

Esse script:
- atualiza pip
- se `import turboquant` já funcionar, encerra com sucesso
- caso contrário, instala `turboquant` em modo editable a partir de `third_party/turboquant` (preferência), ou de `../turboquant` se existir, ou do caminho em `TURBOQUANT_ROOT`

### Configurar o modelo

Crie seu `config.yaml` a partir do template e ajuste os caminhos:

```bash
cp config.example.yaml config.yaml
```

Depois edite `config.yaml` e ajuste:

- `model: /data/models/...`
- `tokenizer: /data/models/...`

### Subir “na mão” (pra validar antes do systemd)

No diretório `vllm/`:

```bash
./start_vllm.sh
```

Se quiser escolher GPU/porta:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_PORT=8000 ./start_vllm.sh
```

### Instalar como serviço (systemd)

1) Instale/ative o serviço:

```bash
sudo ./systemd/install_vllm_turboquant_service.sh
```

2) (Opcional) Ajuste variáveis do serviço:

```bash
sudo nano /etc/default/vllm-turboquant
```

Principais variáveis:

- `VLLM_ROOT`: caminho absoluto onde você clonou esse repo (o instalador já preenche)
- `VLLM_PYTHON`: caminho do python (ex: `/root/anaconda3/bin/python3` ou `.../venv/bin/python`)
- `VLLM_HOST`, `VLLM_PORT`
- `CUDA_VISIBLE_DEVICES`, `CUDA_HOME`
- `TQ_KEY_BITS`, `TQ_VALUE_BITS`, `TQ_BUFFER_SIZE`, `TQ_INITIAL_LAYERS_COUNT`

3) Restart depois de mudanças:

```bash
sudo systemctl restart vllm-turboquant
```

### Comandos úteis

- Status:

```bash
systemctl status vllm-turboquant
```

- Logs:

```bash
journalctl -u vllm-turboquant -f
```

- Parar/iniciar:

```bash
sudo systemctl stop vllm-turboquant
sudo systemctl start vllm-turboquant
```

### Troubleshooting rápido

- Se der “não acha python/pacotes”: set `VLLM_PYTHON=/caminho/do/venv/bin/python` em `/etc/default/vllm-turboquant`
- Se `import turboquant` falhar: rode `./bootstrap.sh` ou `pip install -r requirements.txt` com o **mesmo** Python de `VLLM_PYTHON`
- Se der erro de CUDA: ajuste `CUDA_HOME` e confirme `nvcc`/libs no host
- Se a porta já estiver em uso: mude `VLLM_PORT`
