## vLLM + TurboQuant (systemd)

This directory (`vllm/`) is set up so you can use it as its own repository and run a service with `systemctl`, using `start_vllm.sh` as the entry point.

**Self-contained:** the TurboQuant code needed at runtime lives under `third_party/turboquant/` (vendored copy). You do **not** need a sibling checkout at `../turboquant` unless you want to override the path with `TURBOQUANT_ROOT`.

### Layout

- `start_vllm.sh`: entry point (CUDA + TurboQuant env + starts the server)
- `run_api_server_turboquant.py`: wrapper that enables TurboQuant before vLLM initializes
- `config.yaml`: local vLLM config (model/tokenizer/etc.); create from `config.example.yaml` (often gitignored when publishing)
- `third_party/turboquant/`: vendored TurboQuant used by `bootstrap.sh` and `start_vllm.sh` (`PYTHONPATH` + editable install)
- `requirements.txt`: editable TurboQuant install with the `vllm` extra from `third_party/`
- `systemd/`: systemd unit + installer

### Prerequisites

- Linux with systemd
- CUDA installed (for GPU). Default: `CUDA_HOME=/usr/local/cuda`
- Python 3
- Python dependencies (vLLM + turboquant, etc.). If you use the local venv: `vllm/venv/`

### Quick install (recommended)

1) Install Python dependencies in the interpreter that will run the service (e.g. local venv or conda):

```bash
cd vllm
python3 -m venv venv
./venv/bin/python -m pip install -U pip setuptools wheel
./venv/bin/pip install -r requirements.txt
```

(`requirements.txt` installs TurboQuant in editable mode from `third_party/turboquant[vllm]`.)

2) Alternative to step 1 — only ensure TurboQuant is available for your chosen Python:

```bash
VLLM_PYTHON=/root/anaconda3/bin/python3 ./bootstrap.sh
```

This script:

- upgrades `pip`
- exits successfully if `import turboquant` already works
- otherwise installs `turboquant` in editable mode from `third_party/turboquant` (preferred), or from `../turboquant` if present, or from `TURBOQUANT_ROOT`

### Model configuration

Create your `config.yaml` from the template and set paths:

```bash
cp config.example.yaml config.yaml
```

Then edit `config.yaml` and set at least:

- `model: /data/models/...`
- `tokenizer: /data/models/...`

### Run manually (validate before systemd)

From the `vllm/` directory:

```bash
./start_vllm.sh
```

To pick GPU / port:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_PORT=8000 ./start_vllm.sh
```

### Install as a systemd service

1) Install and enable the service:

```bash
sudo ./systemd/install_vllm_turboquant_service.sh
```

2) (Optional) Tune service environment variables:

```bash
sudo nano /etc/default/vllm-turboquant
```

Main variables:

- `VLLM_ROOT`: absolute path to this repo (the installer can set this)
- `VLLM_PYTHON`: Python binary (e.g. `/root/anaconda3/bin/python3` or `.../venv/bin/python`)
- `VLLM_HOST`, `VLLM_PORT`
- `CUDA_VISIBLE_DEVICES`, `CUDA_HOME`
- `TQ_KEY_BITS`, `TQ_VALUE_BITS`, `TQ_BUFFER_SIZE`, `TQ_INITIAL_LAYERS_COUNT`

3) Restart after changes:

```bash
sudo systemctl restart vllm-turboquant
```

### Useful commands

- Status:

```bash
systemctl status vllm-turboquant
```

- Logs:

```bash
journalctl -u vllm-turboquant -f
```

- Stop / start:

```bash
sudo systemctl stop vllm-turboquant
sudo systemctl start vllm-turboquant
```

### Troubleshooting

- **Python / packages not found:** set `VLLM_PYTHON=/path/to/venv/bin/python` in `/etc/default/vllm-turboquant`
- **`import turboquant` fails:** run `./bootstrap.sh` or `pip install -r requirements.txt` using the **same** interpreter as `VLLM_PYTHON`
- **CUDA errors:** fix `CUDA_HOME` and verify `nvcc` / libraries on the host
- **Port in use:** change `VLLM_PORT`
