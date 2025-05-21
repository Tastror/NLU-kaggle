# NLU kaggle

## prerequisite

we use `uv` as the toolchain for easier usage

install uv locally

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

create `.venv` with a specified python version

```bash
# re-open the terminal
uv venv --python 3.13
```

use cuda-12.4 as an example (you can go to the official website of pytorch for other versions)

```bash
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install protobuf modelscope transformers tiktoken einops transformers_stream_generator accelerate
```

## run

```bash
uv run infer.py deepseek
```

deepseek / qwen / mistral
