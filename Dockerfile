FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg libsndfile1 git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first (smaller image)
RUN pip install --no-cache-dir \
    torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# Install fairseq fork (Python 3.11 compatible)
RUN pip install --no-cache-dir \
    "fairseq @ git+https://github.com/One-sixth/fairseq.git" \
    bitarray sacrebleu

# Install app dependencies
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] anthropic python-multipart aiofiles \
    kokoro-onnx soundfile librosa scipy numpy \
    "faiss-cpu==1.7.4" huggingface_hub

# Install infer-rvc-python without deps (deps already installed)
RUN pip install --no-cache-dir infer-rvc-python --no-deps
RUN pip install --no-cache-dir ffmpeg-python

# Pin pydantic to avoid by_alias=None bug
RUN pip install --no-cache-dir "pydantic==2.9.2" "pydantic-core==2.23.4"

# Patch fairseq: weights_only=False
RUN python -c "\
import fairseq, os; \
f = os.path.join(os.path.dirname(fairseq.__file__), 'checkpoint_utils.py'); \
c = open(f).read().replace( \
    'state = torch.load(f, map_location=torch.device(\"cpu\"))', \
    'state = torch.load(f, map_location=torch.device(\"cpu\"), weights_only=False)'); \
open(f,'w').write(c); print('fairseq patched')"

# Patch infer-rvc-python: weights_only=False
RUN python -c "\
import site, os; sp = site.getsitepackages()[0]; \
f = os.path.join(sp, 'infer_rvc_python/main.py'); \
c = open(f).read().replace( \
    'torch.load(model_path, map_location=\"cpu\")', \
    'torch.load(model_path, map_location=\"cpu\", weights_only=False)'); \
open(f,'w').write(c); print('infer_rvc_python patched')"

# Patch infer-rvc-python rmvpe: weights_only=False
RUN python -c "\
import site, os; sp = site.getsitepackages()[0]; \
f = os.path.join(sp, 'infer_rvc_python/lib/rmvpe.py'); \
c = open(f).read().replace( \
    'torch.load(model_path, map_location=\"cpu\")', \
    'torch.load(model_path, map_location=\"cpu\", weights_only=False)'); \
open(f,'w').write(c); print('rmvpe patched')"

COPY main.py pipeline.py ./

EXPOSE 7860

CMD ["python", "main.py"]
