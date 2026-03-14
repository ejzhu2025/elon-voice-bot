import os
import json
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import aiofiles
import uvicorn

from contextlib import asynccontextmanager

def download_models():
    """Download model files from HF Hub if not present."""
    from huggingface_hub import hf_hub_download
    MODEL_REPO = "ejzhu2026/elon-voice-bot-models"
    BASE = Path(__file__).parent

    files = [
        ("kokoro-v1.0.onnx",           BASE / "assets" / "kokoro-v1.0.onnx"),
        ("voices-v1.0.bin",            BASE / "assets" / "voices-v1.0.bin"),
        ("rvc_model/Elonmusk.pth",     BASE / "assets" / "rvc_model" / "Elonmusk (1).pth"),
        ("rvc_model/elon_flat.index",  BASE / "assets" / "rvc_model" / "elon_flat.index"),
        ("hubert_base.pt",             BASE / "hubert_base.pt"),
        ("reference/elon_ref.wav",     BASE / "assets" / "reference" / "elon_ref.wav"),
    ]
    for repo_path, local_path in files:
        if not local_path.exists():
            print(f"[DL] Downloading {repo_path}...")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=MODEL_REPO, filename=repo_path,
                local_dir=str(BASE), local_dir_use_symlinks=False,
            )
            # hf_hub_download saves to local_dir/filename, move to expected path
            downloaded = BASE / repo_path
            if downloaded.exists() and downloaded != local_path:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                downloaded.rename(local_path)
            print(f"[DL]   -> {local_path}")
        else:
            print(f"[DL] {local_path.name} already exists, skipping.")

@asynccontextmanager
async def lifespan(app):
    # Models are pre-loaded synchronously at import time (see bottom of file)
    yield

app = FastAPI(lifespan=lifespan)

BASE_DIR    = Path(__file__).parent
OUTPUT_DIR  = BASE_DIR / "assets" / "output"
REF_DIR     = BASE_DIR / "assets" / "reference"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REF_DIR.mkdir(parents=True, exist_ok=True)

# In-memory chat history per session (simple, no DB needed)
sessions: dict[str, list] = {}

# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Elon Voice Bot</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f0f0f; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }

  header { padding: 16px 24px; border-bottom: 1px solid #222; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 18px; font-weight: 600; color: #fff; }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #666; transition: background 0.3s; }
  .status-dot.ready { background: #22c55e; }
  .status-dot.loading { background: #f59e0b; animation: pulse 1s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }


  .messages { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }
  .msg { display: flex; flex-direction: column; max-width: 640px; }
  .msg.user { align-self: flex-end; align-items: flex-end; }
  .msg.bot  { align-self: flex-start; align-items: flex-start; }
  .bubble { padding: 10px 14px; border-radius: 12px; font-size: 15px; line-height: 1.5; }
  .msg.user .bubble { background: #1d4ed8; color: #fff; border-bottom-right-radius: 3px; }
  .msg.bot  .bubble { background: #1e1e1e; color: #e0e0e0; border-bottom-left-radius: 3px; }
  .speech-text { font-size: 12px; color: #555; margin-top: 4px; font-style: italic; padding: 0 4px; }
  audio { margin-top: 8px; width: 300px; height: 36px; }
  audio::-webkit-media-controls-panel { background: #1e1e1e; }

  .thinking { display: flex; gap: 4px; padding: 12px 14px; background: #1e1e1e; border-radius: 12px; }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: #555; animation: bounce 1.2s infinite; }
  .dot:nth-child(2) { animation-delay: 0.2s; }
  .dot:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce { 0%,80%,100% { transform: translateY(0); } 40% { transform: translateY(-6px); } }

  .input-bar { padding: 16px 24px; border-top: 1px solid #222; display: flex; gap: 10px; }
  .input-bar input { flex: 1; background: #1e1e1e; border: 1px solid #333; color: #e0e0e0;
                     padding: 10px 14px; border-radius: 10px; font-size: 15px; outline: none; }
  .input-bar input:focus { border-color: #555; }
  .input-bar input::placeholder { color: #444; }
  .send-btn { background: #1d4ed8; color: #fff; border: none; padding: 10px 20px;
              border-radius: 10px; cursor: pointer; font-size: 15px; font-weight: 500; }
  .send-btn:hover { background: #2563eb; }
  .send-btn:disabled { background: #333; color: #555; cursor: not-allowed; }

  .welcome { text-align: center; color: #444; font-size: 14px; margin: auto; }
  .welcome h2 { font-size: 28px; margin-bottom: 8px; color: #666; }
</style>
</head>
<body>
<header>
  <div class="status-dot" id="statusDot"></div>
  <h1>🎙 Elon Voice Bot</h1>
  <span style="color:#22c55e;font-size:13px;margin-left:auto" id="statusText">Ready</span>
</header>


<div class="messages" id="messages">
  <div class="welcome">
    <h2>Ask Elon Anything</h2>
    <p>Upload a reference audio clip of Elon's voice, then start chatting</p>
  </div>
</div>

<div class="input-bar">
  <input type="text" id="questionInput" placeholder="Ask Elon something..."
         onkeydown="if(event.key==='Enter') sendMessage()">
  <button class="send-btn" id="sendBtn" onclick="sendMessage()">Send</button>
</div>

<script>
let sessionId = Math.random().toString(36).slice(2);

window.onload = () => {
  document.getElementById('statusDot').className = 'status-dot ready';
};

function addMsg(role, text, speechText, audioUrl) {
  const container = document.getElementById('messages');
  const welcome = container.querySelector('.welcome');
  if (welcome) welcome.remove();

  const div = document.createElement('div');
  div.className = 'msg ' + role;

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  div.appendChild(bubble);

  if (role === 'bot' && speechText && speechText !== text) {
    const st = document.createElement('div');
    st.className = 'speech-text';
    st.textContent = '🗣 "' + speechText + '"';
    div.appendChild(st);
  }

  if (audioUrl) {
    const audio = document.createElement('audio');
    audio.controls = true;
    audio.autoplay = true;
    audio.src = audioUrl;
    div.appendChild(audio);
  }

  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

function addThinking() {
  const container = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'msg bot';
  div.id = 'thinking';
  div.innerHTML = '<div class="thinking"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function removeThinking() {
  const t = document.getElementById('thinking');
  if (t) t.remove();
}

async function sendMessage() {
  const input = document.getElementById('questionInput');
  const btn = document.getElementById('sendBtn');
  const q = input.value.trim();
  if (!q) return;

  input.value = '';
  btn.disabled = true;
  document.getElementById('statusDot').className = 'status-dot loading';
  document.getElementById('statusText').textContent = 'Generating...';

  addMsg('user', q);
  addThinking();

  try {
    const r = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ question: q, session_id: sessionId })
    });
    const d = await r.json();
    removeThinking();
    if (d.error) {
      addMsg('bot', '(error: ' + d.error + ')', null, null);
    } else {
      addMsg('bot', d.clean_text, d.speech_text, d.audio_url);
    }
  } catch(e) {
    removeThinking();
    addMsg('bot', '(error: ' + e.message + ')', null, null);
  }

  btn.disabled = false;
  document.getElementById('statusDot').className = 'status-dot ready';
  document.getElementById('statusText').textContent = 'Ready';
  input.focus();
}
</script>
</body>
</html>
"""

# ── State ─────────────────────────────────────────────────────────────────────

_default_ref = str(REF_DIR / "elon_ref.wav")
_default_ref_text = (
    "Well, if one looks to say chimpanzee society, it is not friendly. "
    "I mean, the bonobos are an exception. But chimpanzee society is full of violence."
)
ref_audio_path = _default_ref if (REF_DIR / "elon_ref.wav").exists() else ""
ref_text_content = _default_ref_text if (REF_DIR / "elon_ref.wav").exists() else ""

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML

@app.post("/upload_ref")
async def upload_ref(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix
    dest = REF_DIR / f"reference{ext}"
    async with aiofiles.open(dest, "wb") as f:
        await f.write(await file.read())
    global ref_audio_path
    ref_audio_path = str(dest)
    return {"ok": True, "path": str(dest)}

@app.post("/set_ref_text")
async def set_ref_text(payload: dict):
    global ref_text_content
    ref_text_content = payload.get("text", "")
    return {"ok": True}

@app.post("/chat")
async def chat(payload: dict):
    question   = payload.get("question", "").strip()
    session_id = payload.get("session_id", "default")

    if not question:
        raise HTTPException(400, "Empty question")
    if not ref_audio_path:
        raise HTTPException(400, "No reference audio uploaded")

    history = sessions.get(session_id, [])

    import pipeline
    pipeline.os.environ.setdefault("ANTHROPIC_API_KEY",
                                    os.environ.get("ANTHROPIC_API_KEY", ""))

    try:
        result = pipeline.run(
            question=question,
            ref_audio=ref_audio_path,
            ref_text=ref_text_content,
            output_dir=OUTPUT_DIR,
            history=history,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

    # Update history
    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": result["clean_text"]})
    sessions[session_id] = history[-20:]  # keep last 10 turns

    audio_filename = Path(result["audio_path"]).name
    return JSONResponse({
        "clean_text":  result["clean_text"],
        "speech_text": result["speech_text"],
        "audio_url":   f"/audio/{audio_filename}",
    })

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Audio not found")
    return FileResponse(str(path), media_type="audio/wav")

# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("Error: set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)
    # Download models from HF Hub if missing
    download_models()
    # Reload ref audio path after potential download
    _ref = BASE_DIR / "assets" / "reference" / "elon_ref.wav"
    if _ref.exists() and not ref_audio_path:
        ref_audio_path = str(_ref)
        ref_text_content = _default_ref_text
    # Pre-load TTS models before uvicorn starts
    import pipeline as pl
    pl.warmup()
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting Elon Voice Bot at http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
