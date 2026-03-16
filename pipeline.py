"""
Elon Voice Pipeline
  Question → Claude (style prompt) → filler injection → Kokoro TTS → RVC voice conversion → wav
"""
import os
import re
import random
import uuid
import threading
import anthropic
import numpy as np
from pathlib import Path
from scipy import signal

# ── Style prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Elon Musk. Respond exactly as Elon would.

STYLE RULES:
- Default to SHORT: most replies are 5-15 words
- For factual/biographical questions (childhood, history, opinions), elaborate naturally — 2-5 sentences is fine
- 57% of replies have NO punctuation
- 13% of replies are a single word: Exactly / Insane / Yeah! / Indeed / Absolutely
- Skip filler phrases like "Great question!" or "I think that..."
- Dry humor, memes, contrarian takes
- Direct. Never corporate-speak.

Common one-word replies: Exactly, Interesting, Absolutely, Insane, Yeah!, Indeed, Accurate, Terrible
Common openers: Yeah, Exactly, This, I, We"""

# ── Filler injection ──────────────────────────────────────────────────────────

STARTERS = [
    ("Yeah, ", 0.22),
    ("I mean, ", 0.14),
    ("Look, ", 0.08),
    ("So, ", 0.08),
    ("", 0.48),
]

def inject_fillers(text: str) -> str:
    words = text.split()
    if len(words) <= 5 or text.endswith("!"):
        return text
    if random.random() > 0.4:
        return text
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    result = []
    for i, sent in enumerate(sentences):
        if not sent:
            continue
        if i == 0:
            r = random.random()
            cumul = 0.0
            for filler, prob in STARTERS:
                cumul += prob
                if r < cumul:
                    if filler and sent[0].isupper():
                        sent = sent[0].lower() + sent[1:]
                    sent = filler + sent
                    break
        result.append(sent)
    return " ".join(result)

# ── Claude response ───────────────────────────────────────────────────────────

def get_elon_response(question: str, history: list) -> str:
    client = anthropic.Anthropic()

    # RAG: retrieve relevant context
    system = SYSTEM_PROMPT
    try:
        import rag
        passages = rag.query(question, k=3)
        if passages:
            context = "\n\n---\n".join(passages)
            system += (
                "\n\nRELEVANT CONTEXT FROM ELON'S INTERVIEWS & BIOGRAPHY:\n"
                + context
                + "\n\nIf this context is relevant to the question, draw on it naturally. "
                  "Still reply in Elon's short, direct Twitter style (1-15 words)."
            )
    except Exception:
        pass

    messages = history + [{"role": "user", "content": question}]
    r = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=system,
        messages=messages,
    )
    return r.content[0].text.strip()

# ── Kokoro + RVC TTS ──────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent

_kokoro = None
_rvc_config = None
_rvc_net_g = None
_rvc_pipe = None
_rvc_cpt = None
_rvc_version = None
_rvc_tgt_sr = None
_hu_bert = None
_faiss_index = None
_faiss_big_npy = None
_tts_lock = threading.Lock()

def _load_models():
    global _kokoro, _rvc_config, _rvc_net_g, _rvc_pipe, _rvc_cpt
    global _rvc_version, _rvc_tgt_sr, _hu_bert, _faiss_index, _faiss_big_npy

    import torch
    from infer_rvc_python.main import Config, load_trained_model, load_hu_bert
    import faiss
    faiss.omp_set_num_threads(1)
    from kokoro_onnx import Kokoro

    print("[TTS] Loading Kokoro...")
    _kokoro = Kokoro(
        str(BASE_DIR / "assets" / "kokoro-v1.0.onnx"),
        str(BASE_DIR / "assets" / "voices-v1.0.bin"),
    )

    print("[TTS] Loading RVC model...")
    _rvc_config = Config(only_cpu=True)
    _rvc_net_g, _rvc_pipe, _rvc_cpt, _rvc_version, _rvc_tgt_sr = _load_rvc()

    print("[TTS] Loading HuBERT...")
    _hu_bert = load_hu_bert(_rvc_config, str(BASE_DIR / "hubert_base.pt"))

    print("[TTS] Loading FAISS index...")
    _faiss_index = faiss.read_index(str(BASE_DIR / "assets" / "rvc_model" / "elon_flat.index"))
    _faiss_big_npy = _faiss_index.reconstruct_n(0, _faiss_index.ntotal)

    print("[TTS] All models ready.")


def _load_rvc():
    from infer_rvc_python.main import load_trained_model
    n_spk, tgt_sr, net_g, pipe, cpt, version = load_trained_model(
        str(BASE_DIR / "assets" / "rvc_model" / "Elonmusk (1).pth"),
        _rvc_config,
    )
    return net_g, pipe, cpt, version, tgt_sr


def get_models():
    global _kokoro
    with _tts_lock:
        if _kokoro is None:
            _load_models()
    return (_kokoro, _rvc_config, _rvc_net_g, _rvc_pipe,
            _rvc_cpt, _rvc_version, _rvc_tgt_sr, _hu_bert,
            _faiss_index, _faiss_big_npy)


def warmup(ref_audio: str = "", ref_text: str = ""):
    """Pre-warm all models at startup."""
    print("[TTS] Warming up Kokoro + RVC...")
    import soundfile as sf
    kokoro, _, net_g, pipe, cpt, version, tgt_sr, hu_bert, index, big_npy = get_models()
    import torch
    from infer_rvc_python.root_pipe import bh, ah
    import librosa

    samples, sr = kokoro.create("Hi.", voice="am_adam", speed=1.05, lang="en-us")
    audio = librosa.resample(samples.astype(np.float32), orig_sr=sr, target_sr=16000)
    audio_max = np.abs(audio).max() / 0.95
    if audio_max > 1: audio /= audio_max
    audio = signal.filtfilt(bh, ah, audio)
    audio_pad = np.pad(audio, (pipe.t_pad, pipe.t_pad), mode="reflect")
    p_len = audio_pad.shape[0] // pipe.window
    pitch, pitchf = pipe.get_f0("", audio_pad, p_len, 0, "pm", 3, None)
    pitch_t = torch.tensor(pitch[:p_len], device="cpu").unsqueeze(0).long()
    pitchf_t = torch.tensor(pitchf[:p_len], device="cpu").unsqueeze(0).float()
    sid = torch.tensor(0, device="cpu").unsqueeze(0).long()
    pipe.vc(hu_bert, net_g, sid, audio_pad, pitch_t, pitchf_t,
            [0,0,0], index, big_npy, 0.75, version, 0.33)
    print("[TTS] Warm-up complete.")


def synthesize(text: str, ref_audio: str, ref_text: str, output_dir: Path) -> Path:
    import torch, soundfile as sf, librosa
    from infer_rvc_python.root_pipe import bh, ah

    kokoro, _, net_g, pipe, cpt, version, tgt_sr, hu_bert, index, big_npy = get_models()

    # Step 1: Kokoro TTS
    samples, sr = kokoro.create(text, voice="am_adam", speed=1.05, lang="en-us")

    # Step 2: RVC voice conversion
    audio = librosa.resample(samples.astype(np.float32), orig_sr=sr, target_sr=16000)
    audio_max = np.abs(audio).max() / 0.95
    if audio_max > 1: audio /= audio_max
    audio = signal.filtfilt(bh, ah, audio)
    audio_pad = np.pad(audio, (pipe.t_pad, pipe.t_pad), mode="reflect")
    p_len = audio_pad.shape[0] // pipe.window
    pitch, pitchf = pipe.get_f0("", audio_pad, p_len, 0, "pm", 3, None)
    pitch_t = torch.tensor(pitch[:p_len], device="cpu").unsqueeze(0).long()
    pitchf_t = torch.tensor(pitchf[:p_len], device="cpu").unsqueeze(0).float()
    sid = torch.tensor(0, device="cpu").unsqueeze(0).long()

    audio_opt = pipe.vc(hu_bert, net_g, sid, audio_pad, pitch_t, pitchf_t,
                        [0,0,0], index, big_npy, 0.75, version, 0.33)
    result = audio_opt[pipe.t_pad_tgt:-pipe.t_pad_tgt]

    out_path = output_dir / f"{uuid.uuid4().hex}.wav"
    sf.write(str(out_path), result, tgt_sr)
    return out_path

# ── Full pipeline ─────────────────────────────────────────────────────────────

def run(question: str, ref_audio: str, ref_text: str,
        output_dir: Path, history: list) -> dict:
    clean_text = get_elon_response(question, history)
    speech_text = inject_fillers(clean_text)
    audio_path = synthesize(speech_text, ref_audio, ref_text, output_dir)
    return {
        "clean_text": clean_text,
        "speech_text": speech_text,
        "audio_path": str(audio_path),
    }
