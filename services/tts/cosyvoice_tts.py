import asyncio
import threading
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import torch
import torchaudio
from fastapi import WebSocket
from funasr import AutoModel
from loguru import logger

from services.tts.base import BaseTTSService


class CosyVoiceTTSService(BaseTTSService):
    """CosyVoice3 TTS服务实现"""

    active_tasks: Set[asyncio.Task] = set()
    _model: Optional[AutoModel] = None
    _model_lock = threading.Lock()

    def __init__(
        self,
        model_dir: str,
        prompt_text: str,
        prompt_wav: str,
        inference_mode: str = "zero_shot",
        instruct_prompt: str = "You are a helpful assistant.<|endofprompt|>",
    ) -> None:
        super().__init__()
        self.model_dir = model_dir
        self.prompt_text = prompt_text
        self.prompt_wav = prompt_wav
        self.inference_mode = inference_mode
        self.instruct_prompt = instruct_prompt
        self.send_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.send_task: Optional[asyncio.Task[None]] = None
        self.is_processing = False

        logger.info(f"CosyVoice3 TTS服务初始化: model_dir={model_dir}, mode={inference_mode}")

    @classmethod
    def get_model(cls, model_dir: str) -> AutoModel:
        with cls._model_lock:
            if cls._model is None:
                cls._model = AutoModel(model_dir=model_dir)
        return cls._model

    async def synthesize_text(self, text: str, websocket: WebSocket, is_first: bool = False) -> None:
        if not text.strip():
            logger.warning("尝试合成空文本")
            return

        logger.info(f"CosyVoice3合成文本: '{text}'")

        if not self.send_task or self.send_task.done():
            self.send_task = asyncio.create_task(self._process_send_queue(websocket))
            CosyVoiceTTSService.active_tasks.add(self.send_task)
            self.send_task.add_done_callback(CosyVoiceTTSService.active_tasks.discard)

        try:
            loop = asyncio.get_running_loop()
            audio_bytes, sample_rate = await loop.run_in_executor(None, self._generate_audio, text)

            from session import get_session

            if self.session_id is None:
                logger.error("session_id is None")
                return

            session = get_session(self.session_id)
            if session and session.is_interrupted():
                logger.info("会话已中断，跳过添加音频到队列")
                return

            item = {"audio_data": audio_bytes, "is_first": is_first, "text": text, "sample_rate": sample_rate}
            await self.send_queue.put(item)
        except Exception as e:
            logger.error(f"CosyVoice3 TTS处理错误: {e}")
            await websocket.send_json({"type": "error", "message": f"TTS错误: {str(e)}", "session_id": self.session_id})

    def _generate_audio(self, text: str) -> Tuple[bytes, int]:
        model = self.get_model(self.model_dir)
        generator = self._build_generator(model, text)
        wave, sample_rate = self._collect_audio(generator, model.sample_rate)
        pcm = self._to_pcm_16khz(wave, sample_rate)
        return pcm, 16000

    def _build_generator(self, model: AutoModel, text: str) -> Iterable[Dict[str, Any]]:
        if self.inference_mode == "cross_lingual":
            return model.inference_cross_lingual(text, self.prompt_wav, stream=False)
        if self.inference_mode == "instruct":
            return model.inference_instruct2(text, self.instruct_prompt, self.prompt_wav, stream=False)
        return model.inference_zero_shot(text, self.prompt_text, self.prompt_wav, stream=False)

    def _collect_audio(self, generator: Iterable[Dict[str, Any]], sample_rate: int) -> Tuple[torch.Tensor, int]:
        chunks = []
        for chunk in generator:
            tts_speech = chunk.get("tts_speech")
            if tts_speech is None:
                continue
            if isinstance(tts_speech, torch.Tensor):
                tensor = tts_speech
            else:
                tensor = torch.tensor(tts_speech)
            chunks.append(tensor)

        if not chunks:
            return torch.zeros(0, dtype=torch.float32), sample_rate

        audio = torch.cat(chunks, dim=-1)
        return audio, sample_rate

    def _to_pcm_16khz(self, audio: torch.Tensor, sample_rate: int) -> bytes:
        if audio.dim() > 1:
            audio = audio.squeeze(0)
        audio = audio.to(torch.float32)
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(audio, sample_rate, 16000)
        audio = audio.clamp(-1.0, 1.0)
        pcm = (audio * 32767.0).to(torch.int16).numpy().tobytes()
        return pcm

    async def _process_send_queue(self, websocket: WebSocket) -> None:
        self.is_processing = True
        try:
            while True:
                item = await self.send_queue.get()
                audio_data = item["audio_data"]
                is_first = item["is_first"]
                text = item["text"]

                from session import get_session

                if self.session_id is None:
                    logger.error("session_id is None")
                    self.send_queue.task_done()
                    continue

                session = get_session(self.session_id)
                if session and session.is_interrupted():
                    logger.info(f"会话已中断，跳过音频发送: {text[:30]}...")
                    self.send_queue.task_done()
                    continue

                session.is_tts_active = True
                try:
                    await websocket.send_json(
                        {
                            "type": "tts_start",
                            "format": "raw-16khz-16bit-mono-pcm",
                            "is_first": is_first,
                            "text": text,
                            "session_id": self.session_id,
                        }
                    )
                    await websocket.send_bytes(audio_data)
                    await websocket.send_json({"type": "tts_end", "session_id": self.session_id})
                except Exception as e:
                    logger.error(f"发送音频数据错误: {e}")
                finally:
                    session.is_tts_active = False
                    self.send_queue.task_done()
        except asyncio.CancelledError:
            logger.info("CosyVoice3 TTS发送队列任务被取消")
        except Exception as e:
            logger.error(f"CosyVoice3 TTS发送队列处理异常: {e}")
        finally:
            self.is_processing = False

    async def interrupt(self) -> bool:
        interrupted = False
        items_to_clear = self.send_queue.qsize()
        for _ in range(items_to_clear):
            try:
                self.send_queue.get_nowait()
                self.send_queue.task_done()
                interrupted = True
            except asyncio.QueueEmpty:
                break

        if self.send_task and not self.send_task.done():
            self.send_task.cancel()
            interrupted = True
            try:
                await self.send_task
            except asyncio.CancelledError:
                pass

        return interrupted

    async def close(self) -> None:
        await self.interrupt()

    @classmethod
    async def close_all(cls) -> None:
        for task in list(cls.active_tasks):
            if not task.done():
                task.cancel()
        if cls.active_tasks:
            await asyncio.gather(*cls.active_tasks, return_exceptions=True)
