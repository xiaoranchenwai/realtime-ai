import asyncio
import threading
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

import numpy as np
from funasr import AutoModel
from loguru import logger

from services.asr.base import BaseASRService


class FunASRService(BaseASRService):
    """FunASR streaming ASR service implementation."""

    def __init__(
        self,
        model_name: str = "paraformer-zh-streaming",
        chunk_size: Optional[List[int]] = None,
        encoder_chunk_look_back: int = 4,
        decoder_chunk_look_back: int = 1,
        language: str = "zh-CN",
    ) -> None:
        super().__init__(language)
        self.model_name = model_name
        self.chunk_size = chunk_size or [0, 10, 5]
        self.encoder_chunk_look_back = encoder_chunk_look_back
        self.decoder_chunk_look_back = decoder_chunk_look_back
        self.chunk_stride = self.chunk_size[1] * 960
        self.model = AutoModel(model=self.model_name)
        self.cache: Dict[str, Any] = {}
        self.audio_queue: Queue[bytes] = Queue()
        self.processing_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.audio_buffer = np.array([], dtype=np.float32)

    def setup_handlers(self) -> None:
        """FunASR does not require event handlers."""

    def feed_audio(self, audio_chunk: bytes) -> None:
        if not audio_chunk:
            logger.debug("收到空音频块")
            return

        if not self.is_recognizing:
            logger.debug("ASR未启动，忽略音频块")
            return

        self.audio_queue.put(audio_chunk)

    async def start_recognition(self) -> None:
        if self.is_recognizing:
            logger.warning("语音识别已经在运行中")
            return

        logger.info("开始FunASR连续语音识别")
        self.is_recognizing = True
        self.cache = {}
        self.audio_buffer = np.array([], dtype=np.float32)
        self.stop_event.clear()

        self.processing_thread = threading.Thread(target=self._process_audio_stream)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        await self.send_status("listening")

    async def stop_recognition(self) -> None:
        if not self.is_recognizing:
            logger.warning("语音识别未运行")
            return

        logger.info("停止FunASR连续语音识别")
        self.stop_event.set()

        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)

        self.is_recognizing = False
        await self.send_status("stopped")

    def _process_audio_stream(self) -> None:
        while not self.stop_event.is_set() or not self.audio_queue.empty():
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
            except Empty:
                continue

            self._append_audio_chunk(audio_chunk)
            self._process_available_chunks(is_final=False)
            self.audio_queue.task_done()

        if self.audio_buffer.size > 0:
            self._process_available_chunks(is_final=True)

    def _append_audio_chunk(self, audio_chunk: bytes) -> None:
        pcm_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        if pcm_data.size == 0:
            return
        self.audio_buffer = np.concatenate((self.audio_buffer, pcm_data))

    def _process_available_chunks(self, is_final: bool) -> None:
        while self.audio_buffer.size >= self.chunk_stride or (is_final and self.audio_buffer.size > 0):
            if self.audio_buffer.size >= self.chunk_stride:
                chunk = self.audio_buffer[: self.chunk_stride]
                self.audio_buffer = self.audio_buffer[self.chunk_stride :]
                chunk_is_final = False
            else:
                chunk = self.audio_buffer
                self.audio_buffer = np.array([], dtype=np.float32)
                chunk_is_final = is_final

            result = self.model.generate(
                input=chunk,
                cache=self.cache,
                is_final=chunk_is_final,
                chunk_size=self.chunk_size,
                encoder_chunk_look_back=self.encoder_chunk_look_back,
                decoder_chunk_look_back=self.decoder_chunk_look_back,
            )
            self._handle_result(result, chunk_is_final)

    def _extract_text(self, result: Any) -> str:
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                return str(first.get("text", ""))
            if isinstance(first, str):
                return first
        if isinstance(result, dict):
            return str(result.get("text", ""))
        if isinstance(result, str):
            return result
        return ""

    def _handle_result(self, result: Any, is_final: bool) -> None:
        text = self._extract_text(result)
        if not text.strip() or not self.websocket or not self.loop:
            return

        if is_final:

            async def send_final() -> None:
                await self.send_final_transcript(text)
                await self.process_final_transcript(text)

            asyncio.run_coroutine_threadsafe(send_final(), self.loop)
            self.last_partial_result = ""
        else:
            self.last_partial_result = text

            async def send_partial() -> None:
                await self.send_partial_transcript(text)

            asyncio.run_coroutine_threadsafe(send_partial(), self.loop)
