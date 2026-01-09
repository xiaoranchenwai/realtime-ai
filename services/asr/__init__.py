from typing import Optional

from loguru import logger

from config import Config
from services.asr.azure_asr import AzureASRService
from services.asr.base import BaseASRService
from services.asr.funasr_asr import FunASRService


def create_asr_service() -> Optional[BaseASRService]:
    """创建ASR服务实例

    Returns:
        ASR服务实例，如果创建失败则返回None
    """
    try:
        if Config.ASR_PROVIDER == "azure":
            logger.info("创建Azure ASR服务")
            if Config.AZURE_SPEECH_KEY is None or Config.AZURE_SPEECH_REGION is None:
                logger.error("Azure ASR配置缺失")
                return None
            return AzureASRService(
                subscription_key=Config.AZURE_SPEECH_KEY,
                region=Config.AZURE_SPEECH_REGION,
                language=Config.ASR_LANGUAGE,
            )
        if Config.ASR_PROVIDER == "funasr":
            logger.info("创建FunASR服务")
            return FunASRService(
                model_name=Config.FUNASR_MODEL,
                chunk_size=Config.FUNASR_CHUNK_SIZE,
                encoder_chunk_look_back=Config.FUNASR_ENCODER_CHUNK_LOOK_BACK,
                decoder_chunk_look_back=Config.FUNASR_DECODER_CHUNK_LOOK_BACK,
                language=Config.ASR_LANGUAGE,
            )
        else:
            logger.error(f"不支持的ASR提供商: {Config.ASR_PROVIDER}")
            return None
    except Exception as e:
        logger.error(f"ASR服务创建失败: {e}")
        return None
