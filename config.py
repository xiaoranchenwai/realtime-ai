import os
from typing import Any, Dict

from dotenv import load_dotenv
from loguru import logger

from utils.security import mask_sensitive

# Load environment variables
load_dotenv()


class Config:
    """Application configuration settings"""

    # Service provider selection
    ASR_PROVIDER = os.getenv("ASR_PROVIDER", "azure")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    TTS_PROVIDER = os.getenv("TTS_PROVIDER", "cosyvoice")

    # ASR settings
    ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "en-US")
    VOICE_ENERGY_THRESHOLD = float(os.getenv("VOICE_ENERGY_THRESHOLD", "0.05"))

    # Azure Speech service
    AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
    AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
    AZURE_TTS_VOICE = os.getenv("AZURE_TTS_VOICE", "en-US-AriaNeural")

    # FunASR settings
    FUNASR_MODEL = os.getenv("FUNASR_MODEL", "paraformer-zh-streaming")
    FUNASR_CHUNK_SIZE = [
        int(value) for value in os.getenv("FUNASR_CHUNK_SIZE", "0,10,5").split(",") if value.strip()
    ]
    FUNASR_ENCODER_CHUNK_LOOK_BACK = int(os.getenv("FUNASR_ENCODER_CHUNK_LOOK_BACK", "4"))
    FUNASR_DECODER_CHUNK_LOOK_BACK = int(os.getenv("FUNASR_DECODER_CHUNK_LOOK_BACK", "1"))

    # MiniMax TTS
    MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
    MINIMAX_VOICE_ID = os.getenv("MINIMAX_VOICE_ID", "male-qn-qingse")

    # CosyVoice3 TTS
    COSYVOICE_MODEL_DIR = os.getenv("COSYVOICE_MODEL_DIR", "pretrained_models/Fun-CosyVoice3-0.5B")
    COSYVOICE_PROMPT_TEXT = os.getenv(
        "COSYVOICE_PROMPT_TEXT",
        "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
    )
    COSYVOICE_PROMPT_WAV = os.getenv("COSYVOICE_PROMPT_WAV", "./asset/zero_shot_prompt.wav")
    COSYVOICE_INFERENCE_MODE = os.getenv("COSYVOICE_INFERENCE_MODE", "zero_shot")
    COSYVOICE_INSTRUCT_PROMPT = os.getenv("COSYVOICE_INSTRUCT_PROMPT", "You are a helpful assistant.<|endofprompt|>")

    # OpenAI API
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    OPENAI_SYSTEM_PROMPT = os.getenv(
        "OPENAI_SYSTEM_PROMPT",
        "You are an intelligent voice assistant. Please provide concise, conversational answers.",
    )

    # WebSocket settings
    WEBSOCKET_PING_INTERVAL = int(os.getenv("WEBSOCKET_PING_INTERVAL", "30"))

    # Session settings
    SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "600"))

    # Debug settings
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    @classmethod
    def _validate_provider_config(cls) -> Dict[str, bool]:
        """Validate provider-specific configurations"""
        validation_results: Dict[str, bool] = {}

        # Azure Speech validation (ASR/TTS)
        azure_credentials_valid = bool(cls.AZURE_SPEECH_KEY and cls.AZURE_SPEECH_REGION)
        validation_results["azure"] = azure_credentials_valid

        # OpenAI validation
        openai_valid = bool(cls.OPENAI_API_KEY)
        validation_results["openai"] = openai_valid

        # MiniMax validation
        minimax_valid = bool(cls.MINIMAX_API_KEY)
        validation_results["minimax"] = minimax_valid

        validation_results["funasr"] = True
        validation_results["cosyvoice"] = bool(cls.COSYVOICE_MODEL_DIR and cls.COSYVOICE_PROMPT_WAV)

        return validation_results

    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration"""
        validation_errors = []
        provider_validations = cls._validate_provider_config()

        # Check ASR provider configuration
        if cls.ASR_PROVIDER == "azure" and not provider_validations["azure"]:
            validation_errors.append(
                "Azure Speech credentials missing: AZURE_SPEECH_KEY and AZURE_SPEECH_REGION required for ASR"
            )
        elif cls.ASR_PROVIDER == "funasr" and not provider_validations["funasr"]:
            validation_errors.append("FunASR configuration invalid")

        # Check LLM provider configuration
        if cls.LLM_PROVIDER == "openai" and not provider_validations["openai"]:
            validation_errors.append("OpenAI API key missing: OPENAI_API_KEY required for LLM")

        # Check TTS provider configuration
        if cls.TTS_PROVIDER == "azure" and not provider_validations["azure"]:
            validation_errors.append(
                "Azure Speech credentials missing: AZURE_SPEECH_KEY and AZURE_SPEECH_REGION required for TTS"
            )
        elif cls.TTS_PROVIDER == "minimax" and not provider_validations["minimax"]:
            validation_errors.append("MiniMax API key missing: MINIMAX_API_KEY required for TTS")
        elif cls.TTS_PROVIDER == "cosyvoice" and not provider_validations["cosyvoice"]:
            validation_errors.append("CosyVoice configuration missing: COSYVOICE_MODEL_DIR and COSYVOICE_PROMPT_WAV")

        # Report validation errors
        if validation_errors:
            for error in validation_errors:
                logger.error(error)
            return False

        logger.info("Configuration validated successfully")
        return True

    @classmethod
    def get_service_config(cls, service_type: str) -> Dict[str, Any]:
        """Get provider-specific configuration for a service type"""
        config = {"provider": getattr(cls, f"{service_type.upper()}_PROVIDER")}

        if service_type.upper() == "ASR":
            config.update(
                {
                    "language": cls.ASR_LANGUAGE,
                    "energy_threshold": cls.VOICE_ENERGY_THRESHOLD,
                }
            )

            if config["provider"] == "azure":
                config.update(
                    {
                        "speech_key": cls.AZURE_SPEECH_KEY,
                        "speech_region": cls.AZURE_SPEECH_REGION,
                    }
                )
            elif config["provider"] == "funasr":
                config.update(
                    {
                        "model": cls.FUNASR_MODEL,
                        "chunk_size": cls.FUNASR_CHUNK_SIZE,
                        "encoder_chunk_look_back": cls.FUNASR_ENCODER_CHUNK_LOOK_BACK,
                        "decoder_chunk_look_back": cls.FUNASR_DECODER_CHUNK_LOOK_BACK,
                    }
                )

        elif service_type.upper() == "LLM":
            if config["provider"] == "openai":
                config.update(
                    {
                        "api_key": cls.OPENAI_API_KEY,
                        "base_url": cls.OPENAI_BASE_URL,
                        "model": cls.OPENAI_MODEL,
                        "system_prompt": cls.OPENAI_SYSTEM_PROMPT,
                    }
                )

        elif service_type.upper() == "TTS":
            if config["provider"] == "azure":
                config.update(
                    {
                        "speech_key": cls.AZURE_SPEECH_KEY,
                        "speech_region": cls.AZURE_SPEECH_REGION,
                        "voice": cls.AZURE_TTS_VOICE,
                    }
                )
            elif config["provider"] == "minimax":
                config.update(
                    {
                        "api_key": cls.MINIMAX_API_KEY,
                        "voice_id": cls.MINIMAX_VOICE_ID,
                    }
                )
            elif config["provider"] == "cosyvoice":
                config.update(
                    {
                        "model_dir": cls.COSYVOICE_MODEL_DIR,
                        "prompt_text": cls.COSYVOICE_PROMPT_TEXT,
                        "prompt_wav": cls.COSYVOICE_PROMPT_WAV,
                        "inference_mode": cls.COSYVOICE_INFERENCE_MODE,
                        "instruct_prompt": cls.COSYVOICE_INSTRUCT_PROMPT,
                    }
                )

        return config

    @classmethod
    def get_service_config_masked(cls, service_type: str) -> Dict[str, str]:
        """Get provider-specific configuration with sensitive data masked (for logging)"""
        config = cls.get_service_config(service_type)
        return mask_sensitive(config)


# Validate configuration
Config.validate()
