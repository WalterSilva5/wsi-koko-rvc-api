from enum import Enum

class RedisChannelEnum(Enum):
    """Enum for Redis channels."""
    LOAD_MODEL_CHANNEL = "load_model_channel"
    UNLOAD_MODEL_CHANNEL = "unload_model_channel"
    RELOAD_ALL_MODELS_CHANNEL = "reload_all_models_channel"