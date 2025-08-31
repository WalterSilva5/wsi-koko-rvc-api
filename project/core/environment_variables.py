from types import SimpleNamespace
from decouple import config


environment_variables = SimpleNamespace(
    **{
        "PORT": int(config("PORT", default="8881")),
        "MODELS_DIR_PATH": config(
            "MODELS_DIR_PATH", default="/mnt/data/wsi_vc/vc_models/"
        ),
        "SPEAKERS_DIR_PATH": config(
            "SPEAKERS_DIR_PATH", default="/mnt/data/wsi_vc/speakers/"
        ),
    }
)
