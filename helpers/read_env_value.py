import os
import dotenv

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_env_path = os.path.join(_project_root, ".env")
dotenv.load_dotenv(_env_path)


def read_env_value(key: str, default, cast=None):
    if cast is None:
        cast = type(default) if default is not None else str
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return cast(default)
    try:
        return cast(raw)
    except (ValueError, TypeError):
        return cast(default)
