import os

DEFAULT_YOUTUBE_CHANNELS = [
    "UCawZsQWqfGSbCI5yjkdVkTA",
]


def _channels_from_env() -> list[str]:
    raw = os.getenv("YOUTUBE_CHANNELS", "")
    channels = [c.strip() for c in raw.split(",") if c.strip()]
    return channels or DEFAULT_YOUTUBE_CHANNELS


YOUTUBE_CHANNELS = _channels_from_env()
