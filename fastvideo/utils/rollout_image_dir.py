"""Rollout decode PNG scratch directory (shared by training and test scripts)."""

import os

_ENV_KEY = "DANCEGRPO_ROLLOUT_IMAGE_DIR"


def get_rollout_image_dir() -> str:
    return os.environ.get(_ENV_KEY, "./images")


def rollout_image_file(basename: str) -> str:
    d = get_rollout_image_dir()
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, basename)
