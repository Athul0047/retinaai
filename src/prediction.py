"""Local inference for the educational RetinaAI DR grading prototype.

This model is not clinically validated and must not be used for diagnosis or
clinical decision-making.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import models
from torchvision.transforms.functional import pil_to_tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "models" / "config.json"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "retinaai_model.pth"
NUM_CLASSES = 5
IMAGE_SIZE = 224


class PredictionError(RuntimeError):
    """Raised when configuration, model files, or image inference is invalid."""


def _read_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PredictionError(f"Model configuration file was not found: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PredictionError(f"Could not read valid JSON configuration at {path}: {exc}") from exc

    if not isinstance(config, dict):
        raise PredictionError("Model configuration must be a JSON object.")
    if config.get("architecture") != "efficientnet_b0":
        raise PredictionError("Configuration architecture must be 'efficientnet_b0'.")
    if config.get("num_classes") != NUM_CLASSES:
        raise PredictionError(f"Configuration must specify exactly {NUM_CLASSES} classes.")
    if config.get("image_size") != IMAGE_SIZE:
        raise PredictionError(f"Configuration image_size must be {IMAGE_SIZE}.")

    names = config.get("class_names")
    if not isinstance(names, list) or len(names) != NUM_CLASSES or not all(
        isinstance(name, str) and name.strip() for name in names
    ):
        raise PredictionError("Configuration class_names must contain five non-empty names.")
    if len(set(names)) != NUM_CLASSES:
        raise PredictionError("Configuration class_names must be unique.")
    for key in ("mean", "std"):
        values = config.get(key)
        if (
            not isinstance(values, list)
            or len(values) != 3
            or not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in values
            )
        ):
            raise PredictionError(f"Configuration {key} must contain three numeric RGB values.")
        if key == "std" and any(value <= 0 for value in values):
            raise PredictionError("Configuration std values must be greater than zero.")
    return config


def _available_device(requested: str | torch.device | None) -> torch.device:
    if requested is None:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        return torch.device("cpu")
    if device.type not in ("cpu", "mps"):
        raise PredictionError("Only CPU and Apple Silicon MPS inference are supported.")
    return device


class RetinaPredictor:
    """Load the local model once and predict five DR grades from an image."""

    def __init__(
        self,
        config_path: str | Path = CONFIG_PATH,
        checkpoint_path: str | Path = CHECKPOINT_PATH,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = _read_config(Path(config_path))
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.is_file():
            raise PredictionError(f"Model checkpoint was not found: {checkpoint_file}")

        try:
            checkpoint = torch.load(
                checkpoint_file,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise PredictionError(f"Could not safely load model checkpoint {checkpoint_file}: {exc}") from exc

        if not isinstance(checkpoint, dict):
            raise PredictionError("Checkpoint must be a state dictionary or contain model_state_dict.")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state_dict, dict) or not state_dict:
            raise PredictionError("Checkpoint does not contain a usable model state dictionary.")

        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, NUM_CLASSES)
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise PredictionError(f"Checkpoint keys or tensor shapes do not match EfficientNet-B0: {exc}") from exc

        self.device = _available_device(device)
        try:
            self.model.to(self.device)
        except (RuntimeError, NotImplementedError):
            self.device = torch.device("cpu")
            self.model.to(self.device)
        self.model.eval()

        self._mean = torch.tensor(self.config["mean"], dtype=torch.float32).view(1, 3, 1, 1)
        self._std = torch.tensor(self.config["std"], dtype=torch.float32).view(1, 3, 1, 1)

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        rgb = image.convert("RGB")
        width, height = rgb.size
        if width <= 0 or height <= 0:
            raise PredictionError("Image dimensions must be greater than zero.")

        # Colab's exact padding color and resize interpolation are not recorded.
        # Use centered black padding and Pillow bilinear interpolation as defaults.
        side = max(width, height)
        square = Image.new("RGB", (side, side), color=(0, 0, 0))
        square.paste(rgb, ((side - width) // 2, (side - height) // 2))
        resized = square.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR)

        tensor = pil_to_tensor(resized).to(dtype=torch.float32).div_(255.0).unsqueeze(0)
        return (tensor - self._mean) / self._std

    def predict(self, image: Image.Image | str | Path) -> dict[str, Any]:
        """Return the winning grade and softmax probabilities for all five classes."""
        try:
            if isinstance(image, Image.Image):
                source = image.copy()
            elif isinstance(image, (str, Path)):
                with Image.open(image) as opened:
                    source = opened.copy()
            else:
                raise PredictionError("Image input must be a Pillow image or a filesystem path.")
            source.load()
            input_tensor = self._preprocess(source).to(self.device)
        except PredictionError:
            raise
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            raise PredictionError(f"Could not read or preprocess the image: {exc}") from exc

        try:
            with torch.inference_mode():
                logits = self.model(input_tensor)
                probabilities_tensor = torch.softmax(logits, dim=1)[0].to("cpu")
        except (RuntimeError, NotImplementedError) as exc:
            if self.device.type != "mps":
                raise PredictionError(f"Model inference failed: {exc}") from exc
            self.device = torch.device("cpu")
            self.model.to(self.device).eval()
            try:
                with torch.inference_mode():
                    logits = self.model(input_tensor.to(self.device))
                    probabilities_tensor = torch.softmax(logits, dim=1)[0]
            except RuntimeError as cpu_exc:
                raise PredictionError(f"Model inference failed on MPS and CPU: {cpu_exc}") from cpu_exc

        grade_index = int(torch.argmax(probabilities_tensor).item())
        class_names = self.config["class_names"]
        return {
            "grade_index": grade_index,
            "class_name": class_names[grade_index],
            "probabilities": {
                name: float(probabilities_tensor[index].item())
                for index, name in enumerate(class_names)
            },
        }
