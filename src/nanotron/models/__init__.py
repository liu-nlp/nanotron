# flake8: noqa
from .base import DTypeInvariantTensor, NanotronModel, build_model, check_model_has_grad, init_on_device_and_dtype
from .llama import LlamaForTraining
from .qwen import Qwen2ForTraining
from .starcoder2 import Starcoder2ForTraining


CONFIG_TO_TRAINING_MODEL_CLASS = {
    "LlamaConfig": LlamaForTraining,
    "Starcoder2Config": Starcoder2ForTraining,
    "Qwen2Config": Qwen2ForTraining,
}
