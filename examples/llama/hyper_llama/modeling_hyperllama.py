from torch import Tensor, nn
from torch.nn import functional as F
from transformers.models.llama import LlamaForCausalLM, LlamaModel, LlamaPreTrainedModel

from .configuration_hyperllama import HyperLlamaConfig


class ScaledLinear(nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, scaling_factor: int = 1
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.scale_factor = scaling_factor

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(
            input,
            self.weight / self.scale_factor,
            self.bias / self.scale_factor if self.bias is not None else None,
        )


class HyperLlamaForCausalLM(LlamaForCausalLM):
    config_class = HyperLlamaConfig

    def __init__(self, config):
        # Skip initializing LlamaForCausalLM
        super(LlamaPreTrainedModel, self).__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = ScaledLinear(
            config.hidden_size, config.vocab_size, bias=False, scaling_factor=config.lm_head_normalization_factor
        )

        # Initialize weights and apply final processing
        self.post_init()
