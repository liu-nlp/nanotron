import math
from abc import abstractmethod
from enum import Enum, auto
from typing import Dict

import torch
from nanotron.config import Config
from nanotron.config.config import get_config_from_file
from nanotron.config.models_config import HyperCloningInit, InitScalingMethod, LlamaConfig
from nanotron.config.parallelism_config import ParallelismArgs
from nanotron.models.base import build_model
from nanotron.nn.layer_norm import LlamaRMSNorm, TritonRMSNorm
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.parameters import sanity_check
from nanotron.parallel.tensor_parallel.nn import (
    ScaledTensorParallelColumnLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
)
from torch import nn
from torch.nn import init


class ParametrizationMethod(Enum):
    STANDARD = auto()
    SPECTRAL_MUP = auto()


class Parametrizator:
    def __init__(self, config: Config):
        self.config = config

    def parametrize(self, param_name: str, module: nn.Module, full_param_name: str):
        if not isinstance(module, tuple(self.MODULE_TO_PARAMETRIZE.keys())):
            raise Exception(f"Module {type(module)} with parameter {param_name} is not supported for initialization")

        return self.MODULE_TO_PARAMETRIZE[type(module)](param_name, module, full_param_name)


class StandardParametrizator(Parametrizator):
    def __init__(self, config: Config):
        super().__init__(config)
        self.MODULE_TO_PARAMETRIZE = {
            TensorParallelColumnLinear: self._parametrize_column_linear,
            TensorParallelRowLinear: self._parametrize_row_linear,
            TritonRMSNorm: self._parametrize_layer_norm,
            LlamaRMSNorm: self._parametrize_layer_norm,
            TensorParallelEmbedding: self._parametrize_embedding,
        }

        self.std = config.model.init_method.std
        self.num_layers = config.model.model_config.num_hidden_layers
        self.tp = config.parallelism.tp
        self.scaling_method = config.model.init_method.scaling_method
        self.hidden_size = config.model.model_config.hidden_size

    def _parametrize_column_linear(self, param_name: str, module: nn.Module, full_param_name: str):
        assert param_name in ["weight", "bias"]

        if "weight" == param_name:
            # TODO @nouamane: should we use trunc_normal_
            init.normal_(module.weight, mean=0.0, std=self.std)
        elif "bias" == param_name:
            module.bias.zero_()

    def _compute_scaling_factor(self) -> float:
        """Compute initialization scaling based on selected method"""
        if self.scaling_method == InitScalingMethod.NONE:
            return 1.0
        elif self.scaling_method == InitScalingMethod.NUM_LAYERS:
            # Scale based on total network depth
            return math.sqrt(2 * self.num_layers)
        elif self.scaling_method == InitScalingMethod.LAYER_INDEX:
            # Scale based on layer position
            raise NotImplementedError("Layer position scaling not yet implemented")
        else:
            raise ValueError(f"Invalid scaling method: {self.scaling_method}")

    def _parametrize_row_linear(self, param_name: str, module: nn.Module, full_param_name: str):
        assert param_name in ["weight", "bias"]

        if "weight" == param_name:
            scaling = self._compute_scaling_factor()
            adjusted_std = self.std / scaling
            # TODO @nouamane: should we use trunc_normal_
            init.normal_(module.weight, mean=0.0, std=adjusted_std)
        elif "bias" == param_name:
            module.bias.zero_()

    def _parametrize_layer_norm(self, param_name: str, module: nn.Module, full_param_name: str):
        assert param_name in ["weight", "bias"]

        if "weight" == param_name:
            module.weight.fill_(1)
        elif "bias" == param_name:
            module.bias.zero_()

    def _parametrize_embedding(self, param_name: str, module: nn.Module, full_param_name: str):
        assert param_name in ["weight"]

        if "weight" == param_name:
            init.normal_(module.weight, mean=0.0, std=self.std)


class SpectralMupParametrizator(Parametrizator):
    """
    A Spectral Condition for Feature Learning by Greg Yang, et al.
    https://arxiv.org/abs/2310.17813
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.MODULE_TO_PARAMETRIZE = {
            TensorParallelColumnLinear: self._parametrize_mup_weight,
            TensorParallelRowLinear: self._parametrize_mup_weight,
            TritonRMSNorm: self._parametrize_layer_norm,
            TensorParallelEmbedding: self._parametrize_embedding,
        }
        self.std = 1.0

    @staticmethod
    def _compute_spectral_std(std: float, fan_in: int, fan_out: int):
        """
        Parametrization 1 (Spectral parametrization)
        Page 8, A Spectral Condition for Feature Learning by Greg Yang, et al.

        σₗ = Θ(1/√nₗ₋₁ min{1, √(nₗ/nₗ₋₁)})
        """
        return (std / math.sqrt(fan_in)) * min(1, math.sqrt(fan_out / fan_in))

    def _parametrize_mup_weight(self, param_name: str, module: nn.Module, full_param_name: str):
        assert param_name in ["weight", "bias"]

        data = module.weight if param_name == "weight" else module.bias
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(data)
        world_size = module.world_size

        if isinstance(module, TensorParallelColumnLinear):
            fan_out = fan_out * world_size
        elif isinstance(module, TensorParallelRowLinear):
            fan_in = fan_in * world_size
        else:
            raise ValueError(f"Unknown module {module}")

        std = SpectralMupParametrizator._compute_spectral_std(std=self.std, fan_in=fan_in, fan_out=fan_out)
        init.normal_(data, mean=0.0, std=std)

    def _parametrize_layer_norm(self, param_name: str, module: nn.Module, full_param_name: str):
        assert param_name in ["weight", "bias"]

        # NOTE: you're free to change the initialization of layer norm
        # as it's not a part of µTransfer
        if "weight" == param_name:
            module.weight.fill_(1)
        elif "bias" == param_name:
            module.bias.zero_()

    def _parametrize_embedding(self, param_name: str, module: nn.Module, full_param_name: str):
        assert param_name in ["weight"]

        # NOTE: you're free to change the initialization of input embedding/lm head
        if "weight" == param_name:
            init.normal_(module.weight, mean=0.0, std=self.std)


class HyperCloningParametrizator(Parametrizator):
    def __init__(self, config: Config):
        super().__init__(config)
        self._model_args = config.model
        init_method = self._model_args.init_method
        assert isinstance(init_method, HyperCloningInit), "Config class doesn't match parametrizator"
        self._parametrization_config = init_method

        self.MODULE_TO_PARAMETRIZE = {
            TensorParallelColumnLinear: self._clone_linear,
            ScaledTensorParallelColumnLinear: self._clone_embedding,
            TensorParallelRowLinear: self._clone_linear,
            TritonRMSNorm: self._clone_layer_norm,
            LlamaRMSNorm: self._clone_layer_norm,
            TensorParallelEmbedding: self._clone_embedding,
        }

        assert isinstance(
            init_method, HyperCloningInit
        ), f"Expected hyper cloning configuration, got {self._model_args.init_method!r}"

        self._original_path = init_method.path
        self.up_projection_cloning_factor = init_method.up_projection_cloning_factor
        self.embedding_dimension_cloning_factor = init_method.embedding_dimension_cloning_factor
        self._original_weights = None
        self._base_model = None

    def _clone_layer_norm(self, param_name: str, module: nn.Module, full_param_name: str):
        assert self._base_model is not None, "Base model weights have not been initialized for HyperCloning"
        assert param_name in ["weight", "bias"]
        base_param = self._base_model.get_parameter(full_param_name)

        if "weight" == param_name:
            module.weight.copy_(base_param.repeat(self.embedding_dimension_cloning_factor))
        elif "bias" == param_name:
            module.bias.copy_(base_param.repeat(self.embedding_dimension_cloning_factor))

    def _clone_qkv(self, param_name: str, module: nn.Module, base_param: nn.Parameter) -> None:
        q_heads = self._model_args.model_config.num_attention_heads
        # Fall back to number of attention heads is key value heads are not separately defined
        kv_heads = (
            self._model_args.model_config.num_key_value_heads or self._model_args.model_config.num_attention_heads
        )

        expected_q_heads = self._base_model.config.num_attention_heads * self.embedding_dimension_cloning_factor
        if q_heads != expected_q_heads:
            raise ValueError(
                f"num_attention_heads was {q_heads} but expected {expected_q_heads} after cloning with an embedding_dimension_cloning_factor of {self.embedding_dimension_cloning_factor}"
            )

        if self._base_model.config.num_key_value_heads is not None and kv_heads != (
            expected_kv_heads := self._base_model.config.num_key_value_heads * self.embedding_dimension_cloning_factor
        ):
            raise ValueError(
                f"num_key_value_heads was {kv_heads} but expected {expected_kv_heads} after cloning with an embedding_dimension_cloning_factor of {self.embedding_dimension_cloning_factor}"
            )

        # Hidden size per head
        d_qk = self._base_model.config.hidden_size // q_heads
        split_qkv = torch.split(
            base_param,
            [
                q_heads * d_qk,
                kv_heads * d_qk,
                kv_heads * d_qk,
            ],
            dim=0,
        )

        if "weight" == param_name:
            q_weight, k_weight, v_weight = split_qkv

            base_d_model = q_weight.shape[1]
            d_model = self._model_args.model_config.hidden_size
            q_weight = q_weight.view(d_qk, q_heads, base_d_model)
            k_weight = k_weight.view(d_qk, kv_heads, base_d_model)
            v_weight = v_weight.view(d_qk, kv_heads, base_d_model)

            module.weight.copy_(
                torch.concat(
                    [
                        (
                            weight.repeat(
                                self.embedding_dimension_cloning_factor, 1, self.embedding_dimension_cloning_factor
                            )
                            / self.embedding_dimension_cloning_factor
                        ).view(-1, d_model)
                        for weight in (q_weight, k_weight, v_weight)
                    ]
                )
            )
        else:
            q_bias, k_bias, v_bias = split_qkv

            q_bias = q_bias.view(d_qk, q_heads)
            k_bias = k_bias.view(d_qk, kv_heads)
            v_bias = v_bias.view(d_qk, kv_heads)

            module.bias.copy_(
                torch.concat(
                    [
                        bias.repeat(self.embedding_dimension_cloning_factor, 1).view(-1)
                        for bias in (q_bias, k_bias, v_bias)
                    ]
                )
            )

    def _clone_linear(self, param_name: str, module: nn.Module, full_param_name: str):
        assert self._base_model is not None, "Base model weights have not been initialized for HyperCloning"
        assert param_name in ["weight", "bias"]
        base_param = self._base_model.get_parameter(full_param_name)

        # Handle QKV Projection
        layer_name = full_param_name.rsplit(".", 2)[1]
        if layer_name == "qkv_proj":
            return self._clone_qkv(param_name, module, base_param)

        if "weight" == param_name:
            in_features_base = base_param.shape[1]
            out_features_base = base_param.shape[0]
            in_features_scale = module.weight.shape[1] // in_features_base
            out_features = module.weight.shape[0]
            out_features_scale = out_features // out_features_base

            if layer_name == "gate_up_proj":
                out_features_base = out_features_base // 2
                # Unmerge gate and up_proj weights and repeat them individually
                cloned_weight = (
                    base_param.reshape(2, out_features_base, in_features_base)
                    .repeat(1, out_features_scale, in_features_scale)
                    .view(out_features, -1)
                    / in_features_scale
                )
            else:
                # Repeat by the given factor and normalize
                cloned_weight = base_param.repeat(out_features_scale, in_features_scale) / in_features_scale

            module.weight.copy_(cloned_weight)

        elif "bias" == param_name:
            if layer_name == "gate_up_proj":
                # Unmerge gate and up_proj biases and repeat them individually
                module.bias.copy_(base_param.reshape(2, -1).repeat(1, out_features_scale).view(-1))
            else:
                module.bias.copy_(base_param.repeat(out_features_scale))

    def _clone_embedding(self, param_name: str, module: nn.Module, full_param_name: str):
        assert self._base_model is not None, "Base model weights have not been initialized for HyperCloning"
        assert param_name == "weight"

        # Only clone embedding dimension and don't normalize by cloning factor
        module.weight.copy_(
            self._base_model.get_parameter(full_param_name).repeat(1, self.embedding_dimension_cloning_factor)
        )

    def deallocate_base_weights(self) -> None:
        self._base_model = None
        # Clean torch cache
        torch.cuda.empty_cache()

    def load_original_weights(self) -> None:
        # TODO (Kevin): Workaround to avoid circular import for weight loading utilities
        from nanotron.models import CONFIG_TO_TRAINING_MODEL_CLASS
        from nanotron.serialize.weights import load_weights
        from nanotron.trainer import mark_tied_parameters

        checkpoint_path = self._parametrization_config.path

        base_config = get_config_from_file(
            config_path=str(checkpoint_path / "config.yaml"),
        )
        base_model_config = base_config.model.model_config

        assert isinstance(base_model_config, LlamaConfig) and isinstance(
            self._model_args.model_config, LlamaConfig
        ), "Currently only models using the Llama architecture are supported as base models and for continuous pretraining using HyperCloning initialization"

        base_lm_head_normalization = base_model_config.lm_head_normalization_factor
        target_lm_head_normalization = self._model_args.model_config.lm_head_normalization_factor
        if (
            self._model_args.model_config.tie_word_embeddings
            and base_lm_head_normalization * self.embedding_dimension_cloning_factor != target_lm_head_normalization
        ):
            raise ValueError(
                f"lm_head_normalization_factor of the base model ({base_lm_head_normalization}) has to be scaled by the embedding_dimension_cloning_factor ({self.embedding_dimension_cloning_factor})"
                f" when tie_word_embeddings is `true`, but was {target_lm_head_normalization}"
            )

        # TODO (Kevin): Could be restored on CPU to avoid temporary overhead
        parallel_config = ParallelismArgs(
            dp=self.config.parallelism.dp * self.config.parallelism.pp * self.config.parallelism.tp,
            pp=1,
            tp=1,
            # Set to match `run_generate` default
            tp_linear_async_communication=False,
        )

        # Initialise all process groups
        parallel_context = ParallelContext(
            data_parallel_size=self.config.parallelism.dp * self.config.parallelism.pp * self.config.parallelism.tp,
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
        )

        self._base_model = build_model(
            model_builder=lambda: CONFIG_TO_TRAINING_MODEL_CLASS[base_config.model.model_config.__class__.__name__](
                config=base_model_config,
                parallel_context=parallel_context,
                parallel_config=parallel_config,
            ),
            dtype=torch.bfloat16,
            parallel_context=parallel_context,
        )

        # NOTE: Only required to convert parameters to NanotronParameter
        mark_tied_parameters(
            model=self._base_model, parallel_context=parallel_context, parallel_config=parallel_config
        )

        sanity_check(root_module=self._base_model)

        load_weights(model=self._base_model, parallel_context=parallel_context, root_folder=checkpoint_path)


class LearningRateForParametrizator:
    def __init__(self, lr: float, names_to_modules: Dict[str, nn.Module]):
        self.lr = lr
        self.names_to_modules = names_to_modules

    @abstractmethod
    def get_lr(self, param_name: str, module: nn.Module) -> float:
        raise NotImplementedError


class LearningRateForSP(LearningRateForParametrizator):
    """All parameters get the same learning rate."""

    def get_lr(self, param_name: str, param: nn.Module) -> float:
        return self.lr


class LearningRateForSpectralMup(LearningRateForParametrizator):
    """
    A Spectral Condition for Feature Learning by Greg Yang, et al.

    NOTE: each parameter gets a custom learning rate based on its fan-in and fan-out.
    """

    def __init__(self, lr: float, names_to_modules: Dict[str, nn.Module]):
        super().__init__(lr, names_to_modules)
        self.MODULE_TO_PARAMETRIZE = {
            TensorParallelColumnLinear: self._get_mup_lr,
            TensorParallelRowLinear: self._get_mup_lr,
            TritonRMSNorm: self._get_global_lr,
            TensorParallelEmbedding: self._get_global_lr,
        }

    def _get_mup_lr(self, param: nn.Parameter, module: nn.Module):
        """
        Parametrization 1 (Spectral parametrization)
        Page 8, A Spectral Condition for Feature Learning by Greg Yang, et al.

        ηₗ = Θ(nₗ/nₗ₋₁)
        """
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(param)
        world_size = module.world_size

        if isinstance(module, TensorParallelColumnLinear):
            fan_out = fan_out * world_size
        elif isinstance(module, TensorParallelRowLinear):
            fan_in = fan_in * world_size
        else:
            raise ValueError(f"Unknown module {module}")

        return self.lr * (fan_out / fan_in)

    def _get_global_lr(self, param: nn.Parameter, module: nn.Module) -> float:
        return self.lr

    def get_lr(self, param_name: str, param: nn.Parameter) -> float:
        """Return the learning rate for the given parameter."""
        # NOTE: param_name should be like 'model.token_position_embeddings.pp_block.token_embedding.weight'
        # since names_to_modules map module_name to module
        # so we remove the .weight and .bias from param_name to get the module_name
        module_name = param_name.rsplit(".", 1)[0]
        module = self.names_to_modules[module_name]
        return self.MODULE_TO_PARAMETRIZE[type(module)](param, module)
