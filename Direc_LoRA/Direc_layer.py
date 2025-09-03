import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.lora.layer import LoraLayer
from peft.utils.integrations import gather_params_ctx
from peft.utils import transpose

class Direc_Linear(nn.Module, LoraLayer):
    """Custom LoRA linear layer, inherits LoraLayer, supports SVD initialization and dynamic singular vector selection.
    Only supports nn.Linear, adapter dtype aligns with base_layer, compatible with DeepSpeed.
    """
    adapter_layer_names = ("Direc_Ur", "Direc_Sr", "Direc_Vhr", "Direc_Utsd", "Direc_Vhtsd", "Direc_Stsd")
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout", "warmup_steps", "s_tsd", "prefer_small_sigma")

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        warmup_steps: int = 100,
        s_tsd: int = 8,
        prefer_small_sigma: bool = True,
        merge_weights: bool = True,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs
    ):
        nn.Module.__init__(self)
        self.base_layer = base_layer
        self.U = None
        self.S = None
        self.Vh = None
        self._disable_adapters = False
        self.merged_adapters = []
        self.ephemeral_gpu_offload = kwargs.get("ephemeral_gpu_offload", False)

        # Validate base_layer and set dimensions
        if not isinstance(base_layer, nn.Linear):
            raise ValueError(f"Direc_Linear only supports torch.nn.Linear, got {type(base_layer)}")
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        # Get base_layer's dtype and device for alignment
        self.base_dtype = base_layer.weight.dtype
        self.base_device = base_layer.weight.device

        # Initialize adapter parameter containers
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_bias = {}

        self.fan_in_fan_out = fan_in_fan_out
        self.warmup_steps = warmup_steps
        self.s_tsd = s_tsd
        self.merge_weights = merge_weights
        self.prefer_small_sigma = prefer_small_sigma
        self.register_buffer('step', torch.tensor(0, dtype=torch.long))
        self._active_adapter = adapter_name

        # Initialize adapter parameter containers
        self.Direc_Ur = nn.ModuleDict({})
        self.Direc_Sr = nn.ParameterDict({})
        self.Direc_Vhr = nn.ModuleDict({})
        self.Direc_Utsd = nn.ModuleDict({})
        self.Direc_Vhtsd = nn.ModuleDict({})
        self.Direc_Stsd = nn.ParameterDict({})

        self.update_layer(
            adapter_name=adapter_name,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            lora_bias=lora_bias,
        )

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        init_lora_weights: Union[bool, str],
        use_rslora: bool,
        lora_bias: bool,
    ):
        """Allocate adapter parameters and perform SVD initialization with base_layer alignment."""
        if r <= 0:
            raise ValueError(f"`r` must be a positive integer, got {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.lora_bias[adapter_name] = lora_bias
        if lora_dropout > 0.0:
            self.lora_dropout[adapter_name] = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout[adapter_name] = nn.Identity()

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # Allocate adapter parameters with base_layer's dtype and device
        self.Direc_Ur[adapter_name] = nn.Linear(r, self.out_features, bias=False, dtype=self.base_dtype, device=self.base_device)
        self.Direc_Sr[adapter_name] = nn.Parameter(torch.zeros(r, dtype=self.base_dtype, device=self.base_device))
        self.Direc_Vhr[adapter_name] = nn.Linear(self.in_features, r, bias=False, dtype=self.base_dtype, device=self.base_device)
        self.Direc_Utsd[adapter_name] = nn.Linear(self.s_tsd, self.out_features, bias=False, dtype=self.base_dtype, device=self.base_device)
        self.Direc_Utsd[adapter_name].weight.requires_grad = False
        self.Direc_Stsd[adapter_name] = nn.Parameter(torch.zeros(self.s_tsd, dtype=self.base_dtype, device=self.base_device))
        self.Direc_Vhtsd[adapter_name] = nn.Linear(self.in_features, self.s_tsd, bias=False, dtype=self.base_dtype, device=self.base_device)
        self.Direc_Vhtsd[adapter_name].weight.requires_grad = False

        # SVD initialization
        if init_lora_weights:
            with gather_params_ctx(self.get_base_layer().weight):
                weight_float32 = self.get_base_layer().weight.data.float()
                U, S, Vh = torch.linalg.svd(weight_float32, full_matrices=False)
                self.U = U.to(dtype=self.base_dtype, device=self.base_device)
                self.S = S.to(dtype=self.base_dtype, device=self.base_device)
                self.Vh = Vh.to(dtype=self.base_dtype, device=self.base_device)
                # Select top-r singular vectors
                indices = torch.topk(S, r, largest=not self.prefer_small_sigma)[1]
                if self.prefer_small_sigma:
                    indices = indices.flip(0)
                self.Direc_Ur[adapter_name].weight.data = U[:, indices].contiguous().to(dtype=self.base_dtype, device=self.base_device)
                self.Direc_Sr[adapter_name].data = S[indices].contiguous().to(dtype=self.base_dtype, device=self.base_device)
                self.Direc_Vhr[adapter_name].weight.data = Vh[indices, :].contiguous().to(dtype=self.base_dtype, device=self.base_device)

                # Compute merged weights: Vh @ diag(S) @ U
                S_diag = torch.diag(self.Direc_Sr[adapter_name])  # (r, r)
                merge = (self.Direc_Ur[adapter_name].weight @ S_diag @ self.Direc_Vhr[adapter_name].weight) * self.scaling[adapter_name]
                merge = merge.to(dtype=self.base_dtype, device=self.base_device)
                if merge.shape != self.get_base_layer().weight.shape:
                    raise ValueError(
                        f"Expected merge shape {self.get_base_layer().weight.shape}, but got {merge.shape}."
                    )
                self.get_base_layer().weight.data -= merge

        self.set_adapter([adapter_name])

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        """Compute LoRA delta weight with base_layer alignment."""
        weight_U = self.Direc_Ur[adapter].weight
        weight_S = self.Direc_Sr[adapter]
        weight_Vh = self.Direc_Vhr[adapter].weight

        # Construct diagonal matrix
        S_diag = torch.diag(weight_S)  # (r, r)
        output_tensor = transpose((weight_Vh @ S_diag @ weight_U), self.fan_in_fan_out) * self.scaling[adapter]
        return output_tensor.to(dtype=self.base_dtype, device=self.base_device)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass: base weights + LoRA contribution + dynamic singular vector contribution."""
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters or not self.r:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)
        if adapter_names is not None:
            return self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        if self.merged:
            return self.base_layer(x, *args, **kwargs)

        result = self.base_layer(x, *args, **kwargs) # (n, dm) -> 

        for active_adapter in self.active_adapters:
            if active_adapter not in self.Direc_Ur.keys():
                continue
            Direc_Ur = self.Direc_Ur[active_adapter]
            Direc_Sr = self.Direc_Sr[active_adapter]
            Direc_Vhr = self.Direc_Vhr[active_adapter]
            Direc_Utsd = self.Direc_Utsd[active_adapter]
            Direc_Stsd = self.Direc_Stsd[active_adapter]
            Direc_Vhtsd = self.Direc_Vhtsd[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # LoRA contribution
            temp = dropout(x)
            temp = Direc_Vhr(temp)  # (batch, r) -> (batch, out_features)
            temp = temp * Direc_Sr  # (batch, r) * (r,) -> (batch, r)
            temp = Direc_Ur(temp)  # (batch, in_features) -> (batch, r)
            result += (temp * scaling)

            # Dynamic singular vector selection
            if self.step < self.warmup_steps:
                pass
            elif self.step == self.warmup_steps:
                with torch.no_grad():
                    S_diag = torch.diag(Direc_Sr)  # (r, r)
                    deltaW = (Direc_Ur.weight @ S_diag @ Direc_Vhr.weight).to(dtype=self.base_dtype, device=self.base_device)
                    delta_sigma = torch.diag(self.U.T @ deltaW @ self.Vh.T)
                    top_index = self.calculate_change_rate(self.S, delta_sigma, self.s_tsd, not self.prefer_small_sigma)

                    # Freeze Utsd and Vhtsd parameters
                    self.Direc_Utsd[active_adapter].weight.data = self.U[:, top_index].to(dtype=self.base_dtype, device=self.base_device)
                    self.Direc_Utsd[active_adapter].weight.requires_grad = False
                    self.Direc_Stsd[active_adapter].data = self.S[top_index].to(dtype=self.base_dtype, device=self.base_device)
                    self.Direc_Vhtsd[active_adapter].weight.data = self.Vh[top_index, :].to(dtype=self.base_dtype, device=self.base_device)
                    self.Direc_Vhtsd[active_adapter].weight.requires_grad = False
            else:
                temp = dropout(x)
                temp = Direc_Vhtsd(temp)
                temp = temp * Direc_Stsd
                temp = Direc_Utsd(temp)
                result += (temp * scaling)
            
            if self.training:
                with torch.no_grad():
                    self.step += 1

        return result

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """Merge adapter weights into base model weights."""
        adapter_names = self._get_adapters_to_merge(adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.Direc_Ur.keys():
                base_layer = self.get_base_layer()
                delta_weight = self.get_delta_weight(active_adapter)
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += delta_weight

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights

                    if self.lora_bias[active_adapter]:
                        new_bias = base_layer.bias + self.Direc_Vhr[active_adapter].bias
                        if not torch.isfinite(new_bias).all():
                            raise ValueError(
                                f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                            )
                        base_layer.bias.data = new_bias
                else:
                    base_layer.weight.data += delta_weight
                    if self.lora_bias[active_adapter]:
                        base_layer.bias.data += self.Direc_Vhr[active_adapter].bias

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """Unmerge adapter weights."""
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.Direc_Ur.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)
                if self.lora_bias[active_adapter]:
                    self.get_base_layer().bias.data -= self.Direc_Vhr[active_adapter].bias

    def calculate_change_rate(self, a: torch.Tensor, b: torch.Tensor, s: int = 8, largest: bool = True) -> torch.Tensor:
        """Compute singular value change rate and select top-s indices."""
        with torch.no_grad():
            change_rate = abs(b) / (abs(a) + 1e-8)
            _, top_s_indices = torch.topk(change_rate, s, largest=largest)
        return top_s_indices

    def _get_adapters_to_merge(self, adapter_names: Optional[list[str]]) -> list[str]:
        """Check adapter names to merge."""
        if adapter_names is None:
            adapter_names = self.active_adapters
        adapter_names = [name for name in adapter_names if name in self.Direc_Ur.keys()]
        return adapter_names

    def __repr__(self) -> str:
        rep = super().__repr__()
        return rep