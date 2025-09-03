import torch
import torch.nn as nn
import os
import logging
from peft.tuners.lora import LoraModel
from .Direc_layer import Direc_Linear
from .Direc_config import Direc_config

class Direc_Model(LoraModel):

    prefix: str = "Direc_"

    def __init__(self, model: nn.Module, config: Direc_config, adapter_name: str = "default", low_cpu_mem_usage: bool = False):
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
        self.config = config

    def _create_and_replace(
        self,
        lora_config: Direc_config,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ):
        if not isinstance(target, (nn.Linear, Direc_Linear)):
            return

        try:
            # If target is MyLinear, update adapter parameters
            if isinstance(target, Direc_Linear):
                target.update_layer(
                    adapter_name=adapter_name,
                    r=lora_config.r,
                    lora_alpha=lora_config.lora_alpha,
                    lora_dropout=lora_config.lora_dropout,
                    init_lora_weights=lora_config.init_lora_weights,
                    lora_bias=lora_config.lora_bias,
                )
                return

            # Target is nn.Linear, create new module
            new_module = self._create_new_module(lora_config, adapter_name, target)

            # If adapter is not in active_adapters, disable training
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)

            # Replace module
            self._replace_module(parent, target_name, new_module, target)

        except Exception as e:
            raise RuntimeError(f"Failed to replace module {target_name}: {str(e)}")

    def _create_new_module(self, lora_config: Direc_config, adapter_name: str, target: nn.Module, **kwargs) -> nn.Module:
        """
        创建新的 Direc_Linear 模块，仅支持 nn.Linear。
        """
        if not isinstance(target, nn.Linear):
            raise ValueError(
                f"Target module {target} is not supported. Only `torch.nn.Linear` is supported."
            )

        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "lora_bias": lora_config.lora_bias,
            "warmup_steps": lora_config.warmup_steps,
            "s_tsd": lora_config.s_tsd,
            "prefer_small_sigma": lora_config.prefer_small_sigma,
        }

        new_module = Direc_Linear(
            base_layer=target,
            adapter_name=adapter_name,
            **kwargs
        )

        return new_module

    def print_trainable_parameters(self):
        trainable_params = sum(p.numel() for n, p in self.named_parameters() if p.requires_grad and self.prefix in n)
        all_param = sum(p.numel() for _, p in self.named_parameters())
        if all_param > 0:
            print(
                f"Trainable LoRA parameters: {trainable_params} || Total parameters: {all_param} || Trainable ratio%: {100 * trainable_params / all_param:.4f}"
            )
        else:
            print("No parameters found in model.")
    
    def save_module(self, save_directory: str="direc_adapters"):
        """
        Only save the state dictionary of Direc_Linear modules.
        """
        os.makedirs(save_directory, exist_ok=True)
        adapter_state_dicts = {}

        for name, module in self.named_modules():
            # Precisely identify Direc_Linear modules
            if isinstance(module, Direc_Linear):
                # Get state dictionary, ensure on CPU and detach computation graph
                # Note: Direc_Linear inherits from LoraLayer, its state_dict may contain base_layer
                # If only want to save adapter part, need to filter state_dict
                # PEFT usually uses get_adapter_state_dict method, but we are custom
                # Temporarily save complete state_dict, if optimization needed, filtering required
                state_dict = module.state_dict()
                adapter_state_dicts[name] = {k: v.cpu().detach().clone() for k, v in state_dict.items()}
                logging.info(f"Preparing to save module {name} state")

        # Save collected state dictionaries to a file
        save_path = os.path.join(save_directory, "direc_adapter_states.pt")
        try:
            torch.save(adapter_state_dicts, save_path)
            logging.info(f"Successfully saved adapter states to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save adapter states: {e}")
            raise

    def load_module(self, load_directory: str="direc_adapters"):
        """
        Load state dictionary of Direc_Linear modules from saved file, and ensure step buffer is long type.
        """
        load_path = os.path.join(load_directory, "direc_adapter_states.pt")

        if not os.path.exists(load_path):
            logging.error(f"Error: Adapter state file not found at {load_path}")
            raise FileNotFoundError(f"Adapter state file not found at {load_path}")

        try:
            # Load state dictionary to CPU
            adapter_state_dicts = torch.load(load_path, map_location='cpu')
            logging.info(f"Loaded adapter states from {load_path}")
        except Exception as e:
            logging.error(f"Failed to load adapter state file: {e}")
            raise

        loaded_modules = set()
        all_direc_modules = {name for name, module in self.named_modules() if isinstance(module, Direc_Linear)}

        for name, module in self.named_modules():
            if isinstance(module, Direc_Linear):
                if name in adapter_state_dicts:
                    state_dict = adapter_state_dicts[name]

                    # --- New: Check and force convert 'step' buffer type ---
                    buffer_key = 'step' # Buffer name registered in Direc_Linear
                    if buffer_key in state_dict and isinstance(state_dict[buffer_key], torch.Tensor):
                        if state_dict[buffer_key].dtype != torch.long:
                            original_dtype = state_dict[buffer_key].dtype
                            logging.warning(f"Module {name} loaded '{buffer_key}' buffer type is {original_dtype}, will force convert to torch.long.")
                            # Force convert to long type
                            state_dict[buffer_key] = state_dict[buffer_key].long()
                        # Optional: If need to ensure it's a scalar (0-dimensional tensor)
                        # if state_dict[buffer_key].ndim != 0:
                        #    logging.warning(f"Module {name} '{buffer_key}' buffer is not scalar, will try to take first element.")
                        #    state_dict[buffer_key] = state_dict[buffer_key].flatten()[0].long()

                    # ----------------------------------------------

                    # Get current module device (same as previous code)
                    try:
                       target_device = next(module.parameters()).device
                    except StopIteration:
                       try:
                           target_device = next(module.buffers()).device
                       except StopIteration:
                           target_device = torch.device('cpu')
                           logging.warning(f"Module {name} has no parameters or buffers, state dict will be loaded to CPU")

                    # Move tensors in loaded state dict to target device
                    state_dict = {k: v.to(target_device) for k, v in state_dict.items()}

                    try:
                        # Load state dict (strict=False recommended)
                        missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=False)
                        if missing_keys:
                             # Filter out missing key warnings that may appear due to not saving base_layer (if applicable)
                             missing_adapter_keys = [k for k in missing_keys if any(p in k for p in Direc_Linear.adapter_layer_names + ('step',))] # Check if it's Direc_Linear key part
                             if missing_adapter_keys:
                                logging.warning(f"Missing key adapter keys when loading module {name}: {missing_adapter_keys}")
                        if unexpected_keys:
                             # Filter out unexpected keys that may appear due to loading complete state_dict to adapter-only model (if applicable)
                             maybe_base_keys = [k for k in unexpected_keys if not any(p in k for p in Direc_Linear.adapter_layer_names + ('step',))]
                             if len(maybe_base_keys) < len(unexpected_keys): # If there are other unexpected keys besides base keys
                                 other_unexpected = list(set(unexpected_keys) - set(maybe_base_keys))
                                 if other_unexpected:
                                     logging.warning(f"Unexpected keys when loading module {name}: {other_unexpected}")

                        logging.info(f"Successfully loaded state dict for module {name}")
                        loaded_modules.add(name)
                    except Exception as e:
                        logging.error(f"Error loading state dict for module {name}: {e}")
                        # Can choose to continue or raise exception
                else:
                    logging.warning(f"Warning: Module {name} saved state not found in load file.")

        # Check if there are Direc_Linear modules not loaded (same as previous code)
        unloaded_modules = all_direc_modules - loaded_modules
        if unloaded_modules:
            logging.warning(f"Warning: Following Direc_Linear modules exist but their states not found in load file: {unloaded_modules}")