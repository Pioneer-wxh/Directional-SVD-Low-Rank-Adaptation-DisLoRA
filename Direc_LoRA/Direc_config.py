from dataclasses import dataclass, field
from peft import LoraConfig
from typing import Optional, Union, List


@dataclass
class Direc_config(LoraConfig):
    """
    自定义 LoRA 配置，继承自 PEFT 的 LoraConfig，支持 SORSA 结构、训练最小奇异值、预热阶段和动态 top-k 奇异向量选择。

    Args:
        r (int): LoRA 矩阵的秩，即训练的奇异向量/值的数量。
        target_modules (Union[List[str], str]): 应用 LoRA 的模块，例如 ['q_proj', 'v_proj']。
        lora_alpha (int): LoRA 更新的缩放因子（scaling = lora_alpha / r）。
        lora_dropout (float): LoRA 层的 dropout 率。
        fan_in_fan_out (bool): 如果层的权重存储为 (fan_in, fan_out)，设置为 True。
        bias (str): 偏置配置，默认为 'none'。
        task_type (str): 任务类型，例如 'CAUSAL_LM' 或 'SEQ_2_SEQ_LM'。
        warmup_steps (int): 在引入额外低秩更新之前的预热步数。
        s_tsd (int): 预热后动态选择的 top-k 奇异向量数量。
        ortho_lambda (float): 正交正则化损失的权重。
        prefer_small_sigma (bool): 是否在 top-k 选择中优先选择最小的奇异值。
    """
    r: int = field(default=8, metadata={"help": "LoRA 矩阵的秩，即训练的奇异向量/值的数量"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "要替换为 LoRA 的模块名称列表或正则表达式。例如，['q_proj', 'v_proj'] 或 "
                "'.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'。也可以是 'all-linear'，匹配除输出层外的所有线性/Conv1D 层。"
                "如果未指定，将根据模型架构选择模块；如果架构未知，将抛出错误，需要手动指定目标模块。"
            ),
        },
    )
    lora_alpha: int = field(default=8, metadata={
                            "help": "LoRA 缩放因子（scaling = lora_alpha / r）"})
    lora_dropout: float = field(default=0.0, metadata={
                                "help": "LoRA dropout 率"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "如果要替换的层权重存储为 (fan_in, fan_out)，设置为 True"},
    )
    bias: str = field(
        default="none",
        metadata={"help": "偏置配置，'none'、'all' 或 'lora_only'"},
    )
    task_type: str = field(
        default="CAUSAL_LM",
        metadata={"help": "任务类型，例如 'CAUSAL_LM' 或 'SEQ_2_SEQ_LM'"},
    )
    warmup_steps: int = field(
        default=50,
        metadata={"help": "在引入额外低秩更新之前的预热步数"},
    )
    s_tsd: int = field(
        default=8,
        metadata={"help": "预热后动态选择的 top-k 奇异向量数量"},
    )
    ortho_lambda: float = field(
        default=1,
        metadata={"help": "正交正则化损失的权重"},
    )
    prefer_small_sigma: bool = field(
        default=True,
        metadata={"help": "是否在 top-k 选择中优先选择最小的奇异值"},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.target_modules is None:
            raise ValueError("target_modules 必须指定为字符串或字符串列表")
        if self.r <= 0:
            raise ValueError("LoRA 秩 r 必须大于 0")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha 必须大于 0")
