# LoRA (Low-Rank Adaptation)

LoRA已成为采用最广泛的PEFT（参数高效微调）方法。其工作原理是在注意力权重中添加小型秩分解矩阵，通常可将可训练参数减少约90%

## Understanding LoRA

LoRA（低秩适配）是一种参数高效的微调技术，该技术冻结预训练模型的权重，并将可训练的秩分解矩阵注入模型的层中。在微调期间，LoRA不是训练所有模型参数，而是通过低秩分解将权重更新分解为较小的矩阵，从而在保持模型性能的同时显著减少可训练参数的数量。例如，当应用于GPT-3 175B时，与完全微调相比，LoRA将可训练参数减少了10,000倍，GPU内存需求降低了3倍。您可以在[LoRA paper](https://arxiv.org/pdf/2106.09685)论文中了解更多关于LoRA的信息。

LoRA通过在Transformer层中添加成对的秩分解矩阵来工作，通常重点关注注意力权重。在推理期间，这些适配器权重可以与基础模型合并，从而不会产生额外的延迟开销。LoRA特别适用于在保持资源需求可控的同时，将大型语言模型适配到特定任务或领域。

## Loading LoRA Adapters

可以使用load_adapter()将适配器加载到预训练模型上，这对于尝试不同且权重未合并的适配器非常有用。使用set_adapter()函数设置适配器权重。若要返回到基础模型，可以使用unload()卸载所有LoRA模块。这使得在不同任务特定权重之间切换变得容易。

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("<base_model_name>")
peft_model_id = "<peft_adapter_id>"
model = PeftModel.from_pretrained(base_model, peft_model_id)
```
![lora_load_adapter](./images/lora_adapter.png)

## Merging LoRA Adapters

在使用LoRA进行训练后，您可能希望将适配器权重合并回基础模型，以便更轻松地部署。这将创建一个具有合并权重的单一模型，从而消除了在推理期间单独加载适配器的需要。

合并过程需要注意内存管理和精度。由于您需要同时加载基础模型和适配器权重，因此请确保有足够的GPU/CPU内存可用。在transformers中使用`device_map="auto"`将有助于自动内存管理。在整个过程中保持一致的精度（例如，float16），以匹配训练期间使用的精度，并以相同的格式保存合并后的模型以便部署。在部署之前，通过比较合并模型的输出和性能指标与基于适配器的版本来进行验证。

适配器也便于在不同任务或领域之间切换。您可以分别加载基础模型和适配器权重。这允许在不同任务特定的权重之间快速切换。

## Implementation Guide

notebooks/目录包含实施不同PEFT（参数高效微调）方法的实用教程和练习。从`load_lora_adapter_example.ipynb`开始，了解基础知识，然后探索`lora_finetuning.ipynb`，更详细地了解如何使用LoRA和SFT微调模型。

在实施PEFT方法时，从较小的秩值（4-8）开始LoRA，并监控训练损失。使用验证集来防止过拟合，并在可能的情况下与完全微调的基线进行比较。不同方法的有效性可能因任务而异，因此实验是关键。

## OLoRA

[OLoRA](https://arxiv.org/abs/2406.01775) 利用QR分解来初始化LoRA适配器。OLoRA通过QR分解的因子来转换模型的基础权重，即在对其进行任何训练之前改变权重。这种方法显著提高了稳定性，加速了收敛速度，并最终实现了更优的性能。

## Using TRL with PEFT

PEFT方法可以与TRL结合使用，以实现高效的微调。这种集成对于RLHF（基于人类反馈的强化学习）特别有用，因为它减少了内存需求。
```python
from peft import LoraConfig
from transformers import AutoModelForCausalLM

# Load model with PEFT config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Load model on specific device
model = AutoModelForCausalLM.from_pretrained(
    "your-model-name",
    load_in_8bit=True,  # Optional: use 8-bit precision
    device_map="auto",
    peft_config=lora_config
)
```
在上面，我们使用了`device_map="auto"`来自动将模型分配给正确的设备。你也可以使用`device_map={"": device_index}`手动将模型分配给特定的设备。此外，你还可以在多个GPU上扩展训练，同时保持内存使用的高效性。

## Basic Merging Implementation

在训练完LoRA适配器之后，你可以将适配器的权重合并回基础模型中。以下是操作方法：

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    "base_model_name",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. Load the PEFT model with adapter
peft_model = PeftModel.from_pretrained(
    base_model,
    "path/to/adapter",
    torch_dtype=torch.float16
)

# 3. Merge adapter weights with base model
try:
    merged_model = peft_model.merge_and_unload()
except RuntimeError as e:
    print(f"Merging failed: {e}")
    # Implement fallback strategy or memory optimization

# 4. Save the merged model
merged_model.save_pretrained("path/to/save/merged_model")
```

If you encounter size discrepancies in the saved model, ensure you're also saving the tokenizer:

```python
# Save both model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("base_model_name")
merged_model.save_pretrained("path/to/save/merged_model")
tokenizer.save_pretrained("path/to/save/merged_model")
```

## Next Steps

⏩ 继续阅读[Prompt Tuning](prompt_tuning.md)指南，了解如何使用提示调整（Prompt Tuning）对模型进行微调。
⏩ 继续学习[Load LoRA Adapters Tutorial](./notebooks/load_lora_adapter.ipynb)教程，了解如何加载LoRA适配器。

# Resources

- [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Hugging Face blog post on PEFT](https://huggingface.co/blog/peft)
