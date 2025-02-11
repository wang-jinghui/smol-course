# Prompt Tuning

提示调优是一种参数高效的方法，它通过修改输入表示而不是模型权重来进行优化。与传统的微调方法不同，传统微调会更新所有模型参数，而提示调优则是在保持基础模型不变的情况下，添加并优化一组小型可训练标记。

## Understanding Prompt Tuning

提示调优是模型微调的一种参数高效的替代方案，它在输入文本前添加可训练连续向量（soft prompts）。与离散文本提示不同，这些软提示是在保持语言模型不变的情况下通过反向传播学习的。该方法在["The Power of Scale for Parameter-Efficient Prompt Tuning"](https://arxiv.org/abs/2104.08691)一文中被提出，该研究表明，随着模型规模的增加，提示调优在竞争力上逐渐接近模型微调。在该论文中，当模型参数达到约100亿时，提示调优在每个任务仅修改几百个参数的情况下，其性能可与模型微调相媲美。

这些软提示（soft prompts）是模型嵌入空间中的连续向量，在训练过程中得到优化。与使用自然语言标记的传统离散提示不同，软提示本身没有固有含义，但通过梯度下降学习从冻结模型中激发所需行为。该技术对于多任务场景特别有效，因为每个任务只需存储一个小型提示向量（通常为几百个参数），而不是完整的模型副本。这种方法不仅保持了最小的内存占用，还通过简单交换提示向量（无需重新加载模型）实现了快速任务切换。

## Training Process

软提示通常包含8到32个标记，可以随机初始化或从现有文本中初始化。初始化方法在训练过程中起着至关重要的作用，基于文本的初始化通常比随机初始化表现更好。

在训练过程中，只有提示参数被更新，而基础模型保持不变。这种有针对性的方法使用标准的训练目标，但需要仔细关注提示标记的学习率和梯度行为。

## Implementation with PEFT

PEFT库使得实现提示调优变得直接明了。以下是一个基本示例：
```python
from peft import PromptTuningConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
model = AutoModelForCausalLM.from_pretrained("your-base-model")
tokenizer = AutoTokenizer.from_pretrained("your-base-model")

# Configure prompt tuning
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=8,  # Number of trainable tokens
    prompt_tuning_init="TEXT",  # Initialize from text
    prompt_tuning_init_text="Classify if this text is positive or negative:",
    tokenizer_name_or_path="your-base-model",
)

# Create prompt-tunable model
model = get_peft_model(model, peft_config)
```

## Comparison to Other Methods

与其他参数高效微调（PEFT）方法相比，提示调优以其高效性脱颖而出。虽然LoRA提供了较低的参数数量和内存使用量，但需要加载适配器以实现任务切换，而提示调优则实现了更低的资源使用，并允许立即进行任务切换。相比之下，全面微调需要大量资源，并且需要为不同任务分别复制模型。

| Method | Parameters | Memory | Task Switching |
|--------|------------|---------|----------------|
| Prompt Tuning | Very Low | Minimal | Easy |
| LoRA | Low | Low | Requires Loading |
| Full Fine-tuning | High | High | New Model Copy |

在实施提示调优时，先从少量的虚拟标记（8-16个）开始，只有当任务复杂度要求时才增加。与随机初始化相比，使用文本初始化通常能取得更好的结果，尤其是在使用与任务相关的文本时。初始化策略应反映目标任务的复杂度。

训练需要考虑的因素与全面微调略有不同。较高的学习率通常能取得良好效果，但仔细监控提示标记的梯度至关重要。对不同示例进行定期验证有助于确保在不同场景下都能获得稳健的性能。
## Application

提示调优在以下几个场景中表现出色：

- 多任务部署
- 资源受限环境
- 快速任务适应
- 隐私敏感型应用

随着模型变小，提示调优与全面微调相比竞争力减弱。例如，在SmolLM2等规模的模型上，提示调优的重要性不如全面微调。

## Next Steps

⏭️ 继续学习LoRA适配器教程[LoRA Adapters Tutorial](./notebooks/finetune_sft_peft.ipynb)，了解如何使用LoRA适配器对模型进行微调。

## Resources
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Prompt Tuning Paper](https://arxiv.org/abs/2104.08691)
- [Hugging Face Cookbook](https://huggingface.co/learn/cookbook/prompt_tuning_peft)
