# Parameter-Efficient Fine-Tuning (PEFT)

随着语言模型的规模不断扩大，传统的微调变得越来越具有挑战性。即便是对一个拥有17亿参数的模型进行全面微调，也需要大量的GPU内存，存储独立的模型副本成本高昂，而且还存在模型原始能力灾难性遗忘的风险。参数高效微调（PEFT）方法通过仅修改模型参数的一小部分子集，同时保持模型的大部分参数不变，来解决这些挑战。

传统微调在训练过程中会更新所有模型参数，这对于大型模型来说是不切实际的。PEFT方法引入了使用更少可训练参数来适应模型的方法，这些参数通常少于原始模型大小的1%。可训练参数的显著减少使得：

- 在GPU内存有限的消费级硬件上进行微调成为可能
- 能够高效地存储多个针对特定任务的模型调整版本
- 在数据稀缺的场景下具有更好的泛化能力
- 训练和迭代周期更快”

## Available Methods

在这个模块中，我们将介绍两种流行的参数高效微调（PEFT）方法：

### 1️⃣ LoRA（低秩适配）

LoRA已成为最广泛采用的PEFT方法，为高效模型适配提供了一种优雅的解决方案。LoRA不是修改整个模型，而是在模型的注意力层中注入可训练的矩阵。这种方法通常能将可训练参数减少约90%，同时保持与全面微调相当的性能。我们将在[LoRA (Low-Rank Adaptation)](./lora_adapters.md)部分探讨LoRA。 
 
### 2️⃣ Prompt Tuning

提示调优通过向输入中添加**可训练标记**而不是修改**模型权重**，提供了一种更**轻量级**的方法。提示调优不如LoRA流行，但可以作为快速使模型适应新任务或领域的有用技术。我们将在提示[Prompt Tuning](./prompt_tuning.md)部分探讨提示调优。


## Exercise Notebooks

| Title | Description | Exercise | Link | Colab |
|-------|-------------|----------|------|-------|
| LoRA Fine-tuning | 学习如何使用LoRA适配器微调模型 | 🐢 使用LoRA训练模型<br>🐕 尝试不同的秩值进行实验<br>🦁 与全面微调的性能进行比较 | [Notebook](./notebooks/finetune_sft_peft.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/3_parameter_efficient_finetuning/notebooks/finetune_sft_peft.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Load LoRA Adapters | 学习如何加载和使用已训练的LoRA适配器 | 🐢 加载预训练的适配器<br>🐕 将适配器与基础模型合并<br>🦁 在多个适配器之间切换 | [Notebook](./notebooks/load_lora_adapter.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/3_parameter_efficient_finetuning/notebooks/load_lora_adapter.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
<!-- | Prompt Tuning | Learn how to implement prompt tuning | 🐢 Train soft prompts<br>🐕 Compare different initialization strategies<br>🦁 Evaluate on multiple tasks | [Notebook](./notebooks/prompt_tuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/3_parameter_efficient_finetuning/notebooks/prompt_tuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | -->

## Resources
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Prompt Tuning Paper](https://arxiv.org/abs/2104.08691)
- [Hugging Face PEFT Guide](https://huggingface.co/blog/peft)
- [How to Fine-Tune LLMs in 2024 with Hugging Face](https://www.philschmid.de/fine-tune-llms-in-2024-with-trl) 
- [TRL](https://huggingface.co/docs/trl/index)
