# Direct Preference Optimization (DPO)

提供了一种将语言模型与人类偏好进行对齐的简化方法。与需要单独奖励模型和复杂强化学习的传统RLHF方法不同，DPO直接使用偏好数据来优化模型。

## Understanding DPO

DPO将偏好对齐重新定义为一个人类偏好数据上的分类问题。传统的RLHF方法需要训练一个单独的奖励模型，并使用像PPO这样的复杂强化学习算法来对齐模型输出。DPO通过定义一个损失函数来简化这个过程，该损失函数基于偏好输出与非偏好输出直接优化模型的策略。

这种方法在实践中已被证明非常有效，被用于训练如Llama等模型。通过消除对单独奖励模型和强化学习阶段的需求，DPO使得偏好对齐更加可行和稳定。

## How DPO Works

DPO过程需要进行监督微调（SFT），以适应目标领域。这通过在标准指令遵循数据集上进行训练，为偏好学习奠定基础。模型在学习基本任务完成的同时，保持其通用能力。

接下来是偏好学习阶段，模型在成对输出上进行训练——一个优选输出和一个非优选输出。这些偏好对帮助模型理解哪些响应更好地与人类价值观和期望对齐。

DPO的核心创新在于其直接优化方法。DPO不使用单独的奖励模型，而是使用二元交叉熵损失来根据偏好数据直接更新模型权重。这个简化的过程使得训练更加稳定和高效，同时实现了与传统RLHF相当或更好的结果。

## DPO datasets

DPO的数据集通常是通过将成对响应标注为优选或非优选来创建的。这可以手动完成，也可以使用自动化过滤技术。以下是DPO偏好数据集的一个示例结构：

| Prompt | Chosen | Rejected |
|--------|--------|----------|
| ...    | ...    | ...      |
| ...    | ...    | ...      |
| ...    | ...    | ...      |

`Prompt`列包含了用于生成`Chosen`（优选）和`Rejected`（非优选）响应的提示。`Chosen`和`Rejected`列分别包含了优选和非优选的响应。这种结构有多种变体，例如，可以包括一个系统提示列或包含参考资料的Input（输入）列。`chosen`和`rejected`的值可以表示为单轮对话的字符串，也可以表示为对话列表。

你可以在Hugging Face上找到一系列DPO数据集。[here](https://huggingface.co/collections/argilla/preference-datasets-for-dpo-656f0ce6a00ad2dc33069478).

## Implementation with TRL

Transformers强化学习（TRL）库使得实现直接偏好优化（DPO）变得简单直接。DPOConfig和DPOTrainer类遵循了与transformers库相同的API风格。

以下是一个设置DPO训练的基本示例：

```python
from trl import DPOConfig, DPOTrainer

# Define arguments
training_args = DPOConfig(
    ...
)

# Initialize trainer
trainer = DPOTrainer(
    model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    ...
)

# Train model
trainer.train()
```
我们将在[DPO Tutorial](./notebooks/dpo_finetuning_example.ipynb).中详细介绍如何使用DPOConfig和DPOTrainer类。

## Best Practices

数据质量对于成功实施直接偏好优化（DPO）至关重要。偏好数据集应包含涵盖期望行为不同方面的多样化示例。明确的标注指南可确保对优选和非优选响应进行一致标注。通过提高偏好数据集的质量，可以改进模型性能。例如，通过筛选较大的数据集，仅包含高质量示例或与您的用例相关的示例。

在训练过程中，仔细监控损失收敛情况，并在留出数据上验证性能。可能需要调整beta参数，以在偏好学习与保持模型通用能力之间取得平衡。在多样化的提示上进行定期评估，有助于确保模型在学习预期偏好的同时不会出现过拟合。

将模型的输出与参考模型进行比较，以验证偏好对齐方面的改进。在各种提示（包括边缘情况）上进行测试，有助于确保在不同场景下都能实现稳健的偏好学习。

## Next Steps

⏩ 要亲身体验DPO，请尝试DPO教程[DPO Tutorial](./notebooks/dpo_finetuning_example.ipynb)。本实践指南将指导您使用自己的模型从数据准备到训练和评估，实现偏好对齐。

⏭️ 完成教程后，您可以浏览ORPO页面，了解另一种偏好对齐技术。[ORPO](./orpo.md) 