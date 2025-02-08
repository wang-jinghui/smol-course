# Supervised Fine-Tuning

监督微调（SFT）是将预训练语言模型适应特定任务或领域的关键过程。虽然预训练模型具有令人印象深刻的通用能力，但它们通常需要进行定制才能在特定用例中表现出色。监督微调通过在人工验证的精心策划的数据集上进一步训练模型来弥合这一差距。

## Understanding Supervised Fine-Tuning

其核心在于，监督微调是通过标记令牌的示例来教授预训练模型执行特定任务。该过程涉及向模型展示许多期望的输入输出行为示例，使其学习到您用例中特有的模式。

监督微调之所以有效，是因为它在适应模型行为以满足您的特定需求的同时，利用了预训练期间获得的基础知识。

## When to Use Supervised Fine-Tuning

使用监督微调（SFT）的决定通常取决于您模型的当前能力与您的特定需求之间的差距。当您需要精确控制模型的输出或在专业领域工作时，监督微调变得尤为重要。

例如，如果您正在开发一个客户服务应用程序，您可能希望您的模型始终遵循公司准则并以标准化的方式处理技术查询。同样，在医疗或法律应用中，准确性和遵循特定领域的术语变得至关重要。在这些情况下，监督微调可以帮助模型的回复与专业标准和领域专业知识保持一致。

## The Fine-Tuning Process

监督微调过程涉及在特定任务的数据集上调整模型的权重。

首先，您需要准备或选择一个代表目标任务的数据集。该数据集应包含涵盖模型将遇到的各种场景的多样化示例。数据的质量很重要——每个示例都应展示您希望模型生成的输出类型。接下来是实际的微调阶段，您将使用如Hugging Face的`transformers`和`trl`等框架在数据集上训练模型。

在整个过程中，持续评估至关重要。您需要监控模型在验证集上的表现，以确保其正在学习所需行为而不会丧失通用能力。在模块[module 4](../4_evaluation)中，我们将介绍如何评估您的模型。

## The Role of SFT in Preference Alignment

监督微调（SFT）在使语言模型与人类偏好保持一致方面发挥着基础性作用。诸如来自人类反馈的强化学习（RLHF）和直接偏好优化（DPO）等技术，依赖于监督微调来形成任务理解的基础水平，然后再进一步使模型的响应与期望结果保持一致。尽管预训练模型具有通用语言熟练度，但它们并不总是生成与人类偏好相匹配的输出。监督微调通过引入特定领域的数据和指导来弥合这一差距，从而提高了模型生成与人类期望更一致的响应的能力。

## Supervised Fine-Tuning With Transformer Reinforcement Learning

TRL是一个使用强化学习（RL）训练Transformer语言模型的工具包。

TRL建立在Hugging Face `Transformers`库之上，允许用户直接加载预训练的语言模型，并支持大多数decoder 和 encoder-decoder架构。该库简化了语言建模中使用的强化学习的主要过程，包括监督微调（SFT）、奖励建模（RM）、近端策略优化（PPO）和直接偏好优化（DPO）。我们将在本存储库的多个模块中使用TRL。

# Next Steps

尝试以下教程，使用TRL亲身体验监督微调（SFT）：

⏭️ [Chat Templates Tutorial](./notebooks/chat_templates_example.ipynb)

⏭️ [Supervised Fine-Tuning Tutorial](./notebooks/sft_finetuning_example.ipynb)
