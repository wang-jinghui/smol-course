# Odds Ratio Preference Optimization (ORPO)

ORPO（比值比偏好优化）是一种新颖的微调技术，它将微调与偏好对齐整合为一个统一的流程。与传统方法（如RLHF或DPO）相比，这种综合方法在效率和性能上具有优势。

## Understanding ORPO

与DPO等方法的对齐通常涉及两个独立步骤：首先是监督微调，使模型适应特定领域和格式；其次是偏好对齐，以符合人类偏好。虽然监督微调（SFT）能有效使模型适应目标领域，但也可能无意中增加了生成理想和非理想响应的概率。ORPO通过将这两个步骤整合为一个流程来解决这一局限，如下面的比较所示：

![Alignment Techniques Comparison](https://argilla.io/images/blog/mantisnlp-rlhf/part-8-alignments.png)
*不同模型对齐技术的比较*

## How ORPO Works

训练过程利用了一个与DPO所使用的类似的偏好数据集，其中每个训练示例都包含一个输入提示以及两个响应：一个优选响应和一个被拒绝响应。与其他需要分阶段进行和参考模型的对齐方法不同，ORPO将偏好对齐直接整合到了监督微调过程中。这种方法计算效率更高，内存效率也更高，浮点运算次数（FLOPs）更少。

ORPO通过结合两个主要组件来创建一个新的目标：

1.**SFT损失**：语言建模中使用的标准负对数似然损失，它最大化生成参考标记的概率。这有助于保持模型的通用语言能力。
2.**比值比损失**：一个新颖组件，它对非理想响应进行惩罚，同时奖励优选响应。这个损失函数使用比值比在标记级别上有效对比优选和非优选响应。

这两个组件共同引导模型适应特定领域的理想生成，同时积极避免生成被拒绝响应集中的内容。比值比机制为测量和优化模型在优选和被拒绝输出之间的偏好提供了一种自然方式。如果您想深入了解其数学原理，可以阅读ORPO论文[ORPO paper](https://arxiv.org/abs/2403.07691)。如果您想从实现角度了解ORPO，应该查看TRL库中ORPO损失的计算方式[TRL library](https://github.com/huggingface/trl/blob/b02189aaa538f3a95f6abb0ab46c0a971bfde57e/trl/trainer/orpo_trainer.py#L660)。

## Performance and Results

ORPO在各种基准测试中取得了令人印象深刻的结果。在MT-Bench上，它在不同类别中获得了具有竞争力的分数：

![MT-Bench Results](https://argilla.io/images/blog/mantisnlp-rlhf/part-8-mtbench.png)
*Mistral-ORPO模型在MT-Bench上的各类别结果*

与其他对齐方法相比，ORPO在AlpacaEval 2.0上表现出色：

![AlpacaEval Results](https://argilla.io/images/blog/mantisnlp-rlhf/part-8-winrate.png)
*不同对齐方法在AlpacaEval 2.0上的分数*

与SFT+DPO相比，ORPO通过消除对reference model的需求并将每批的前向传递次数减半，降低了计算要求。此外，训练过程在不同模型大小和数据集上更加稳定，需要调整的超参数更少。在性能方面，ORPO与更大模型相当，同时与人类偏好的对齐程度更高。

## Implementation 

ORPO的成功实现很大程度上依赖于高质量的偏好数据。训练数据应遵循明确的标注指南，并在不同场景下提供优选和被拒绝响应的平衡表示。

### Implementation with TRL

可以使用Transformers强化学习（TRL）库来实现ORPO。以下是一个基本示例：

```python
from trl import ORPOConfig, ORPOTrainer

# Configure ORPO training
orpo_config = ORPOConfig(
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_steps=1000,
    orpo_alpha=1.0,  # Controls strength of preference optimization
    orpo_beta=0.1,   # Temperature parameter for odds ratio
)

# Initialize trainer
trainer = ORPOTrainer(
    model=model,
    args=orpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# Start training
trainer.train()
```
关键参数：
- orpo_alpha：控制偏好优化的强度
- orpo_beta：用于计算赔率比的温度参数
- learning_rate（学习率）：应相对较小，以防止灾难性遗忘
- gradient_accumulation_steps（梯度累积步长）：有助于训练稳定性

## Next Steps

⏩ Try the [ORPO Tutorial](./notebooks/orpo_finetuning_example.ipynb).

## Resources
- [ORPO Paper](https://arxiv.org/abs/2403.07691)
- [TRL Documentation](https://huggingface.co/docs/trl/index)
- [Argilla RLHF Guide](https://argilla.io/blog/mantisnlp-rlhf-part-8/) 