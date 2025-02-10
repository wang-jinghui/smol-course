# Preference Alignment（偏好对齐）

此模块涵盖了将语言模型与人类偏好进行对齐的技术。虽然监督微调有助于模型学习任务，但偏好对齐则鼓励输出与人类期望和价值观相匹配。

## Overview
典型的对齐方法涉及多个阶段：
- 监督微调（SFT）：使模型适应特定领域
- 偏好对齐（如RLHF或DPO）：提高响应质量

像ORPO这样的替代方法将指令调优和偏好对齐结合为一个过程。在这里，我们将重点关注DPO和ORPO算法。

如果您想了解更多关于不同对齐技术的信息，可以在[Argilla Blog](https://argilla.io/blog/mantisnlp-rlhf-part-8)上阅读更多相关内容。

### 1️⃣ Direct Preference Optimization (DPO)

直接偏好优化（DPO）通过直接使用偏好数据优化模型来简化偏好对齐。这种方法无需单独的奖励模型和复杂的强化学习，相比传统的人类反馈强化学习（RLHF）更为稳定和高效。更多详细信息，请参阅直接偏好优化[Direct Preference Optimization (DPO) documentation](./dpo.md)文档。

### 2️⃣ Odds Ratio Preference Optimization (ORPO)

ORPO在单个过程中引入了一种将指令调优和偏好对齐相结合的方法。它通过在标记级别上将负对数似然损失与比率项相结合，修改了标准语言建模目标。该方法具有统一的单阶段训练过程、参考model-free架构以及改进的计算效率。ORPO在各种基准测试中取得了令人印象深刻的结果，在AlpacaEval上的表现优于传统方法。更多详细信息，请参阅比率偏好优化（ORPO）文档[Odds Ratio Preference Optimization (ORPO) documentation](./orpo.md)。

## Exercise Notebooks

| Title | Description | Exercise | Link | Colab |
|-------|-------------|----------|------|-------|
| DPO Training | 学习如何使用直接偏好优化来训练模型 | 🐢 Train a model using the Anthropic HH-RLHF dataset<br>🐕 Use your own preference dataset<br>🦁 Experiment with different preference datasets and model sizes | [Notebook](./notebooks/dpo_finetuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/2_preference_alignment/notebooks/dpo_finetuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| ORPO Training | 学习如何使用比率偏好优化来训练模型 | 🐢 Train a model using instruction and preference data<br>🐕 Experiment with different loss weightings<br>🦁 Compare ORPO results with DPO | [Notebook](./notebooks/orpo_finetuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/2_preference_alignment/notebooks/orpo_finetuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |


## Resources

- [TRL Documentation](https://huggingface.co/docs/trl/index) - Documentation for the Transformers Reinforcement Learning (TRL) library, which implements various alignment techniques including DPO.
- [DPO Paper](https://arxiv.org/abs/2305.18290) - Original research paper introducing Direct Preference Optimization as a simpler alternative to RLHF that directly optimizes language models using preference data.
- [ORPO Paper](https://arxiv.org/abs/2403.07691) - Introduces Odds Ratio Preference Optimization, a novel approach that combines instruction tuning and preference alignment in a single training stage.
- [Argilla RLHF Guide](https://argilla.io/blog/mantisnlp-rlhf-part-8/) - A guide explaining different alignment techniques including RLHF, DPO, and their practical implementations.
- [Blog post on DPO](https://huggingface.co/blog/dpo-trl) - Practical guide on implementing DPO using the TRL library with code examples and best practices.
- [TRL example script on DPO](https://github.com/huggingface/trl/blob/main/examples/scripts/dpo.py) - Complete example script demonstrating how to implement DPO training using the TRL library.
- [TRL example script on ORPO](https://github.com/huggingface/trl/blob/main/examples/scripts/orpo.py) - Reference implementation of ORPO training using the TRL library with detailed configuration options.
- [Hugging Face Alignment Handbook](https://github.com/huggingface/alignment-handbook) - Resource guides and codebase for aligning language models using various techniques including SFT, DPO, and RLHF.
