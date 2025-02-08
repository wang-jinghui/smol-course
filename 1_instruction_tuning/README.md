# Instruction Tuning

本模块将指导您进行指令调优语言模型的操作。指令调优涉及通过对特定任务的数据集进行进一步训练，使预训练模型适应特定任务。这一过程有助于模型在目标任务上提高性能。

在本模块中，我们将探讨两个主题：1) 聊天模板、2) 监督微调。

## 1️⃣ Chat Templates

聊天模板构建了用户与AI模型之间的交互结构，确保回复的一致性和上下文恰当性。它们包括系统提示和基于角色的消息等组件。更多详细信息，请参阅聊天模板[Chat Templates](./chat_templates.md)。

## 2️⃣ Supervised Fine-Tuning

监督微调（SFT）是将预训练语言模型适应特定任务的关键过程。它涉及使用带有标签的示例在特定任务的数据集上对模型进行训练。有关SFT的详细指南，包括关键步骤和最佳实践，请参阅监督微调[Supervised Fine-Tuning](./supervised_fine_tuning.md)。

## Exercise Notebooks

| Title | Description | Exercise | Link | Colab |
|-------|-------------|----------|------|-------|
| Chat Templates | 学习如何使用SmolLM2中的聊天模板，并将数据集处理为chatml格式 | 🐢 Convert the `HuggingFaceTB/smoltalk` dataset into chatml format <br> 🐕 Convert the `openai/gsm8k` dataset into chatml format | [Notebook](./notebooks/chat_templates_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/1_instruction_tuning/notebooks/chat_templates_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Supervised Fine-Tuning | 学习如何使用SFTTrainer微调SmolLM2 | 🐢 Use the `HuggingFaceTB/smoltalk` dataset<br>🐕 Try out the `bigcode/the-stack-smol` dataset<br>🦁 Select a dataset for a real world use case | [Notebook](./notebooks/sft_finetuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/1_instruction_tuning/notebooks/sft_finetuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

## References

- [Transformers documentation on chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Script for Supervised Fine-Tuning in TRL](https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py)
- [`SFTTrainer` in TRL](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [Direct Preference Optimization Paper](https://arxiv.org/abs/2305.18290)
- [Supervised Fine-Tuning with TRL](https://huggingface.co/docs/trl/main/en/tutorials/supervised_finetuning)
- [How to fine-tune Google Gemma with ChatML and Hugging Face TRL](https://www.philschmid.de/fine-tune-google-gemma)
- [Fine-tuning LLM to Generate Persian Product Catalogs in JSON Format](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_to_generate_persian_product_catalogs_in_json_format)
