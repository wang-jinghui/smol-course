# Evaluation

评估是开发和部署语言模型中的一个关键步骤。它有助于我们了解模型在不同能力上的表现情况，并识别出需要改进的领域。本模块既包含标准基准测试，也涵盖特定领域的评估方法，以全面评估您的smol模型。

我们将使用[`lighteval`](https://github.com/huggingface/lighteval)，这是Hugging Face开发的一款强大的评估库，它与Hugging Face生态系统无缝集成。想要深入了解评估概念和最佳实践，请查阅评估[guidebook](https://github.com/huggingface/evaluation-guidebook)。

## Module Overview 

一个全面的评估策略会考察模型性能的多个方面。我们评估特定任务的能力，如问答和摘要生成，以了解模型处理不同类型问题的能力。我们通过连贯性和事实准确性等因素来衡量输出质量。安全性评估有助于识别潜在的有害输出或偏见。最后，领域专业知识测试验证了模型在目标领域中的专业知识。


### 1️⃣ [Automatic Benchmarks](./automatic_benchmarks.md)
学习使用标准化基准和指标来评估您的模型。我们将探讨常见的基准，如`MMLU`和`TruthfulQA`，理解关键的评估指标和设置，并介绍可重复评估的最佳实践。

### 2️⃣ [Custom Domain Evaluation](./custom_evaluation.md)
了解如何创建适合您特定用例的评估流程。我们将逐步指导您设计自定义评估任务、实现专门指标，以及构建符合您需求的评估数据集。

### 3️⃣ [Domain Evaluation Project](./project/README.md)
跟随一个完整的构建特定领域评估流程的例子。您将学习如何生成评估数据集，使用`Argilla`进行数据标注，创建标准化数据集，以及使用`LightEval`评估模型。

### Exercise 
- 🐢 使用医学领域任务来评估模型
- 🐕 使用不同的MMLU任务创建一个新的领域评估
- 🦁 为您的领域创建一个自定义评估任务

## Resources

- [Evaluation Guidebook](https://github.com/huggingface/evaluation-guidebook) - Comprehensive guide to LLM evaluation
- [LightEval Documentation](https://github.com/huggingface/lighteval) - Official docs for the LightEval library
- [Argilla Documentation](https://docs.argilla.io) - Learn about the Argilla annotation platform
- [MMLU Paper](https://arxiv.org/abs/2009.03300) - Paper describing the MMLU benchmark
- [Creating a Custom Task](https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task)
- [Creating a Custom Metric](https://github.com/huggingface/lighteval/wiki/Adding-a-New-Metric)
- [Using existing metrics](https://github.com/huggingface/lighteval/wiki/Metric-List)