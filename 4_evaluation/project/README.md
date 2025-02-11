# 使用Argilla、Distilabel和LightEval进行领域特定评估

大多数流行的基准测试都关注非常通用的能力（如推理、数学、代码），但你是否有过研究更具体能力的需求？

如果你需要在与你的用例相关的自定义领域中评估模型，你应该怎么做？（例如，金融、法律、医疗用例）

本教程将向你展示你可以遵循的完整流程，从创建相关数据、标注样本到在这些样本上评估模型，这一切都可以通过使用易于操作的[Argilla](https://github.com/argilla-io/argilla)、[distilabel](https://github.com/argilla-io/distilabel)和[lighteval](https://github.com/huggingface/lighteval)来完成。在我们的示例中，我们将重点关注从多个文档中生成考试题目。

## 项目结构

我们的流程将遵循四个步骤，每个步骤都有一个相应的脚本：生成数据集、标注数据集、从中提取用于评估的相关样本，以及实际评估模型。

| 脚本名称 | 描述 |
|-------------|-------------|
| generate_dataset.py | 使用指定的语言模型从多个文本文档中生成考试题目。 |
| annotate_dataset.py | 创建一个Argilla数据集，用于手动标注生成的考试题目。 |
| create_dataset.py | 处理来自Argilla的标注数据，并创建一个Hugging Face数据集。 |
| evaluation_task.py | 定义一个自定义的LightEval任务，用于在考试题目数据集上评估语言模型。 |

## 步骤

### 1. 生成数据集
`generate_dataset.py`脚本使用 distilabel 库根据多个文本文档生成考试题目。它使用指定的模型（默认为 Meta-Llama-3.1-8B-Instruct）来创建问题、正确答案和错误答案（称为干扰项）。您应该添加自己的数据样本，并且可能希望使用不同的模型

运行生成：

```sh
python generate_dataset.py --input_dir path/to/your/documents --model_id your_model_id --output_path output_directory
```

这将创建一个[Distiset](https://distilabel.argilla.io/dev/sections/how_to_guides/advanced/distiset/)，其中包含输入目录中所有文档的生成考试题目。

### 2. 标注数据集

`annotate_dataset.py`脚本会接收生成的问题，并创建一个用于标注的Argilla数据集。它会设置数据集的结构，并用生成的问题和答案填充它，同时随机化答案的顺序以避免偏见。一旦数据进入Argilla，您或领域专家就可以使用正确答案验证数据集。

您将看到来自大型语言模型（LLM）的建议正确答案，这些答案以随机顺序显示，您可以批准正确答案或选择另一个答案。此过程的持续时间将取决于您的评估数据集的规模、领域数据的复杂性以及LLM的质量。例如，在使用Llama-3.1-70B-Instruct的迁移学习领域，我们主要通过批准正确答案并丢弃错误答案，在1小时内创建了150个样本。

要运行标注过程：

```sh
python annotate_dataset.py --dataset_path path/to/distiset --output_dataset_name argilla_dataset_name
```
这将创建一个Argilla数据集，可用于手动审查和标注。

![argilla_dataset](./images/domain_eval_argilla_view.png)

如果您尚未使用Argilla，请按照此指南在本地或Spaces上进行部署[quickstart guide](https://docs.argilla.io/latest/getting_started/quickstart/)。

### 3. 创建数据集

`create_dataset.py`脚本会处理来自Argilla的已标注数据，并创建一个Hugging Face数据集。它能够处理建议的答案和手动标注的答案。该脚本将创建一个包含问题、可能的答案以及正确答案列名的数据集。要创建最终的数据集，请执行以下操作：

```sh
huggingface_hub login
python create_dataset.py --dataset_path argilla_dataset_name --dataset_repo_id your_hf_repo_id
```
这将把数据集推送到指定的Hugging Face Hub存储库中。您可以在此处[here](https://huggingface.co/datasets/burtenshaw/exam_questions/viewer/default/train)查看Hub上的示例数据集，数据集的预览效果如下：

![hf_dataset](./images/domain_eval_dataset_viewer.png)

### 4. 评估任务

`evaluation_task.py`脚本定义了一个自定义的LightEval任务，用于在考试问题数据集上评估语言模型。它包括一个提示函数、一个自定义的准确性指标以及任务配置。

要使用lighteval和自定义的考试问题任务来评估模型，请执行以下操作：

```sh
lighteval accelerate \
    --model_args "pretrained=HuggingFaceH4/zephyr-7b-beta" \
    --tasks "community|exam_questions|0|0" \
    --custom_tasks domain-eval/evaluation_task.py \
    --output_dir "./evals"
```
在lighteval的wiki中，您可以找到关于这些步骤的详细指南：
- [Creating a Custom Task](https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task)
- [Creating a Custom Metric](https://github.com/huggingface/lighteval/wiki/Adding-a-New-Metric)
- [Using existing metrics](https://github.com/huggingface/lighteval/wiki/Metric-List)