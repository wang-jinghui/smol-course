# 自定义领域评估

虽然标准基准测试提供了有价值的见解，但许多应用需要针对特定领域或用例量身定制的专门评估方法。本指南将帮助您创建自定义评估流程，以准确评估您的模型在目标领域中的性能。

## 设计您的评估策略

成功的自定义评估策略始于明确的目标。考虑您的模型需要在您的领域中展示哪些特定能力。这可能包括技术知识、推理模式或领域特定的格式。仔细记录这些要求——它们将指导您的任务设计和指标选择。

您的评估应测试标准用例和边缘用例。例如，在医疗领域，您可能需要评估常见的诊断场景和罕见病症。在金融应用中，您可能需要测试常规交易和涉及多种货币或特殊条件的复杂边缘用例。

## 使用LightEval进行实现

LightEval提供了一个灵活的框架来实现自定义评估。以下是创建自定义任务的方法：

```python
from lighteval.tasks import Task, Doc
from lighteval.metrics import SampleLevelMetric, MetricCategory, MetricUseCase

class CustomEvalTask(Task):
    def __init__(self):
        super().__init__(
            name="custom_task",
            version="0.0.1",
            metrics=["accuracy", "f1"],  # Your chosen metrics
            description="Description of your custom evaluation task"
        )
    
    def get_prompt(self, sample):
        # Format your input into a prompt
        return f"Question: {sample['question']}\nAnswer:"
    
    def process_response(self, response, ref):
        # Process model output and compare to reference
        return response.strip() == ref.strip()
```

## 自定义指标

特定领域的任务通常需要专门的指标。LightEval提供了一个灵活的框架，用于创建能够捕捉与领域相关性能方面的自定义指标：

```python
from aenum import extend_enum
from lighteval.metrics import Metrics, SampleLevelMetric, SampleLevelMetricGrouping
import numpy as np

# Define a sample-level metric function
def custom_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> dict:
    """Example metric that returns multiple scores per sample"""
    response = predictions[0]
    return {
        "accuracy": response == formatted_doc.choices[formatted_doc.gold_index],
        "length_match": len(response) == len(formatted_doc.reference)
    }

# Create a metric that returns multiple values per sample
custom_metric_group = SampleLevelMetricGrouping(
    metric_name=["accuracy", "length_match"],  # Names of sub-metrics
    higher_is_better={  # Whether higher values are better for each metric
        "accuracy": True,
        "length_match": True
    },
    category=MetricCategory.CUSTOM,
    use_case=MetricUseCase.SCORING,
    sample_level_fn=custom_metric,
    corpus_level_fn={  # How to aggregate each metric
        "accuracy": np.mean,
        "length_match": np.mean
    }
)

# Register the metric with LightEval
extend_enum(Metrics, "custom_metric_name", custom_metric_group)
```
对于更简单的情况，即您每个样本只需要一个指标值的情况：

```python
def simple_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> bool:
    """Example metric that returns a single score per sample"""
    response = predictions[0]
    return response == formatted_doc.choices[formatted_doc.gold_index]

simple_metric_obj = SampleLevelMetric(
    metric_name="simple_accuracy",
    higher_is_better=True,
    category=MetricCategory.CUSTOM,
    use_case=MetricUseCase.SCORING,
    sample_level_fn=simple_metric,
    corpus_level_fn=np.mean  # How to aggregate across samples
)

extend_enum(Metrics, "simple_metric", simple_metric_obj)
```
然后，您可以在任务配置中引用自定义指标，在评估任务中使用它们。指标将自动在所有样本上计算，并根据您指定的函数进行聚合。

对于更复杂的指标，请考虑：
- 使用格式化文档中的元数据为分数加权或调整分数
- 为语料库级别的统计信息实现自定义聚合函数
- 为指标输入添加验证检查
- 记录边缘案例和预期行为

要了解自定义指标在实际应用中的完整示例，请参阅[domain evaluation project](./project/README.md)。

## 数据集创建
高质量的评估需要精心策划的数据集。以下是数据集创建的几种方法：
1. 专家标注：与领域专家合作，创建并验证评估示例。像[Argilla](https://github.com/argilla-io/argilla)这样的工具可以提高这一过程的效率。
2. 真实世界数据：收集并匿名化处理实际使用数据，确保其能代表实际的部署场景。
3. 合成生成：使用大型语言模型（LLMs）生成初始示例，然后由专家进行验证和完善。

## 最佳实践
- 彻底记录您的评估方法，包括任何假设或限制
- 包含覆盖您领域不同方面的多样化测试用例
- 在适当的情况下，同时考虑自动指标和人工评估
- 对评估数据集和代码进行版本控制
- 随着发现新的边缘案例或需求，定期更新评估套件

# 下一步

⏩ 要了解实施这些概念的完整示例，请参阅我们的[domain evaluation project](./project/README.md)。

# Resources

- [LightEval Custom Task Guide](https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task)
- [LightEval Custom Metrics](https://github.com/huggingface/lighteval/wiki/Adding-a-New-Metric)
- [Argilla Documentation](https://docs.argilla.io) for dataset annotation
- [Evaluation Guidebook](https://github.com/huggingface/evaluation-guidebook) for general evaluation principles