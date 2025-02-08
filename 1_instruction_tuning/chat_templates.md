# Chat Templates

聊天模板对于构建语言模型与用户之间的交互至关重要。它们为对话提供了一种一致的格式，确保模型能够理解每条消息的上下文和角色，同时保持适当的回应模式。

## Base Models vs Instruct Models

基础模型是在原始文本数据上进行训练，以预测下一个标记，而指令模型则经过专门微调，以遵循指令并参与对话。例如，SmolLM2-135M是一个基础模型，而SmolLM2-135M-Instruct则是其经过指令调优的变体。

为了使基础模型表现得像指令模型，我们需要以模型能够理解的一致方式格式化我们的提示。这就是聊天模板发挥作用的地方。ChatML是这样一种模板格式，它使用明确的角色指示符（系统、用户、助手）来结构化对话。

重要的是要注意，基础模型可以在不同的聊天模板上进行微调，因此，当我们使用指令模型时，我们需要确保我们使用的是正确的聊天模板。

## Understanding Chat Templates

从根本上讲，聊天模板定义了在与语言模型通信时对话的格式。它们以模型能够理解的结构化格式包含系统级指令、用户消息和助手回复。这种结构有助于保持交互的一致性，并确保模型对不同类型的输入做出适当的回应。下面是一个聊天模板的示例：

```sh
<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
<|im_start|>assistant
```

`transformers`库会为您处理与模型分词器相关的聊天模板。有关transformers如何构建聊天模板的更多信息，请点击此处[here](https://huggingface.co/docs/transformers/en/chat_templating#how-do-i-use-chat-templates)查阅。我们只需要以正确的方式构建消息，分词器会处理其余部分。以下是一个对话的基本示例：

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant focused on technical topics."},
    {"role": "user", "content": "Can you explain what a chat template is?"},
    {"role": "assistant", "content": "A chat template structures conversations between users and AI models..."}
]
```

让我们分解上面的示例，看看它是如何映射到聊天模板格式的。

## System Messages

系统消息为模型的行为方式奠定了基础。它们作为持久指令，影响所有后续交互。例如：

```python
system_message = {
    "role": "system",
    "content": "You are a professional customer service agent. Always be polite, clear, and helpful."
}
```

## Conversations

聊天模板通过对话历史记录维持上下文，存储用户和助手之间的先前交流。这有助于实现更加连贯的多轮对话：

```python
conversation = [
    {"role": "user", "content": "I need help with my order"},
    {"role": "assistant", "content": "I'd be happy to help. Could you provide your order number?"},
    {"role": "user", "content": "It's ORDER-123"},
]
```

## Implementation with Transformers

transformers库为聊天模板提供了内置支持。以下是使用方法：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a Python function to sort a list"},
]

# Apply the chat template
formatted_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

## Custom Formatting

您可以自定义不同消息类型的格式。例如，为不同角色添加特殊标记或格式：

```python
template = """
<|system|>{system_message}
<|user|>{user_message}
<|assistant|>{assistant_message}
""".lstrip()
```

## Multi-Turn Support

模板可以在保持上下文的同时处理复杂的多轮对话：

```python
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is calculus?"},
    {"role": "assistant", "content": "Calculus is a branch of mathematics..."},
    {"role": "user", "content": "Can you give me an example?"},
]
```

⏭️ [Next: Supervised Fine-Tuning](./supervised_fine_tuning.md)

## Resources

- [Hugging Face Chat Templating Guide](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Chat Templates Examples Repository](https://github.com/chujiezheng/chat_templates) 
