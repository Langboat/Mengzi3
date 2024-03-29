<div align="left">
<h1>
Mengzi3
</h1>
</div>

<p align="center">
    <img src="./assets/mengzi_logo.png" width="200"/>
<p>

<p align="center">
        🤗 <a href="https://huggingface.co/Langboat">Hugging Face</a> | 🤖 <a href="https://modelscope.cn/organization/Langboat">ModelScope</a> |  <a href="https://wisemodel.cn/organization/Langboat">Wisemodel</a> ｜ 💬 <a href="https://github.com/Langboat/Mengzi3/blob/main/assets/wechat.png">WeChat</a> | <a href="https://www.langboat.com/document/mengzi/mengzi-gpt/call">API</a> | <a href="https://www.langboat.com/portal/mengzi-gpt">孟子GPT</a>
</p>

# 模型介绍/Introduction

本次开源Mengzi3 13B系列模型，模型的地址如下:

The address of the open source Mengzi3 13B series model is as follows:

|    |                                                                                      Mengzi3-13B-Base                                                                                      | Mengzi3-13B-Chat |
| :-: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------: |
| 13B | [🤗](https://huggingface.co/Langboat/Mengzi3-13B-Base) / [🤖](https://modelscope.cn/organization/Langboat/Mengzi3-13B-Base) / [Wisemodel](https://wisemodel.cn/models/Langboat/Mengzi3-13B-Base) |     敬请期待     |

Mengzi3-13B模型基于Llama架构，语料精选自网页、百科、社交、媒体、新闻，以及高质量的开源数据集。通过在万亿tokens上进行多语言语料的继续训练，模型的中文能力突出并且兼顾多语言能力。

Mengzi3-13B is based on the Llama architecture, and the corpus is selected from web pages, encyclopedias, social networking, media, news, and high-quality open source data sets. By continuing to train multilingual corpus on trillions of tokens, the model has outstanding Chinese capabilities and takes into account multilingual capabilities.

# 快速开始/Quickstart

首先进行环境配置，安装项目需要的依赖

First configure the environment and install the dependencies required by the project

```bash
pip install -r requirements.txt
```

简单代码调用：

Simple demo:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Langboat/Mengzi3-13B-Base", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Langboat/Mengzi3-13B-Base", device_map="auto", trust_remote_code=True)
inputs = tokenizer('介绍一下孟子：', return_tensors='pt')
if torch.cuda.is_available():
    inputs = inputs.to('cuda')
pred = model.generate(**inputs, max_new_tokens=512, repetition_penalty=1.1, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(pred[0], skip_special_tokens=True))
```

我们另外提供一个样例代码，可以对基座模型进行单轮的交互推理。

We provide this sample code to perform a single round of interactive reasoning on the base model.

```bash
cd examples
python examples/base_streaming_gen.py --model model_path --tokenizer tokenizer_path
```

# 性能评测/Evaluation

Mengzi3-13B-Base在各项基准测试中与同等参数量大语言模型相比，语言能力成绩领先，数学和编程能力位于前列。

Mengzi3-13B-Base leads in language proficiency and is at the forefront in math and programming proficiency compared to the equivalent large language model in various benchmark tests.

|                            |          MMLU          |          CMMLU          |          OCNLI          | GSM8K | HumanEval |
| :------------------------: | :---------------------: | :---------------------: | :---------------------: | :---: | :-------: |
|     Baichuan2-13B-Base     |          0.530          |          0.489          |          0.433          | 0.528 |   0.171   |
|          Qwen-14B          |          0.589          |          0.539          |          0.550          | 0.613 |   0.323   |
|      ChatGLM3-6B-base      |          0.551          |          0.495          |          0.754          | 0.723 |     -     |
|       InternLM2-20B       |          0.610          |          0.538          |          0.650          | 0.761 |   0.488   |
|      Skywork-13B-base      |          0.557          |          0.524          |          0.426          | 0.558 |     -     |
|       LingoWhale-8B       |          0.541          |          0.495          |          0.352          | 0.550 |   0.329   |
|        DeepSeek-7B        |          0.436          |          0.424          |          0.356          | 0.174 |   0.262   |
|   DeepSeek-MoE-16B-base   |          0.423          |          0.388          |          0.342          | 0.188 |   0.268   |
|       MindSource-7B       |          0.498          |          0.425          |          0.528          |   -   |     -     |
| **Mengzi3-13B-Base** | **0.651 (+6.7%)** | **0.588 (+9.1%)** | **0.776 (+2.9%)** | 0.631 |   0.287   |

> 以上结果基于5-shot，MMLU/CMMLU/OCNLI结果来自[FlagEval](https://flageval.baai.ac.cn/)
>
> The above results are based on 5-shot，MMLU/CMMLU/OCNLI results from [FlagEval](https://flageval.baai.ac.cn/)

## 模型微调/Finetuning

微调代码在finetune_demo文件夹下。

首先需要准备jsonl格式的微调数据。参考 finetune_demo/example.jsonl，每一行为一条json数据，需满足下面格式：

The finetune code in the finetune_demo folder.
Before run the code, first need to prepare the training data in jsonl format. For details, see finetune_demo/example.jsonl. Each line represents one json data in the following format:

```json
{
  "conversation": [
    {
      "role": "human",
      "text": "hello, how are you?"
    },
    {
      "role": "assistant",
      "text": "I am fine."
    },
    ...
  ]
}
```

然后运行全参数微调的脚本。

Then run the supervised finetune script.

```bash
bash finetune.sh
```

# 协议/License Agreement

Mengzi3-13B-Base依照Apache 2.0协议开源，对学术研究完全开放，同时支持免费商用。如需申请商业许可证，请[联系我们](https://www.langboat.com/form?p=3)，其他商务合作请联系[bd@langboat.com](mailto:bd@langboat.com)。

Mengzi3-13B-Base is open source under the Apache 2.0 protocol, fully open for academic research, and free for commercial use. If you need to apply for business license, please [contact us](https://www.langboat.com/en/form?p=3), other business cooperation, please contact [bd@langboat.com](mailto:bd@langboat.com).
