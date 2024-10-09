<div align="left">
<h1>
Mengzi3
</h1>
</div>

<p align="center">
    <img src="./assets/mengzi_logo.png" width="200"/>
<p>

<p align="center">
        🤗 <a href="https://huggingface.co/Langboat">Hugging Face</a> | 🤖 <a href="https://modelscope.cn/organization/Langboat">ModelScope</a> | <a href="https://gitee.com/mindspore/mindformers/blob/r1.0/research/mengzi3/mengzi3.md"><img src="./assets/logo-zh-light.99fc9222.svg" width="50" style="white-space: nowrap;display: inline-block;overflow: hidden;max-width: 100%;"/></a> ｜  <a href="https://wisemodel.cn/organization/Langboat">Wisemodel</a> ｜ 💬 <a href="https://github.com/Langboat/Mengzi3/blob/main/assets/wechat.png">WeChat</a> | <a href="https://www.langboat.com/document/mengzi/mengzi-gpt/call">API</a> | <a href="https://www.langboat.com/portal/mengzi-gpt"><img src="./assets/mengzi_logo.png" width="16" style="white-space: nowrap;display: inline-block;overflow: hidden;max-width: 100%;"/> 孟子GPT</a>
</p>

# 模型介绍/Introduction

本次开源Mengzi3 8B/13B系列模型，模型的地址如下:

The address of the open source Mengzi3 8B/13B series model is as follows:

|    |                                                                                                                                                    Base                                                                                                                                                    |                                                                                                                                                  Chat                                                                                                                                                  |
| :-: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| 8B |   **Mengzi3-8B-Base**([🤗](https://huggingface.co/Langboat/Mengzi3-8B-Base) / [🤖](https://modelscope.cn/models/langboat/Mengzi3-8B-Base) / [MindSpore](https://gitee.com/mindspore/mindformers/blob/r1.0/research/mengzi3/mengzi3.md) / [Wisemodel](https://wisemodel.cn/models/Langboat/Mengzi3-8B-Base))   | **Mengzi3-8B-Chat**([🤗](https://huggingface.co/Langboat/Mengzi3-8B-Chat) / [🤖](https://modelscope.cn/models/langboat/Mengzi3-8B-Chat) / [MindSpore](https://gitee.com/mindspore/mindformers/blob/r1.0/research/mengzi3/mengzi3.md) / [Wisemodel](https://wisemodel.cn/models/Langboat/Mengzi3-8B-Chat)) |
| 13B | **Mengzi3-13B-Base**([🤗](https://huggingface.co/Langboat/Mengzi3-13B-Base) / [🤖](https://modelscope.cn/models/Langboat/Mengzi3-13B-Base) / [MindSpore](https://gitee.com/mindspore/mindformers/blob/r1.0/research/mengzi3/mengzi3.md) / [Wisemodel](https://wisemodel.cn/models/Langboat/Mengzi3-13B-Base)) |                                                                                                                                                敬请期待                                                                                                                                                |
| 13B |                                                                                                                                   **Mengzi3.5-13B-Base (即将更新)**                                                                                                                                   |                                                                                                                                                敬请期待                                                                                                                                                |

Mengzi3 8B/13B模型基于Llama架构，语料精选自网页、百科、社交、媒体、新闻，以及高质量的开源数据集。通过在万亿tokens上进行多语言语料的继续训练，模型的中文能力突出并且兼顾多语言能力。

Mengzi3 8B/13B is based on the Llama architecture, and the corpus is selected from web pages, encyclopedias, social networking, media, news, and high-quality open source data sets. By continuing to train multilingual corpus on trillions of tokens, the model has outstanding Chinese capabilities and takes into account multilingual capabilities.

# 快速开始/Quickstart

首先进行环境配置，安装项目需要的依赖

First configure the environment and install the dependencies required by the project

```shell
pip install -r requirements.txt
```

简单代码调用：

Simple demo:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Langboat/Mengzi3-13B-Base", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Langboat/Mengzi3-13B-Base", device_map="auto", trust_remote_code=True)
inputs = tokenizer('指令：回答以下问题。输入：介绍一下孟子。输出：', return_tensors='pt')
if torch.cuda.is_available():
    inputs = inputs.to('cuda')
pred = model.generate(**inputs, max_new_tokens=512, repetition_penalty=1.01, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(pred[0], skip_special_tokens=True))
```

我们另外提供一个样例代码，可以对基座模型进行单轮的交互推理。

We provide this sample code to perform a single round of interactive reasoning on the base model.

```shell
cd examples
python base_streaming_gen.py --model model_path --tokenizer tokenizer_path
```

# 性能评测/Evaluation

Mengzi3-13B-Base在各项基准测试中与同等参数量大语言模型相比，语言能力成绩领先，数学和编程能力位于前列。

Mengzi3-13B-Base leads in language proficiency and is at the forefront in math and programming proficiency compared to the equivalent large language model in various benchmark tests.

|                              |          MMLU          |          CMMLU          |          OCNLI          | GSM8K |       HumanEval       |
| :--------------------------: | :---------------------: | :---------------------: | :---------------------: | :---: | :--------------------: |
|      Baichuan2-13B-Base      |          0.530          |          0.489          |          0.433          | 0.528 |         0.171         |
|           Qwen-14B           |          0.589          |          0.539          |          0.550          | 0.613 |         0.323         |
|       ChatGLM3-6B-base       |          0.551          |          0.495          |          0.754          | 0.723 |           -           |
|        InternLM2-20B        |          0.610          |          0.538          |          0.650          | 0.761 |         0.488         |
|       Skywork-13B-base       |          0.557          |          0.524          |          0.426          | 0.558 |           -           |
|        LingoWhale-8B        |          0.541          |          0.495          |          0.352          | 0.550 |         0.329         |
|         DeepSeek-7B         |          0.436          |          0.424          |          0.356          | 0.174 |         0.262         |
|    DeepSeek-MoE-16B-base    |          0.423          |          0.388          |          0.342          | 0.188 |         0.268         |
|        MindSource-7B        |          0.498          |          0.425          |          0.528          |   -   |           -           |
|  **Mengzi3-13B-Base**  |      0.651 (+6.7%)      |      0.588 (+9.1%)      | **0.776 (+2.9%)** | 0.631 |         0.287         |
| **Mengzi3.5-13B-Base** | **0.776(+27.2%)** | **0.813(+50.8%)** |            -            |   -   | **0.532(+9.0%)** |

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

```shell
bash finetune.sh
```

# 声明/Disclaimer

我们在此声明，我们的开发团队并未基于 Mengzi 3 模型开发任何应用，无论是在 iOS、Android、网页或任何其他平台。我们按“原样”的形式提供服务，不作任何形式的保证，我们不保证服务将满足您的要求。在不限制这一点的情况下，我们明确声明不提供关于服务的所有明示、默示或法定保证，包括但不限于对适销性、特定用途之适用性、所有权、安全性、准确性和不侵权的任何保证。我们强烈呼吁所有使用者，不要利用 Mengzi 3 模型进行任何危害国家社会安全或违法或侵犯他人合法权益的活动。另外，我们也要求使用者不要将 Mengzi 3 模型用于未经适当安全审查和备案的互联网服务。我们希望所有的使用者都能遵守这个原则，确保科技的发展能在规范和合法的环境下进行。 我们已经尽我们所能，来确保模型训练过程中使用的数据的合规性。然而，尽管我们已经做出了巨大的努力，但由于模型和数据的复杂性，仍有可能存在一些无法预见的问题。因此，如果由于使用 Mengzi 3 开源模型而导致的任何问题，包括但不限于数据安全问题、公共舆论风险，或模型被误导、滥用、传播或不当利用所带来的任何风险和问题，我们将不承担任何责任。对于因您使用从服务获取的文件、信息、内容或其他材料而造成的任何损失，您应承担全部责任和风险。

We hereby declare that our team has not developed any applications based on Mengzi 3 models, not on iOS, Android, the web, or any other platform. We provide our service “as is” without warranty of any kind. We do not warrant that the service will meet your requirements. Without limiting this, we expressly disclaim all warranties, whether express, implied or statutory, regarding the service including without limitation any warranty of merchantability, fitness for a particular purpose, title, security, accuracy and non-infringement. We strongly call on all users not to use Mengzi 3 models for any activities that harm national / social security or violate the law or violate the legitimate rights and interests of others. Also, we ask users not to use Mengzi 3 models for Internet services that have not undergone appropriate security reviews and filings. We hope that all users can abide by this principle and ensure that the development of technology proceeds in a regulated and legal environment. We have done our best to ensure the compliance of the data used in the model training process. However, despite our considerable efforts, there may still be some unforeseeable issues due to the complexity of the model and data. Therefore, if any problems arise due to the use of Mengzi 3 open-source models, including but not limited to data security issues, public opinion risks, or any risks and problems brought about by the model being misled, abused, spread or improperly exploited, we will not assume any responsibility. You shall assume full responsibility and risk of loss resulting from your use of files, information, content or other material obtained from the service.

# 协议/License Agreement

Mengzi3-13B-Base依照Apache 2.0协议开源，对学术研究完全开放，同时支持免费商用。如需申请商业许可证，请[联系我们](https://www.langboat.com/form?p=3)，其他商务合作请联系[bd@langboat.com](mailto:bd@langboat.com)。

Mengzi3-13B-Base is open source under the Apache 2.0 protocol, fully open for academic research, and free for commercial use. If you need to apply for business license, please [contact us](https://www.langboat.com/en/form?p=3), other business cooperation, please contact [bd@langboat.com](mailto:bd@langboat.com).
