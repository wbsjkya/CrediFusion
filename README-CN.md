# CrediFusion

[English](README.md) | [中文](README-CN.md)

> 面向隐私保护的自监督纵向联邦学习与人机协同信贷智能系统。

CrediFusion 是一个研究与工程原型，目标是在原始数据始终留在各参与机构本地的前提下，学习跨机构可用的数据表示。项目将自监督纵向联邦学习实验，与面向已生成模型信号的隐私保护型本地审查层结合起来。

## CrediFusion 这个名字的含义

`Credi` 是 **credit**（信用、信贷）与 **credibility**（可信度）的组合，既代表项目面向信贷风险评估，也体现了项目对可信、可复核模型辅助的关注。

`Fusion` 表示“融合”：一方面指通过纵向联邦学习，在不移动原始数据的前提下融合多方学习到的表示和风险信号；另一方面也指模型输出与人工审查之间的人机协作。

因此，**CrediFusion** 可以理解为：

> 在数据不离开所属机构的前提下，融合跨机构知识的可信信贷智能系统。

## 项目内容

- **联邦训练代码**：位于 `colla/`、`local/` 和 `finetune/`，用于自监督预训练、纵向协作和下游微调实验。
- **信贷审查副驾**：位于 `ai_credit_copilot/`，包含基于 Python 标准库的 HTTP 服务、浏览器工作台、兼容 Qwen 的客户端、输入白名单、脱敏和面向审计的响应处理。
- **测试与安全示例**：包含审查服务的单元测试。示例案件为合成且去标识化数据。

本公开副本仅包含源代码、测试、配置模板、合成示例数据以及 README 文件。

## 目录结构

```text
CrediFusion/
|-- colla/                     # FATE 协作训练示例
|-- local/                     # 本地自监督预训练与分类代码
|-- finetune/                  # 下游纵向联邦微调示例
|-- ai_credit_copilot/         # 审查服务与浏览器工作台
|   |-- credit_bridge.py       # 安全的模型输出适配器
|   |-- sample_case.json       # 合成的去标识化示例案件
|   `-- tests/                 # 审查边界单元测试
|-- README.md                  # English README
`-- README-CN.md               # 中文 README
```

## 快速开始：启动审查工作台

审查服务仅使用 Python 标准库，可以独立于 PyTorch/FATE 研究环境运行。

在仓库根目录执行：

```powershell
$env:QWEN_API_KEY = "your_dashscope_api_key"
$env:QWEN_MODEL = "qwen3.5"
python -m ai_credit_copilot --port 8787
```

然后在浏览器中打开 `http://127.0.0.1:8787`。同时支持 `DASHSCOPE_API_KEY`。`QWEN_BASE_URL` 默认使用 `https://dashscope.aliyuncs.com/compatible-mode/v1`；只有在使用经过批准的兼容网关时才需要修改它。不要将 API 密钥提交到 Git。

如需将联邦模型产生的聚合风险输出接入审查服务：

```python
from ai_credit_copilot.credit_bridge import build_copilot_request

payload = build_copilot_request(
    case_id="CASE-2026-0042",
    model_score=0.72,
    model_version="credit-demo-2026.08",
    party_count=2,
    alignment_rate=0.96,
    feature_coverage=0.91,
    evidence=[
        {
            "label": "Recent repayment volatility",
            "impact": "up",
            "value": "Aggregated repayment signals require human verification.",
            "confidence": 0.86,
        }
    ],
)
```

将 `payload` 发送到 `POST /api/analyze`。如需在同一受约束的案件上下文中进行审查员追问，可使用 `POST /api/chat`。请求中只能放入聚合且不含身份信息的证据；原始参与方特征、标签、嵌入、源文件和直接标识符必须留在大模型边界之外。

## 运行研究示例

研究脚本需要兼容的 FATE、PyTorch、scikit-learn 和数据集环境。数据集路径与部署设置由使用者自行配置；本公开副本不包含实验数据。

```powershell
cd colla
python stage1.py --parties guest:9999 host:10000 --log_level INFO
```

`local/` 包含本地预训练和下游分类脚本，`finetune/` 包含下游协作微调示例。

## 验证

```powershell
python -m unittest discover -s ai_credit_copilot/tests -v
```

## 安全边界与适用范围

CrediFusion 是研究原型，不是自动化信贷决策系统。审查副驾只能解释模型信号并准备人工核验问题，不得自动批准、拒绝、定价或决定授信额度。任何真实部署都需要经过数据授权、隐私与安全审查、模型验证、公平性评估、人工监督以及适用的合规审查。
