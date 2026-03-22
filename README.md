# 🛡️AIGC-Nexus 面向复杂视觉生成的全自动、强对齐 AIGC 智能体框架

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Framework](https://img.shields.io/badge/Framework-MetaGPT%20%7C%20vLLM-green)

**** 是一个开箱即用的多智能体（Multi-Agent）AIGC 工作流调度系统。它专为解决当前大语言模型（LLM）在调用底层生图或图像编辑工具时经常发生的**“调度幻觉”（即由于不懂底层模型真实能力边界，导致任务规划崩溃和工具乱选）**而设计。

通过引入量化的多维性能矩阵与基于真实视觉反馈的强化微调闭环，AIGC-Nexus 实现了从**模糊的文本意图**到**最优视觉生成工具**的精准、动态分发。

---

## 🚀 核心特性与工程价值

在构建长链路的图文生成 Agent 时，单纯依靠 Prompt 让 LLM 挑选工具（如 FLUX, SD3, 各种 Edit 模型）往往面临极高的失败率。AIGC-Nexus 提供了工业级的解法：

* 🎯 **基于性能矩阵的动态路由 **
  彻底告别大模型的“盲盒式”文本匹配。系统内置了涵盖色彩、空间关系、材质等 14 个维度的量化性能矩阵，通过多维向量点乘计算，为每一个 Sub-task 动态匹配最合适的底层模型，将不确定的调度转化为确定性的数学寻优。
* 🔄 **基于反馈的在线自适应进化 **
  告别一成不变的静态跑分。系统引入了 Exploration-Exploitation（探索-利用）策略，在推理时利用 MLLM 作为判别器对并发生成结果进行打分，动态修正工具库的性能权重。让你的智能体在真实业务中“越跑越准”。
* 🧠 **对齐物理底层能力的规划器 **
  基于 DPO 偏好对齐架构，利用底层工具的实际执行效果作为 Reward 信号，反向微调最上层的 Planner 模型（支持 Qwen3 系列等）。让宏观的任务拆解逻辑（例如“先处理背景再修改主体”）深度契合底层扩散模型的物理执行极限。

## 📊 性能表现

- **调度准确率飞跃**：在复杂的长尾调度场景中，工具分发错误率由传统文本方案的 **77.8%** 断崖式降低至 **14.2%**。
- **降本增效**：得益于量化的矩阵寻址，绕开了 LLM 冗长的上下文推理，调度延迟和 Token 消耗极低，轻松支持未来挂载成百上千个定制化 LoRA / Checkpoint 的巨型工具集。
- **生图质量霸榜**：在复杂多意图、多轮次编辑任务中，端到端的语义对齐度、视觉推理逻辑与空间一致性均达到 SOTA 级别。

---

## 🛠️ 快速上手 (Quick Start)

### 1. 环境准备

建议使用 Conda 创建纯净的 Python 环境：

```bash

cd AIGC-Nexus

conda create -n AIGC-Nexus python=3.10
conda activate AIGC-Nexus

# 安装核心依赖
pip install -r requirements.txt
