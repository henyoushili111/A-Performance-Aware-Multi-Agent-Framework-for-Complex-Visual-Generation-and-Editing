# 🛡️ AIGC-Nexus: A Fully Automated, Strongly Aligned AIGC Agent Framework for Complex Visual Generation

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Framework](https://img.shields.io/badge/Framework-MetaGPT%20%7C%20vLLM-green)

**** is an out-of-the-box Multi-Agent AIGC workflow orchestration system. It is specifically designed to address the "Scheduling Hallucination"—a common issue where Large Language Models (LLMs) fail when invoking underlying image generation or editing tools, resulting in task planning breakdowns and arbitrary tool selection due to a lack of understanding regarding the actual capability boundaries of the underlying models.

By introducing a quantified, multi-dimensional performance matrix alongside a closed-loop reinforcement fine-tuning mechanism based on real-world visual feedback, AIGC-Nexus achieves a precise and dynamic distribution process—seamlessly translating **vague textual intentions** into the **optimal visual generation tools**.

---

## 🚀 Core Features and Engineering Value

When building long-chain agents for text-to-image generation, relying solely on text prompts to let LLMs select tools (such as FLUX, SD3, or various editing models) often leads to extremely high failure rates. AIGC-Nexus offers an industrial-grade solution:

* 🎯 **Performance Matrix-Based Dynamic Routing**
Say goodbye to the "blind box" style of text matching often seen in large models. The system features a built-in, quantified performance matrix covering 14 dimensions—including color, spatial relationships, and textures. Through multi-dimensional vector dot-product calculations, it dynamically matches the most suitable underlying model to each sub-task, transforming uncertain scheduling into a deterministic mathematical optimization process.
* 🔄 **Feedback-Based Online Adaptive Evolution**
Move beyond rigid, static benchmarking scores. The system incorporates an "Exploration-Exploitation" strategy; during inference, it utilizes a Multimodal Large Language Model (MLLM) as a discriminator to score concurrent generation results, thereby dynamically adjusting the performance weights within the tool library. This ensures your agents become "more accurate with every run" within real-world business scenarios. * 🧠 **A Planner Aligned with Underlying Physical Capabilities**
Built upon a DPO (Direct Preference Optimization) alignment architecture, this system utilizes the actual execution outcomes of underlying tools as a reward signal to fine-tune the top-level Planner model (supporting models such as the Qwen3 series). This ensures that the high-level task decomposition logic—for instance, "process the background first, then modify the subject"—is deeply aligned with the physical execution limits of the underlying diffusion models.

## 📊 Performance Highlights

- **Leap in Scheduling Accuracy:** In complex, long-tail scheduling scenarios, the tool dispatch error rate plummeted from **77.8%** (typical of traditional text-based approaches) to a mere **14.2%**.
- **Cost Reduction & Efficiency Gains:** Thanks to quantized matrix addressing—which bypasses the lengthy contextual reasoning typically required by LLMs—scheduling latency and token consumption are kept extremely low. This architecture effortlessly supports the future integration of massive toolsets comprising hundreds or even thousands of customized LoRA adapters and Checkpoints.
- **Chart-Topping Image Generation Quality:** In complex, multi-intent, and multi-turn editing tasks, the system achieves State-of-the-Art (SOTA) performance across end-to-end semantic alignment, visual reasoning logic, and spatial consistency.

---

## 🛠️ 快速上手 (Quick Start)

## Environment Setup
1. Install Conda Environment
```shell
conda create -n AIGC-Nexus python=3.10
conda activate AIGC-Nexus
```

2. Install [MetaGPT](https://github.com/FoundationAgents/MetaGPT)

3. Please go to the following GitHub repository to install the necessary tool dependencies.
```shell
https://github.com/HaozheZhao/UltraEdit
https://github.com/HuiZhang0812/CreatiLayout
https://github.com/stepfun-ai/Step1X-Edit
https://github.com/weichow23/AnySD
https://github.com/bytedance/DreamO
https://github.com/Xiaojiu-z/EasyControl
https://github.com/tencent-ailab/IP-Adapter
```

4. Modify the relevant code to avoid environment conflicts.
```shell
# Insert the following code into './Step1X_Edit/inference.py'
import sys
sys.path.append("Step1X_Edit")
# Insert the following code into  './AnySD/anysd/src/model.py'
import sys
sys.path.append("AnySD")
```

5. Install the remaining dependencies
```shell
pip install -r requirement.txt
```

6. Download the planner checkpoint zip file and unzip it.
```shell
link: https://pan.baidu.com/s/1DyK9rZeTebdwAKfl_pl6DQ?pwd=ziy2 passward: ziy2 
```

7. Deploy LLM locally using vLLM
```shell
export VLLM_USE_MODELSCOPE=True 
vllm serve Qwen/Qwen3-14B   --port 8001
vllm serve your_planner_checkpoint  --port 8002
```

8. Modify the LLM config file under `./config`

## Run the program
```shell
python run.py
```


/AIGC-Nexus
├── configs/             # System configurations, model routing matrix, and Agent Prompts
├── AIGC-Nexus/           # Core source code package
│   ├── agents/          # Multi-agent role definitions (Analyst, Planner, Worker, Evaluator)
│   ├── routing/         # PASM Performance-Aware Routing Module & APU Dynamic Update Engine
│   ├── tools/           # Low-level vision model integration layer (Diffusers/ControlNet wrappers)
│   └── trainer/         # CAPO Planner DPO fine-tuning scripts
├── examples/            # Common use cases and Jupyter Notebooks
├── scripts/             # One-click startup and environment testing scripts
├── requirements.txt     # Dependency list
└── README.md




