# LLM Guard_Scan - 批量 LLM 安全扫描工具

一个基于 [protectai/llm-guard](https://github.com/protectai/llm-guard) 的**批量离线扫描脚本**，用于检测 Prompt Injection、Toxicity、Secrets 等输入风险，以及 NoRefusal、Bias、Relevance、Sensitive 等输出风险。

主要特点：
- **完全支持离线模式**（本地缓存模型，无需联网）
- 支持 **批量处理 CSV 文件**（单个文件 / 多个文件 / 通配符）
- 自动识别 prompt / response 列，支持手动确认
- 健壮的 CSV 读取（多编码兼容，几乎不会因为编码崩溃）
- 可交互式选择要启用的扫描器
- 支持合并所有结果到一个文件，或每个输入文件单独输出
- 进度条 + 详细风险分数展示


## 适用场景

- 红队测试 / 安全评估 LLM 应用时的大规模 prompt / response 检查
- 审计历史对话数据是否存在 jailbreak、毒性内容、敏感信息泄露、偏见等风险
- 需要完全离线运行的环境（内网、空气隙机器）
- 批量处理从各种渠道收集的对话数据集（CSV 格式）

## 环境要求

- Python 3.8+
- 操作系统：Windows（已充分测试） / Linux / macOS
- 推荐 ≥ 8GB 内存（模型加载 + 批量处理时使用）
- 磁盘空间：约 1.5GB+（模型缓存总大小，包含以下所有模型）

## 配置运行环境（离线模式必备）

脚本已配置为完全离线运行（`HF_HUB_OFFLINE=1`），依赖以下本地模型缓存（默认路径位于 `E:/ai-tools/llmguard/hf_cache/` 或更新为自定义的 `HF_HOME`路径）：

| 模型目录（HF 路径）                                      | 文件/大小示例                  | 主要用途（llm-guard 相关扫描器）                   |     
|----------------------------------------------------------|--------------------------------|----------------------------------------------------
| models--protectai--deberta-v3-base-prompt-injection-v2   | ~400MB (model.safetensors 等)  | PromptInjection（提示注入检测，主模型）           
| models--ProtectAI--distilroberta-base-rejection-v1       | ~300MB 左右                    | NoRefusal（拒绝响应检测，拒绝类输出识别）        
| models--unitary--unbiased-toxic-roberta                  | ~500MB 左右                    | Toxicity（毒性/仇恨言论检测）                    
| models--valurank--distilroberta-bias                     | ~300MB 左右                    | Bias（偏见检测）                                  
| models--BAAI--bge-base-en-v1.5                           | ~400MB 左右                    | 可能用于 Relevance 或内部 embedding（英文嵌入模型）
| models--Isotonic--deberta-v3-base_finetuned_ai4privacy_v2| ~400MB 左右                    | Secrets / Sensitive（PII/隐私信息检测，细粒度 NER）
| en_core_web_sm-3.8.0-py3-none-any.whl                   | 12,806,118 bytes (~12MB)       | spaCy 英文小模型（Toxicity / Secrets 等可能依赖） 
| zh_core_web_sm-3.8.0-py3-none-any.whl                   | 48,515,003 bytes (~48MB)       | spaCy 中文小模型（多语言支持下的实体/毒性检测）  

初次使用项目时，请在https://huggingface.co/下载模型（若访问慢，可访问Hugging Face Hub 的国内镜像站https://hf-mirror.com/下载所需模型
 HuggingFace 模型ID | 标准保存缓存目录名
protectai/deberta-v3-base-prompt-injection-v2
→ models--protectai--deberta-v3-base-prompt-injection-v2

unitary/unbiased-toxic-roberta
→ models--unitary--unbiased-toxic-roberta

ProtectAI/distilroberta-base-rejection-v1
→ models--ProtectAI--distilroberta-base-rejection-v1

valurank/distilroberta-bias
→ models--valurank--distilroberta-bias

BAAI/bge-base-en-v1.5
→ models--BAAI--bge-base-en-v1.5

Isotonic/deberta-v3-base_finetuned_ai4privacy_v2
→ models--Isotonic--deberta-v3-base_finetuned_ai4privacy_v2

请在https://github.com/explosion/spacy-models/releases下载：
en_core_web_sm-3.8.0-py3-none-any.whl
zh_core_web_sm-3.8.0-py3-none-any.whl

**说明**：
- 这些模型覆盖了脚本中所有已启用的扫描器（PromptInjection、Toxicity、Secrets、NoRefusal、Bias、Relevance、Sensitive）。
- `en_core_web_sm` 和 `zh_core_web_sm` 是 spaCy 的核心语言模型，常被 llm-guard 的 Toxicity、Secrets 等扫描器内部使用（用于分词、NER 等预处理）。
- 如果你后续添加其他扫描器（如 Relevance 可能额外依赖 embedding 模型），需手动补充对应缓存。
- 所有模型已本地化，无需联网（脚本强制离线模式也可以手动开启联网模式，修改（`HF_HUB_OFFLINE`））。

# 安装使用方法
python -m venv llmguard-env #创建虚拟环境避免环境冲突，非必要
llmguard-env\Scripts\activate  #进入虚拟环境
python -m ensurepip --upgrade  #安装llm-guard，配置好运行环境，详见配置运行环境（离线模式必备）部分
python -m pip install --upgrade pip
pip install llm-guard

python LLMGuard_Scan.py  #选择相应输入输出扫描器-->选择表格文件中模型的输入和输出所在列-->输出结果路径-->等待输出

示例：

(llmguard-env) E:\ai-tools\llmguard>python LLMGuard_Scan.py

    _      _      __  __    ____                     _
    | |    | |    |  \/  |  / ___|_   _  __ _ _ __ __| |
    | |    | |    | |\/| | | |  _| | | |/ _` | '__/ _` |
    | |___ | |___ | |  | | | |_| | |_| | (_| | | | (_| |
    |_____||_____||_|  |_|  \____|\__,_|\__,_|_|  \__,_|

       L L M   G U A R D   S C A N   B Y  R E W I N D

-------------------------- 可选输入扫描器 --------------------------
PromptInjection, Toxicity, Secrets
-------------------------- 可选输出扫描器 --------------------------
NoRefusal, Bias, Relevance, Sensitive

请选择输入扫描器（逗号分隔，回车全选）：
请选择输出扫描器（逗号分隔，回车全选）：
正在加载 PromptInjection 本地离线模型（首次可能稍慢）...

-------------------------------------------------------------------------   分割线

√ 已启用输入扫描器：['PromptInjection', 'Toxicity', 'Secrets']
√ 已启用输出扫描器：['NoRefusal', 'Bias', 'Relevance', 'Sensitive']


支持输入方式：
   • 单个文件路径
   • 多个路径（逗号分隔）
   • 通配符（如 data/*.csv 或 *.csv）
→ 输入文件路径（支持多个）:


等待输出结果



