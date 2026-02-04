import os
import csv
from datetime import datetime
from pathlib import Path
from typing import List
import pandas as pd
from tqdm import tqdm
import glob  # 支持通配符展开

# ===============================
# 1. 本地缓存设置(离线模式)
# ===============================
HF_CACHE = Path("E:/ai-tools/llmguard/hf_cache/")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 指向国内镜像（hf-mirror.com）
#os.environ["HF_HUB_URL"] = "https://hf-mirror.com"
#os.environ["HF_HUB_OFFLINE"] = "0"  # 开启联网模式才能用镜像

# ===============================
# 2. 导入 llm-guard 相关类
# ===============================
from llm_guard.input_scanners import PromptInjection, Toxicity, Secrets
from llm_guard.output_scanners import NoRefusal, Bias, Relevance, Sensitive
from llm_guard.transformers_helpers import Model

# ===============================
# 3. 离线加载 PromptInjection 模型函数
# ===============================
def get_offline_prompt_injection(threshold: float = 0.5) -> PromptInjection:
    base_model_dir = HF_CACHE / "models--protectai--deberta-v3-base-prompt-injection-v2" / "snapshots"
    snapshots = [d for d in base_model_dir.iterdir() if d.is_dir()]
    if not snapshots:
        raise FileNotFoundError(f"未找到 snapshot 子目录，请确认 {base_model_dir} 下有模型快照文件夹")

    model_dir = snapshots[0]

    required_files = ["config.json", "tokenizer.json", "model.safetensors"]
    for f in required_files:
        if not (model_dir / f).exists():
            raise FileNotFoundError(f"离线模型缺失文件: {model_dir / f}")

    print("   └─ 模型文件检查通过，正在初始化 PromptInjection 扫描器...")

    offline_model = Model(
        path=str(model_dir),
        subfolder="",
        revision=None,
        onnx_path=None,
        onnx_subfolder="onnx",
        onnx_filename="model.onnx",
        kwargs={},
        pipeline_kwargs={
            "device": "cpu",
            "return_token_type_ids": False,
            "max_length": 512,
            "truncation": True
        },
        tokenizer_kwargs={}
    )

    return PromptInjection(threshold=threshold, model=offline_model)

# ===============================
# 4. 交互式选择扫描器（根据需求选择加载）
# ===============================
def choose_scanners():
    banner = r"""
    _      _      __  __    ____                     _ 
    | |    | |    |  \/  |  / ___|_   _  __ _ _ __ __| |
    | |    | |    | |\/| | | |  _| | | |/ _` | '__/ _` |
    | |___ | |___ | |  | | | |_| | |_| | (_| | | | (_| |
    |_____||_____||_|  |_|  \____|\__,_|\__,_|_|  \__,_|
                                                        
        L L M   G U A R D   S C A N   B Y  R E W I N D
    """
    print(banner)
    print("-------------------------- 可选输入扫描器 --------------------------")
    print("PromptInjection, Toxicity, Secrets")
    print("-------------------------- 可选输出扫描器 --------------------------")
    print("NoRefusal, Bias, Relevance, Sensitive\n")

    inp = input("请选择输入扫描器（逗号分隔，回车全选）：").strip()
    outp = input("请选择输出扫描器（逗号分隔，回车全选）：").strip()

    selected_input_names = ["PromptInjection", "Toxicity", "Secrets"] if not inp else [n.strip() for n in inp.split(",") if n.strip()]
    selected_output_names = ["NoRefusal", "Bias", "Relevance", "Sensitive"] if not outp else [n.strip() for n in outp.split(",") if n.strip()]

    available_input = ["PromptInjection", "Toxicity", "Secrets"]
    available_output = ["NoRefusal", "Bias", "Relevance", "Sensitive"]

    invalid_input = [n for n in selected_input_names if n not in available_input]
    invalid_output = [n for n in selected_output_names if n not in available_output]
    if invalid_input:
        print(f"⚠️ 输入扫描器名称错误，已忽略: {invalid_input}")
    if invalid_output:
        print(f"⚠️ 输出扫描器名称错误，已忽略: {invalid_output}")

    selected_input_names = [n for n in selected_input_names if n in available_input]
    selected_output_names = [n for n in selected_output_names if n in available_output]

    input_scanners = []
    if "PromptInjection" in selected_input_names:
        print("正在加载 PromptInjection 本地离线模型（首次可能稍慢）...")
        input_scanners.append(get_offline_prompt_injection(threshold=0.5))
    if "Toxicity" in selected_input_names:
        print("   └─ 初始化 Toxicity 扫描器")
        input_scanners.append(Toxicity(threshold=0.5))
    if "Secrets" in selected_input_names:
        print("   └─ 初始化 Secrets 扫描器")
        input_scanners.append(Secrets())

    output_scanners = []
    if "NoRefusal" in selected_output_names:
        print("   └─ 初始化 NoRefusal 扫描器")
        output_scanners.append(NoRefusal())
    if "Bias" in selected_output_names:
        print("   └─ 初始化 Bias 扫描器")
        output_scanners.append(Bias())
    if "Relevance" in selected_output_names:
        print("   └─ 初始化 Relevance 扫描器")
        output_scanners.append(Relevance())
    if "Sensitive" in selected_output_names:
        print("   └─ 初始化 Sensitive 扫描器")
        output_scanners.append(Sensitive())

    print(f"\n√ 已启用输入扫描器：{[type(s).__name__ for s in input_scanners] or '无'}")
    print(f"√ 已启用输出扫描器：{[type(s).__name__ for s in output_scanners] or '无'}\n")

    return input_scanners, output_scanners

# ===============================
# 5. 扫描函数
# ===============================
def scan_input(prompt: str, scanners: List) -> dict:
    results = {}
    for scanner in scanners:
        _, valid, risk = scanner.scan(prompt)
        results[scanner.__class__.__name__] = {
            "valid": valid,
            "risk_score": round(float(risk), 3)
        }
    return results

def scan_output(prompt: str, response: str, scanners: List) -> dict:
    results = {}
    for scanner in scanners:
        _, valid, risk = scanner.scan(prompt, response)
        results[scanner.__class__.__name__] = {
            "valid": valid,
            "risk_score": round(float(risk), 3)
        }
    return results

# ===============================
# 6. 格式化结果为字符串
# ===============================
def format_scan_result(result: dict) -> str:
    if not result:
        return ""
    lines = [f"{k}: {'PASS' if v['valid'] else 'BLOCK'} (risk: {v['risk_score']})"
             for k, v in result.items()]
    return "; ".join(lines)

# ===============================
# 7. 健壮读取 CSV（自动尝试多种编码，永不崩溃）
# ===============================
def robust_read_csv(file_path: str) -> pd.DataFrame:
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb18030', 'gb2312']
    for enc in encodings:
        try:
            df = pd.read_csv(
                file_path,
                encoding=enc,
                on_bad_lines='skip',
                dtype=str,
                engine='python'
            ).fillna('')
            print(f"√ 成功读取，使用编码: {enc}")
            return df
        except Exception:
            continue

    # 最终保底：latin1（能读任何字节，不会报错）
    df = pd.read_csv(
        file_path,
        encoding='latin1',
        on_bad_lines='skip',
        dtype=str,
        engine='python'
    ).fillna('')
    print(f"warning: 使用保底编码 latin1 读取（可能有少量乱码，但不会崩溃）")
    return df

# ===============================
# 8. 智能识别并确认 prompt/response 列
# ===============================
def select_prompt_response_columns(df: pd.DataFrame, file_name: str):
    print(f"\n 【{os.path.basename(file_name)}】 共 {len(df)} 行，{df.shape[1]} 列")
    print("列名列表：")
    for i, col in enumerate(df.columns):
        print(f"  [{i}] {col}")

    prompt_keywords = ["prompt", "input", "question", "user", "query", "message", "text"]
    response_keywords = ["response", "output", "answer", "reply", "model", "completion", "reply"]

    prompt_candidates = [i for i, col in enumerate(df.columns) if any(kw in str(col).lower() for kw in prompt_keywords)]
    response_candidates = [i for i, col in enumerate(df.columns) if any(kw in str(col).lower() for kw in response_keywords)]

    default_prompt = prompt_candidates[0] if prompt_candidates else None
    default_response = response_candidates[0] if response_candidates else None

    print("\n 自动识别结果：")
    print(f"   Prompt 列 → {f'[{default_prompt}] {df.columns[default_prompt]}' if default_prompt is not None else '未识别'}")
    print(f"   Response 列 → {f'[{default_response}] {df.columns[default_response]}' if default_response is not None else '未识别'}")

    print("\n请确认或手动指定（支持列号或列名，回车使用自动识别）")
    prompt_input = input(f"→ Prompt 列（默认 {default_prompt if default_prompt is not None else '无'}）: ").strip()
    response_input = input(f"→ Response 列（默认 {default_response if default_response is not None else '无'}）: ").strip()

    def resolve(user_input, default_idx):
        if not user_input and default_idx is not None:
            return default_idx
        if not user_input:
            raise ValueError("必须指定有效的列")
        try:
            idx = int(user_input)
            if 0 <= idx < len(df.columns):
                return idx
        except ValueError:
            if user_input in df.columns:
                return df.columns.get_loc(user_input)
        raise ValueError(f"无效的列名或列号: {user_input}")

    try:
        prompt_col = resolve(prompt_input, default_prompt)
        response_col = resolve(response_input, default_response)
    except ValueError as e:
        print(f"X {e}")
        print("程序退出，请重新运行并正确输入")
        exit(1)

    print(f"√ 已确认：Prompt=[{prompt_col}] {df.columns[prompt_col]} | Response=[{response_col}] {df.columns[response_col]}\n")
    return prompt_col, response_col

# ===============================
# 9. 主流程（批量处理 + 合并/分开输出）
# ===============================
def main():
    input_scanners, output_scanners = choose_scanners()

    print("\n支持输入方式：")
    print("   • 单个文件路径")
    print("   • 多个路径（逗号分隔）")
    print("   • 通配符（如 data/*.csv 或 *.csv）")
    input_str = input("→ 输入文件路径（支持多个）: ").strip().strip('"')

    input_paths = [p.strip().strip('"') for p in input_str.split(",") if p.strip()]
    csv_files = []
    for p in input_paths:
        expanded = glob.glob(p)
        if expanded:
            csv_files.extend(expanded)
        else:
            print(f" 未找到匹配文件: {p}")

    if not csv_files:
        print("X 未找到任何 CSV 文件")
        return

    csv_files = sorted(set(csv_files))
    print(f"\n√ 共找到 {len(csv_files)} 个文件：")
    for f in csv_files:
        print(f"   • {f}")

    print("\n 输出方式：")
    merge_choice = input("→ 是否合并所有结果到一个文件？（y/回车=是，n=各自输出）: ").strip().lower()
    merge_output = merge_choice != 'n'

    if merge_output:
        output_csv = input("→ 合并输出文件名（回车自动生成）: ").strip().strip('"')
        if not output_csv:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_csv = f"llmguard_merged_result_{timestamp}.csv"
        print(f"\n所有结果将合并写入：{os.path.abspath(output_csv)}")
    else:
        print("\n每个文件将生成独立结果文件")

    merged_file = None
    merged_writer = None
    if merge_output:
        merged_file = open(output_csv, "w", encoding="utf-8", newline="")
        merged_writer = csv.writer(merged_file)

    total_processed = 0

    for csv_file in csv_files:
        print("\n" + "="*70)
        print(f"正在处理：{os.path.basename(csv_file)}")
        print("="*70)

        try:
            # 使用健壮读取函数
            df = robust_read_csv(csv_file)

            if len(df) == 0:
                print(" 文件为空，跳过")
                continue

            prompt_col, response_col = select_prompt_response_columns(df, csv_file)

            if merge_output:
                current_writer = merged_writer
                current_file = merged_file
                add_source_col = True
            else:
                base_name = Path(csv_file).stem
                out_path = f"{base_name}_llmguard_result_{datetime.now().strftime('%H%M%S')}.csv"
                current_file = open(out_path, "w", encoding="utf-8", newline="")
                current_writer = csv.writer(current_file)
                add_source_col = False
                print(f"结果将保存到：{out_path}")

            header = list(df.columns) + ["Input_Result", "Input_Scan_Detail", "Output_Result", "Output_Scan_Detail"]
            if total_processed == 0 and merge_output:
                merged_writer.writerow(["Source_File"] + header)
            elif not merge_output:
                current_writer.writerow(header)

            with tqdm(total=len(df), desc="检测进度", unit="条", ncols=100) as pbar:
                for _, row in df.iterrows():
                    prompt = str(row.iloc[prompt_col]).strip()
                    response = str(row.iloc[response_col]).strip()

                    input_result = scan_input(prompt, input_scanners)
                    output_result = scan_output(prompt, response, output_scanners)

                    input_pass = all(r["valid"] for r in input_result.values()) if input_result else True
                    output_pass = all(r["valid"] for r in output_result.values()) if output_result else True

                    row_data = row.tolist()
                    if add_source_col:
                        row_data = [os.path.basename(csv_file)] + row_data

                    current_writer.writerow([
                        *row_data,
                        "PASS" if input_pass else "BLOCK",
                        format_scan_result(input_result),
                        "PASS" if output_pass else "BLOCK",
                        format_scan_result(output_result),
                    ])

                    pbar.update(1)
                    total_processed += 1

            if not merge_output:
                current_file.close()
                print(f"√ {os.path.basename(csv_file)} 处理完成")

        except Exception as e:
            print(f"X 处理 {csv_file} 时发生未知错误：{e}")

    if merge_output and merged_file:
        merged_file.close()

    print(f"\n 批量检测全部完成！共处理 {total_processed} 条对话")
    if merge_output:
        print(f" 合并结果文件：{os.path.abspath(output_csv)}")
    else:
        print(" 各文件结果已分别保存")

if __name__ == "__main__":
    main()