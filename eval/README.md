# Capability Evaluation

`ninfer_eval` 是仓库内的能力评测协调器，通过配置的 API target 调用服务。
Coordinator 管理 job、并发、进度与恢复；backend 管理数据集和评分。EvalScope 与无网络的 mock
使用同一 backend 边界，不把 evaluator 实现耦合进推理引擎。

## Environment

使用选定的 Python 3.11 环境，依赖版本以 [requirements.txt](requirements.txt) 为准。
命令通过 `PYTHONPATH=eval` 使用仓库内 package。依赖安装、数据集缓存和模型服务分别管理，
运行评测不隐式转换 `.ninfer` 权重。

## Configuration

[configs](configs/)保存任务配置；[capability-suite](configs/capability-suite.yaml)与
[mock-suite](configs/mock-suite.yaml)提供结构示例。
Target 定义 endpoint、model、凭据环境变量、超时和并发；generation 定义生成参数，
backend_args 定义数据集相关配置。精确字段由 [config.py](ninfer_eval/config.py)维护。

并发分为同时运行的 jobs、target 总请求额度及单 job 可申请额度。多个 job 共用 target 时
不能超过服务实际容量；长输出的 KV 压力也需纳入任务配置。
API key 通过环境变量取得，不进入保存的 effective config。

临时评测将 `runtime.runs_dir` 显式设为 `.agent-tmp/<task>/runs`。
已有发布结果的来源由对应 [model cards](../model-cards/)维护，不从当前环境重跑就声称复现历史分数。

## Commands

```bash
PYTHONPATH=eval python3 -m ninfer_eval plan \
  --config eval/configs/capability-suite.yaml --suite reasoning
PYTHONPATH=eval python3 -m ninfer_eval validate \
  --config eval/configs/capability-suite.yaml --suite reasoning
PYTHONPATH=eval python3 -m ninfer_eval run \
  --config .agent-tmp/eval/suite.yaml --suite reasoning
```

`plan` 展示工作量与前置条件，`--check-runtime` 另检查依赖与配置的凭据；`validate` 检查运行依赖。
服务启动方式由任务或 managed script 指定，评测框架不假设某个模型或并行方式。
[run_qwen3_8_27b_reasoning.sh](run_qwen3_8_27b_reasoning.sh)等脚本保存特定负载的启动配方，
其 `--plan` 可先查看选择，不把旧机器的容量数字复制为通用默认值。

```bash
PYTHONPATH=eval python3 -m ninfer_eval status --run .agent-tmp/eval/runs/RUN
PYTHONPATH=eval python3 -m ninfer_eval resume --run .agent-tmp/eval/runs/RUN
PYTHONPATH=eval python3 -m ninfer_eval summarize --run .agent-tmp/eval/runs/RUN
```

Resume 核对 effective config 与 backend，跳过已完成 job，继续使用可用的预测缓存。
不在恢复中悄悄改变采样、题目或评分条件。

## Results and extension

Run directory 保存 effective config、manifest、state、events、log、backend 原始产物及 summary。
原始响应、截断、失败和不完整 job 保留；未知总数不伪造进度，partial 不报告为完整正式成绩。
不同 benchmark 独立报告，不合成无定义的跨数据集平均分。

普通新数据集通过 job 配置接入；不同 evaluator 实现
[backend 协议](ninfer_eval/backends/base.py)，保留原始产物并返回规范结果。
控制与恢复语义由 [tests](tests/)和 mock backend 检查，真实模型质量仍由实际完整评测建立。
