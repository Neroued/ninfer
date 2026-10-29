# Serve TTFT 冻结语料

本目录保存外部 TTFT 场景的固定输入。Runner 读取文件、检查身份、编码媒体与发送请求，
计时期间不加载 tokenizer、生成图像或调用 token-count API。

[manifest.json](manifest.json)维护文件路径、hash、prompt token 数、共享位置和媒体事实。
这些 token 事实来自其声明的 Qwen3.6-family tokenizer/template，不适用于所有已注册模型。

独立长请求在早期 source content 分叉，用于区分私有工作集与意外共享命中；
多轮和两组工作集使用 cases 中声明的改写规则。图片为确定性生成内容，重型输入分别覆盖
合法媒体准备与超出 envelope 的拒绝。具体文件和大小以 manifest 为准，不重复维护清单。

仅在负载合同改变时使用明确的本地 tokenizer 重建：

```bash
python3 tools/bench/ttft/build_fixtures.py --tokenizer /path/to/local/tokenizer
python3 tools/bench/ttft/build_fixtures.py --check
```

`--check` 不加载 tokenizer。测量边界和资源解释见
[TTFT 说明](../../../tools/bench/ttft/README.md)。
