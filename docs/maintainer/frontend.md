# Frontend 架构

Frontend 将产品输入转换为模型输入，并解释模型已提交的输出。公共机制服务 Qwen 与 GLM，
模板、媒体算法和输出语言仍属于各自模型。它与 GPU 模型执行通过拥有型输入及事务边界连接。

## 输入与模型策略

产品层负责协议字段、文件/URL 获取和连接生命周期，传入统一 PromptInput。
Frontend 使用 artifact 的 tokenizer、模板、generation 与 processor 资源，产生 PreparedPrompt。
PreparedPrompt 持有 token、位置、媒体载荷和输入身份，跨异步执行保持有效。

Tokenizer 的 normalization、pre-tokenization 和 BPE 行为来自资源描述。模板渲染、特殊 token、
采样预设、思考选项和工具参数编码由模型策略解释。公共层提供 Jinja、文本资源、工具声明和
媒体缓存机制，不把 Qwen 的标记或位置规则当作所有模型的默认语义。

模型 embedding/head 行数、tokenizer 的可寻址 ID 域、可选 proposal 的行域分别定义。
Padding 权重行不会自动成为可生成 token；indexed proposal 通过自己的映射回到真实 token ID。

实现分工见 [text](../../src/text/)、[共享 Frontend](../../src/models/common/frontend/)、
[Qwen Frontend](../../src/models/common/qwen/frontend/)和
[GLM Frontend](../../src/models/glm5next/frontend/)。

## 输入身份与缓存边界

缓存复用需要真实的序列化前缀一致性。Frontend 保留来源定位，处理模板闭合、历史 thinking
清除、工具历史规范化和 BPE 尾部变化，提供可比较的输入身份和候选恢复位置。
共享 marker 与私有改写恢复点有不同用途；实际捕获、保留和恢复由
[Context 管理](context-management.md)完成。

媒体身份包含真实内容和影响结果的预处理计划。相同占位 token 不证明图像相同。
压缩索引的内部比较位置可以落在不能捕获 State 的媒体内部，因此身份查询与合法 checkpoint
边界不能用同一个判断代替。

请求的输出约束不改变已准备 prompt 的数学身份。相同输入可以复用模型前缀，同时为本请求
建立不同的 matcher、采样参数和工具策略。

## 输出事务与工具

每请求独占 OutputSession，持有 detokenization、thinking/content 状态、工具 decoder、停止策略
及可选 grammar matcher。公共输出事件供 Chat、Responses、Anthropic 和 CLI 使用；
产品层不重新解析模型原生工具标签。

一次执行先 preview 候选 token，模型接受实际前缀后再 commit 输出。Discard 不留下语法或
解析状态的副作用。跨 token 的 UTF-8、标记和参数片段由同一 decoder 持续处理，流式和聚合
共享语义。长度停止时的未完成工具调用与正常完整调用分别表达。

Qwen 与 GLM 的工具值编码、thinking framing、停止与强制关闭规则各自维护。
模板生成、grammar 和 decoder 必须使用同一模型合同；不能让一套规则允许的参数在另一套规则中
被解释为不同类型。具体实现与测试维护标记字节和协议字段，本文不复制这些表。

## 约束解码

XGrammar CPU 核心负责语言编译与 matcher。Frontend 组合正文约束、工具语言和模型输出封装；
compiled grammar 与词表资源可共享，matcher 和可变输出状态归请求。
编译和缓存属于准备边界，不在 Engine 执行锁内完成。暂停保留 matcher，Replay 不再次消费旧输出。

执行期间，Host 根据真实候选前缀构造合法 token mask。Mask 经原 ticket 的 decision 交接到
RankProgram，采样和 acceptance 只消费数值输入，不解释 JSON、工具名称或模板。
依赖 draft 的 lookahead 可以与 target forward 重叠；只有最终决策等待所需 mask。

Speculative 约束改变 target 分布，不重定义 proposal 的概率。拒绝校正使用实际 proposal q，
不能先按 target mask 裁剪 q 再归一化。具体概率过程由 sampling Ops 和
[Speculative 执行](speculative-decoding.md)说明。

语法准备错误、生成到合法集合为空、内部 matcher/执行合同错误分别处理。已有输出只随成功提交
前进；终止、内容完整与输出上限是不同事实。自由生成的工具格式错误与受约束合同违规也不合并。

## 有界媒体准备

媒体解码、CPU processor 与 GPU Vision encoder 是分开的阶段。
模型 processor 选择采样、resize、patch 顺序和展开 token；公共 decoder 按明确选择读取，
不替模型决定帧数或布局。

共享媒体资源使用有界任务执行、singleflight 和 live payload 预算。被缓存保留及被请求持有的
同一载荷按真实寿命计费；缓存额度不是 live 内存的额外无限空间。取消与等待沿准备控制传播，
生产中的结果和异步上传来源保持到消费者结束。

GPU encoder 使用模型提供的容量查询，在独立 Vision 预算内选择执行窗口。
Scratch 可以借用 Program 的 compute workspace，features 保留到相应 Text spans 的最后消费者。
一次输入含很多媒体，不意味着同时物化所有 encoder 中间结果；一个 attention segment 的完整
数学也不能通过任意截断来满足预算。

Qwen 与 GLM 共享上述资源机制，各自拥有 processor、encoder 与 segment 的数学。
媒体请求进入 decode 后使用普通模型上下文，不继续占用无消费者的 Vision 临时激活。

入口：[媒体缓存](../../src/models/common/frontend/media_cache.h)、
[Vision 资源](../../src/models/common/vision/resources.h)、
[Vision 窗口](../../src/models/common/vision/window.h)。
