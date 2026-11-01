# 构建系统

构建按执行目标选择源码，数学接口和模型语义保持独立。当前目标为 RTX 5090 与 DGX Spark。
用户安装与运行入口见[README](../../README.md)，本文件说明编译归属和维护边界。

## 编译目标

`NINFER_TARGET` 为每个 build directory 选择一个目标：

| 目标 | 系统 | CUDA architecture |
|---|---|---|
| `rtx5090` | Linux x86-64 | 120a |
| `dgx_spark` | Linux AArch64 | 121a |

平台配置在 CUDA toolchain 检测前解析，验证系统、架构和 build directory 一致性。
修改编译目标或 compiler 使用独立构建目录。维护入口为
[NinferPlatform.cmake](../../cmake/NinferPlatform.cmake)与
[CMakePresets.json](../../CMakePresets.json)。

生成的内部 platform header 区分 target identity 与 capabilities。
`NINFER_TARGET_RTX5090` / `NINFER_TARGET_SPARK` 对称表达目标；共享物理内存和 CPU expert 能力
通过独立 capability 转发。小型差异可使用这些宏，完整实现目录由 CMake 选择。
模型不据此复制 Op 支持判断，硬件 roofline 参照归 bench。

## 源码与依赖归属

小组件直接列源文件，大组件使用明确的 `sources.cmake`。每个实现一个编译 owner，
新增文件显式注册；不递归扫描目录或自动发现 shape。Manifest 按 `CMAKE_CURRENT_LIST_DIR`
定位文件，新增 manifest 不自动要求新增 library。

可共享源码注册一次，target-private route 由各自 manifest 选择。
AVX CPU executor/MoE 只进入 x86 目标；Spark 不编译这些对象。
Op 自己保留 format/shape 分派和调优，CMake 不成为另一套 runtime registry。

配置选项与实际 target 以 [根 CMake](../../CMakeLists.txt)、
[src](../../src/CMakeLists.txt)和[依赖声明](../../cmake/Dependencies.cmake)为准。
模型加载、控制与设备执行保持不同依赖边界；runtime 通过 contracts/support 组织它们，
不让通用 Core/Artifact 依赖模型或 Ops。

`NINFER_ENABLE_DISTRIBUTED` 独立控制执行组和 NCCL 依赖，关闭时不发现或链接 NCCL。
系统库正常安装，vendor 源码固定在仓库；配置阶段不下载依赖。
编译器、Python 和本地依赖路径放入忽略的 `CMakeUserPresets.json`。

## 共享库与 CUDA

产品、测试和 benchmark 共用内部 Core/Ops/CPU shared libraries，输出位于 `<build-dir>/lib/`。
相对 RPATH 保持 build tree 可运行，移动产物需保留相应目录关系。模型、artifact 和专用测试支持
按实际依赖保持 static；不提供安装或导出的 SDK/版本化 ABI。

CUDA shared library 各自拥有 device link 与 registration，不跨库暴露 device symbols，
调用边界使用 Host 函数。非 RDC archive 用于有实际指令或寄存器要求的独立 CUDA 单元，
通过 Host launcher 接入。Consumers 不重复嵌入整个 Ops fatbin。

源码整理必须保留 CUDA architecture、RDC、numerical compiler options 和唯一 device-link owner。
拆 manifest 不等于减少实例化，不能因此合并全部 shape translation units 或启用 unity build。
具体 compile policy 在 [NinferTargets.cmake](../../cmake/NinferTargets.cmake)与所属 family。

## 开发构建

`*-release` 构建产品，`*-dev` 加入 tests/benchmarks，`*-ops` 只构建相应基础与 Op。
例如：

```bash
cmake --preset rtx5090-dev
cmake --build build/rtx5090-dev -j
```

Spark 使用 `dgx-spark-*`；分布式另启用对应选项。独立 build tree 防止目标、缓存和产物混用。
具体选项默认值直接查 CMake，测试与测量命令分别见
[tests](../../tests/README.md)和[bench](../../bench/README.md)。

测试可以共享编译支持，但独立数值 oracle 的编译选项必须作用到实际包含 oracle 的 library。
失败注入仅在测试 executable 内通过现有 interpose 机制实现，不增加生产 fault hooks。
构建组织变化检查源码归属、compile/device-link flags 与受影响行为。
