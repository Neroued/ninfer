# ChatGPT Web Workflow Entrypoint

> Status: workflow note. Date: 2026-06-27.
> Purpose: make the web ChatGPT project workflow reproducible without re-explaining the project every time.

This repository is developed with a split workflow:

- **Web ChatGPT**: architecture control, planning, research, design review, task slicing, documentation design, and Codex prompt generation.
- **Codex**: concrete implementation, test execution, formatting, commits, and iterative code changes.
- **Repo MCP**: the source of truth for current repository state. Every planning session should inspect the repo before making phase or priority claims.

The project target is fixed: a from-scratch C++/CUDA inference engine specialized for **Qwen3.6-27B** on a single **RTX 5090**, with a custom q5090 weight format and hand-written operators. It is not a general-purpose model runtime.

---

## 1. Session boot prompt

Use this at the start of a web ChatGPT session when you want the assistant to re-enter the repository workflow quickly:

```text
进入 qwen3.6-ultraspeed 工作流。

请通过 repo MCP 先读取 project brief、recent modified files、docs/plans 当前阶段、README/design 状态，并判断：
1. 当前实际 M 阶段；
2. 文档与代码是否漂移；
3. 当前最高优先级阻塞项；
4. 下一条适合交给 Codex 的自包含任务。

输出格式：
- 当前状态摘要
- 发现的风险 / 漂移
- 下一步建议
- Codex task prompt
- 审查 checklist
```

---

## 2. Suggested ChatGPT Project Instructions

The ChatGPT Project's shared instructions can include the following text. These instructions guide the assistant, but they do not by themselves execute tools before a user sends a message.

```text
You are working inside the qwen3.6-ultraspeed project.

This project is a from-scratch C++/CUDA inference engine specialized only for Qwen3.6-27B on a single RTX 5090. It uses a custom q5090 quantized weight format and hand-written CUDA operators. It is not a general-purpose runtime.

Default role split:
- Web ChatGPT is the planning, research, architecture, documentation, review, and Codex-task design layer.
- Codex is the implementation layer.
- The repo MCP is the source of truth for current repository state.

At the start of any substantive repository-planning session, first use the repo MCP to inspect the current project state before claiming the current phase or next task. Prefer reading: project brief, recent modified files, docs/plans, README/design status, and any task-specific files.

Assume the repository may have moved beyond older docs. In particular, do not trust stale
design-only phase statements without checking current code and plans.

When planning work:
1. Identify the actual current M stage.
2. Detect documentation/code drift.
3. Separate correctness gates, hardening gates, and performance gates.
4. Produce Codex-self-contained tasks with reading list, files, hard rules, DoD, test commands, and commit message.
5. Do not ask the user to repeat the project background unless the repo MCP is unavailable.

Default response format for workflow entry:
- Current state summary
- Risks / drift
- Recommended next step
- Codex task prompt
- Review checklist
```

---

## 3. What Project Instructions can and cannot do

Project instructions are useful for making the assistant consistently choose the qwen3.6-ultraspeed workflow once a conversation starts. They can establish the role split, preferred reading order, output format, and rule that repo state must be checked before planning.

They should not be treated as an automatic job runner. A new chat still needs a user message such as the session boot prompt above. After that message, the assistant should follow the project instructions and call the repo MCP as needed.

---

## 4. Recommended workflow loop

1. **Inspect**: read bounded repo state via MCP; do not infer from memory alone.
2. **Classify**: decide whether the task is design, hardening, correctness, performance, documentation, or review.
3. **Plan**: produce a small number of Codex-ready tasks, not broad vague milestones.
4. **Review**: compare Codex output against the design docs, q5090 ABI, numerical rules, graph-readiness invariants, and test standard.
5. **Sync docs**: update docs when implementation state or design decisions change.

---

## 5. Current caveat

Older top-level docs may lag implementation. Before using milestone labels, verify them against
current code and plans through the repo MCP. As of this documentation sync, the repo state is:
M2 correctness baseline implemented; M2.5 hardening/documentation sync mostly landed; M2.8
benchmark/I/O/memory observability is the active pre-M3 gate; M3 performance optimization follows
after M2.8 readiness.
