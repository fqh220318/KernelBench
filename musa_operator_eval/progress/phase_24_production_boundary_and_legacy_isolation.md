# Phase 24：生产评测边界与 legacy 研究资产隔离

日期：2026-10-07

## 完成结论

Phase 24 的仓库内软件边界已经完成：Agent release 包只包含一个通过门禁的公开任务；公开
evaluator 明确降级为开发镜像/mock，生产评测必须由私有挂载提供；legacy A/B 配对和 compiled
路径只能用于研究诊断，不能进入正式评分、排名或发布。

“完成”指代码、打包规则和验收测试已经建立。生产环境是否实际使用私有 evaluator 挂载仍是
部署责任，需要在上线流程中执行，不能由公开仓库本身证明。

## 1. 生产 evaluator 边界

- release packager 只接受一个 `release_ready` 任务；
- 打包验收拒绝 `evaluator/`、`private/` 或任何维护者资产进入 Agent 包；
- 包元数据明确要求生产 evaluator 由私有、访问受控的位置挂载；
- 公开 `evaluator/` 仅用于开发、规则说明和可重复 mock，不被宣称为生产秘密实现。

## 2. A/B 排他分类

- 一个正式语义合同只能属于 A 或 B；
- 当前 11 个合同已收口为 A：2 keep/4 research-only，B：1 keep/4 research-only；
- 历史配对目录可以继续用于研究对照，但测试强制这些配对全部为 `research_only`；
- 集合清单和符合性说明不再把同一 entry 的 A/B 两份合同描述为两条正式赛道。

## 3. legacy 资产隔离

- C++/直接后端库执行器标记为 legacy research-only diagnostic；
- 8 个 `research_only` 合同不能被 release packager 接受，也不会进入 baseline 正式采集队列；
- inventory、admission 和 ranker 对不可发布合同 fail closed；
- 任何需要扩展语义重新参评的方案必须创建新 task id，不能继承旧合同的 admission 或历史分数。

## 4. 下一阶段入口

Phase 25 只处理三个 `keep / hardware_pending` 合同：先在有 torch 的软件环境完成 A 类 oracle
交叉重放和 golden 重生，再在目标 MUSA 硬件按 Phase 23 冻结协议采集全部 Case 的正确性、
分派轨迹（仅 B）与计时证据。未满足全部发布条件前继续保持 `hardware_pending`。

## 验证

- release packager 的部署边界、research-only 拒绝与单任务包测试通过；
- 11 个合同的 A/B lifecycle、8 个 legacy pair 隔离和正式 timing/profile 合同测试通过；
- normative disclosure 与原始指南冻结测试通过；
- 全量软件回归结果见 Phase 23：594 项执行，54 项按环境跳过，无失败。

