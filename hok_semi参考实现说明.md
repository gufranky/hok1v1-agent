# hok_semi 参考实现说明

## 1. 这份参考实现是什么

`cankoa/hok_semi` 是一套更完整的《王者荣耀 1v1》强化学习实现，比当前仓库根目录下的简化版 PPO 明显更复杂。

它的主实现目录是：

- `cankoa/hok_semi/code/agent_ppo`

从配置看，默认训练主线就是这套 PPO：

- `cankoa/hok_semi/code/conf/algo_conf_hok1v1.toml`
- `cankoa/hok_semi/code/train_test.py`

它和你当前项目最大的区别有两个：

1. 观测不再是 10 维小特征，而是一个手工构造的 **3910 维大观测**
2. 奖励不再只有塔血和前进，而是包含血量、塔血、金币、经验、法力、击杀、死亡、补刀、前进等多项

---

## 2. 训练主流程

训练流程在：

- `cankoa/hok_semi/code/agent_ppo/workflow/train_workflow.py`

主线和你当前仓库类似：

1. 创建 `EnvConfManager`
2. 启动 `EpisodeRunner`
3. 每局 reset 环境和 agent
4. 每帧：
   - observation 进 `Agent.observation_process()`
   - 模型推理得到动作
   - `env.step(actions)`
   - `reward_manager.result(frame_state)` 计算奖励
   - `build_frame()` 存样本
5. 对局结束后：
   - `FrameCollector` 做 GAE
   - 打包成 learner 样本
   - 发给 learner 训练

当前对局配置在：

- `cankoa/hok_semi/code/agent_ppo/conf/train_env_conf.toml`

当前默认：

- `opponent_agent = "selfplay"`
- `eval_opponent_type = "common_ai"`

英雄池在代码里配置为：

- `169`：后羿
- `173`：李元芳
- `174`：虞姬

---

## 3. 输入是什么

## 3.1 环境原始输入

环境每帧给 agent 的原始输入包含：

- `frame_state`
- `legal_action`
- `sub_action_mask`
- `player_id`
- `player_camp`

其中：

- `frame_state` 是原始对局状态
- `legal_action` 是合法动作掩码
- `sub_action_mask` 用于训练样本中的子动作约束

---

## 3.2 模型真正吃到的输入

`Agent.observation_process()` 里先把原始 observation 解包成 `Info`，再交给：

- `cankoa/hok_semi/code/agent_ppo/feature/obs_builder.py`

最终由 `ObsBuilder.build_observation()` 构造出模型输入特征。

配置在：

- `cankoa/hok_semi/code/agent_ppo/conf/conf.py`

关键定义：

- `Config.SERI_VEC_SPLIT_SHAPE = [(Args.DIM_ALL,), (85,)]`
- `Args.DIM_ALL = 3910`

也就是说，每帧模型输入由两部分组成：

1. **3910 维观测特征**
2. **85 维合法动作相关输入**

拼起来就是：

- `3910 + 85 = 3995` 维

---

## 3.3 3910 维观测是怎么组成的

### 单位总特征：2610 维

`ObsBuilder.build_observation()` 会拼这些对象：

- 我方英雄 1 个
- 敌方英雄 1 个
- 我方小兵最多 4 个
- 敌方小兵最多 4 个
- 河蟹 1 个
- 我方外塔 1 个
- 敌方外塔 1 个

也就是：

- `2 x 英雄 + 8 x 小兵 + 1 x 河蟹 + 2 x 防御塔`

对应总维度：

- `2610`

### 子弹总特征：1300 维

还会额外拼接最多 10 个敌方飞行物：

- 最近 9 个敌方英雄子弹
- 最近 1 个敌方防御塔子弹

对应总维度：

- `1300`

### 合计

- 单位特征：2610
- 子弹特征：1300
- 总计：`3910`

---

## 3.4 各类对象的具体输入内容

### 1. 单位通用特征 `DIM_UNIT = 167`

所有单位公用一套基础信息，来自 `process_unit()`：

- 相对位置
- 全局位置
- 相对距离/全局位置离散化
- 当前血量比例
- 离散化血量桶
- mark 层数信息

这里的 mark 是针对特定英雄机制做的离散特征，代码里显式支持：

- 后羿 mark
- 李元芳 mark
- 其他未知 mark 占一个兜底槽位

### 2. 英雄特征 `DIM_HERO = 347`

`process_hero()` 在通用特征基础上再加入：

- 英雄类型
  - `169 / 173 / 174` 被编码成 `(-1, 0, 1)` 这种单值表示
- 行为状态
  - 如 `State_Dead`、`State_Idle`、`Direction_Move`、`Normal_Attack`、`UseSkill_1/2/3`
- 法力值比例和离散桶
- 5 个技能槽位的 CD 特征
  - 1/2/3 技能
  - 闪现
  - 回复
- 等级 one-hot
- 金币变化特征
  - 两帧之间的金币增量离散化
  - 当前总金币比例
- 是否在草丛
- 是否在敌方塔攻击范围内
- 是否为敌方塔的当前攻击目标
- buff one-hot

### 3. 小兵特征 `DIM_SOLDIER = 175`

`process_soldier()` 会先按距离从近到远排序，再取最多 4 个。

每个小兵特征包括：

- 小兵行为
- 小兵类型
  - 近战 / 远程 / 炮车
- 是否在对方塔攻击范围内
- 是否为对方塔攻击目标
- 单位通用特征

### 4. 河蟹特征 `DIM_RIVER_CRAB = 172`

包括：

- 河蟹行为状态
- 单位通用特征

### 5. 防御塔特征 `DIM_ORGAN = 172`

`process_sub_tower()` 在通用特征上加入：

- 当前攻击目标类型
  - 无目标 / 英雄 / 小兵
- 塔后是否有血包
- 下次血包生成剩余时间

### 6. 子弹特征 `DIM_BULLET = 130`

`process_bullets()` 只保留最近的敌方飞行物。

每个子弹包括：

- 来源技能槽位
  - 普攻 / 1 技能 / 2 技能 / 3 技能 / 其他
- 位置与距离相关特征

---

## 4. 奖励是什么

奖励逻辑在：

- `cankoa/hok_semi/code/agent_ppo/feature/reward_manager.py`

权重在：

- `cankoa/hok_semi/code/agent_ppo/conf/conf.py`

当前 reward 项如下：

- `hp_point`: `2.0`
- `tower_hp_point`: `10.0`
- `money`: `4e-3`
- `exp`: `4e-3`
- `ep_rate`: `0.75`
- `death`: `-1.0`
- `kill`: `-0.6`
- `last_hit`: `0.5`
- `forward`: `0.01`

---

## 4.1 各奖励项含义

### `hp_point`

英雄血量奖励。

代码里不是直接用线性血量比，而是：

```text
(hp / max_hp)^(1/4)
```

然后再做我方减敌方的差分奖励。

### `tower_hp_point`

塔血奖励。

使用我方塔血比例与敌方塔血比例做差，再取前后帧增量。

### `money`

金币奖励。

使用双方英雄累计金币差分的前后帧变化。

### `exp`

经验奖励。

经验不是直接用当前等级，而是：

- 先累计历史等级所需经验
- 再加当前等级内经验

然后做双方差分。

另外：

- 英雄到 15 级后，这项奖励直接置 0

### `ep_rate`

法力值奖励。

只取我方当前 `ep / max_ep` 的变化量，不做敌我差分。

如果英雄死亡或 `max_ep == 0`，则记为 0。

### `death`

死亡次数差分奖励。

死亡增加会触发负奖励。

### `kill`

击杀次数差分奖励。

注意这份代码里 `kill` 的权重是负数 `-0.6`。这和常见直觉不一样，但它配合 `death`、`hp_point`、`money`、`exp` 共同构成整体 shaping，代码就是这么写的。

### `last_hit`

补刀奖励。

通过 `frame_action.dead_action` 判断：

- 我方击杀小兵：`+1`
- 敌方击杀小兵：`-1`

再乘权重 `0.5`。

### `forward`

前进奖励。

逻辑和你当前仓库里的简化版基本同源：

- 计算英雄到敌方塔的距离
- 计算我方塔到敌方塔的距离
- 如果英雄几乎满血且比我方塔更靠前，则给前压奖励

另外还有一个限制：

- `GameConfig.REMOVE_FORWARD_AFTER = 1000`

也就是超过一定帧数后，`forward` 奖励会被关掉。

---

## 4.2 时间衰减

这套 reward 还带时间衰减：

- `TIME_SCALE_ARG = 8000`

实际是把大多数奖励项乘上：

```text
0.6 ^ (frame_no / 8000)
```

也就是越到后期，reward shaping 越弱。

当前 `REWARD_WITHOUT_TIME_SCALE` 是空集合，所以默认这些奖励项都会参与时间衰减。

---

## 4.3 最终 reward

`reward_manager.get_reward()` 会同时记录：

- 每个奖励项的原始值：`xxx_origin`
- 加权后的值：`xxx_weight`
- 最终总和：`reward_sum`

训练实际使用的是：

- `reward["reward_sum"]`

---

## 5. 输出是什么

这套参考实现的动作输出格式，和你当前根目录那套 PPO 一样，仍然是：

- **6 个离散动作头**
- **1 个 value 头**

动作头配置：

- `LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]`

也就是每帧输出：

1. 12 维
2. 16 维
3. 16 维
4. 16 维
5. 16 维
6. 9 维
7. value 1 维

最终给环境的仍然是一个 6 维离散动作列表。

最后一个动作头仍然依赖第一个动作头：

- 先采样前 5 个头
- 再根据第 1 个动作头筛 target 的合法动作

所以它本质上还是一个带条件 target 选择的动作空间。

---

## 6. 训练样本里保存了什么

样本打包在：

- `cankoa/hok_semi/code/agent_ppo/feature/definition.py`

每帧样本包含：

- `feature`
- `legal_action`
- `reward_sum`
- `advantage`
- `action`
- `prob`
- `sub_action`
- `is_train`
- `lstm_info`

序列配置：

- `LSTM_TIME_STEPS = 16`
- `LSTM_UNIT_SIZE = 512`

所以 learner 侧吃的是：

- **16 帧拼接样本 + 初始 LSTM 状态**

---

## 7. 模型结构大致是什么

模型在：

- `cankoa/hok_semi/code/agent_ppo/model/model.py`

它不是简单把 3910 维直接喂进 MLP，而是先按语义拆块处理：

- 英雄特征单独编码
- 小兵特征单独编码
- 防御塔特征单独编码
- 河蟹单独编码
- 子弹单独编码
- 位置特征和单位非位置特征分别过共享 MLP
- 再做拼接
- 再经过 LSTM 和旁路 MLP 融合
- 最后输出 policy 和 value

代码里 `Config.MULTI_HEAD = True`，但当前主入口仍然是：

- `agent_ppo/agent.py`
- `agent_ppo/model/model.py`

不是 `agent_multi_head.py` 那条支线。

---

## 8. 关键维度汇总

这套参考实现里最重要的维度可以直接记成下面这些：

- `DIM_UNIT = 167`
- `DIM_HERO = 347`
- `DIM_SOLDIER = 175`
- `DIM_RIVER_CRAB = 172`
- `DIM_ORGAN = 172`
- `DIM_ALL_UNITS = 2610`
- `DIM_BULLET = 130`
- `DIM_BULLETS = 1300`
- `DIM_ALL = 3910`
- `legal_action` 相关维度 = `85`
- 单帧拼接输入 = `3995`

---

## 9. 和你当前项目的对比

你当前根目录那套 PPO 更像是极简教学版：

- 输入只有 10 维
- 奖励只有 2 项
- 特征只看己方英雄和敌方塔

`hok_semi` 这套更像真正打比赛时的版本：

- 输入扩展到 3910 维
- 奖励项更完整
- 显式建模了英雄、小兵、河蟹、塔、子弹、buff、mark、草丛、血包等信息

如果你现在要从参考实现里“借设计”，最值得迁移的部分是：

1. `ObsBuilder` 的观测拆解思路
2. `RewardManager` 的多项 reward shaping
3. 小兵、塔、子弹这种对象级特征组织方式

---

## 10. 一句话总结

`hok_semi` 这份参考实现本质上是一套更完整的 1v1 PPO 方案：

- **输入**：3910 维手工构造大观测 + 85 维合法动作信息
- **奖励**：血量、塔血、金币、经验、法力、击杀、死亡、补刀、前进，多项加权并带时间衰减
- **输出**：和当前仓库一样，仍然是 6 个离散动作头加 1 个 value 头

如果后面你要，我可以继续把这份参考实现里的：

1. `obs_builder.py` 画成结构图
2. `reward_manager.py` 拆成表格
3. 再对照你当前仓库，给出一版“该抄哪些、先抄哪些”的迁移方案
