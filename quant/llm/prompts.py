"""AI 策略生成的提示词与输出解析。

两种生成模式：
- code：直接生成可执行的 Python 策略源码（默认，功能不受限，用户可手改）
- rule：生成声明式 JSON 配置（复用 quant.strategies.llm.schema + validator 校验链）

三种对话意图（intent，与 mode 正交）：
- write：写策略（首轮或改需求重写）
- revise：基于当前代码继续改（用户反复聊天修改）
- ideas：只聊思路，不产出策略（AI 给新想法，用户挑一条「按这个写」）
"""
from __future__ import annotations

import json
import re
from typing import Any

from quant.strategies.llm.schema import (BUILTIN_REFS, INDICATOR_WHITELIST, OPERATORS,
                                        POSITION_MODES, RuleStrategyConfig,
                                        RuleParams)

MODES = {"code": "Python 代码策略", "rule": "声明式规则策略（JSON）"}

# 对话意图：用户这轮到底想干什么。三种意图决定系统提示与输出契约。
INTENTS = {"write": "写策略", "revise": "改当前策略", "ideas": "聊思路（不产出代码）"}

# ====================== 代码模式 ======================

CODE_SYSTEM_PROMPT = """你是 QuantSim 量化系统的策略工程师。你的任务：把用户的自然语言交易想法，写成**可直接运行的 Python 策略代码**。

# 运行环境
- 已预先注入以下名字，**不要重复 import**：`pd`(pandas)、`np`(numpy)、`math`、`BaseStrategy`、`BaseModel`、`Field`、`ClassVar`、`indicators`（量化指标库）
- 允许 import 的模块仅限：pandas、numpy、math、statistics、typing、dataclasses、pydantic、quant.core.indicators、quant.strategies.base
- 禁止：文件/网络/系统调用（open、eval、exec、__import__、os、sys、subprocess 等一律不允许）

# 必须遵守的代码骨架
```python
class Strategy(BaseStrategy):
    key = "英文短横线key"
    name = "中文策略名"
    description = "一句话说明逻辑"

    class Params(BaseModel):
        window: int = Field(20, ge=2, le=500)      # 每个参数必须给 default 与范围
        threshold: float = Field(0.02, ge=0.0, le=1.0)

    def generate_signals(self, bars, ctx):
        # bars: DataFrame，index 是时间，列有 open/high/low/close/volume
        close = bars["close"]
        window = self.params.window            # 参数必须这样取：self.params.字段名（小写 params）
        n = len(bars)
        target = pd.Series(0.0, index=bars.index)
        reason = pd.Series("", index=bars.index)
        # ... 你的逻辑，给 target 赋目标仓位 ...
        return pd.DataFrame({"target_weight": target, "reason": reason, "confidence": 1.0})
```

# 硬性契约（违反会导致校验失败）
1. **必须**返回 DataFrame，且列名恰好为 `target_weight`、`reason`、`confidence`，行数与 `bars` 完全一致。
2. `target_weight` 取值 0.0~1.0（做空为 -1.0~0.0，需用户明确要求），表示该 K 线收盘后应持有的目标仓位比例。
3. 信号必须**逐根确定**（用 for 循环维护状态，或向量化后 ffill），不要有前视偏差：第 i 根只能用到第 i 根及以前的数据。
4. `Params` 里每个字段都要有 `default` 与 `ge/le` 范围（UI 会据此生成滑块）。
5. **参数一律用 `self.params.字段名` 读取（小写 `params`，不是 `self.Params`、不是 `Params.字段名`）**。`self.params` 是运行时已校验的参数对象（pydantic 实例）。
6. 只输出**一个类**，类名 `Strategy`。

# 可用指标（indicators.xxx，返回 Series 或 DataFrame）
- `indicators.sma(close, window)` / `indicators.ema(close, window)`
- `indicators.macd(close, fast=12, slow=26, signal=9)` → DataFrame[macd, signal, hist]
- `indicators.rsi(close, period=14)`
- `indicators.bollinger(close, period=20, num_std=2.0)` → DataFrame[up, mid, low]
- `indicators.atr(bars, period=14)`（需 high/low/close）
- `indicators.donchian(close, period)` → DataFrame[high, low]（已 shift(1)，无前视）
- `indicators.roc(close, window)` / `indicators.zscore(close, window)`
也可以直接用 pandas/numpy 自己算。

# 输出格式
先输出一段不超过 120 字的中文说明（策略思路、适用行情），然后输出代码块：
```python
<完整可运行代码>
```
不要输出多余的内容。"""

CODE_USER_TEMPLATE = """请按下面的需求写策略代码。

需求：{request}

{extra}"""

CODE_REVISE_TEMPLATE = """这是上一版的策略代码，请按新的要求修改后重新输出完整代码（不要只给片段）。

=== 当前代码 ===
```python
{code}
```

=== 修改要求 ===
{request}
{extra}"""

RULE_USER_TEMPLATE = """请按下面的需求生成策略配置。

需求：{request}

{extra}"""

RULE_REVISE_TEMPLATE = """这是当前已生成的策略配置，请按新的要求修改后**重新输出完整配置**（不要只给片段）。

=== 当前配置 ===
```json
{config}
```

=== 修改要求 ===
{request}
{extra}"""

# ====================== 规则模式 ======================

_RULE_SCHEMA_DOC = json.dumps({
    "name": "策略名",
    "description": "一句话说明",
    "indicators": [{"name": "sma", "params": {"window": 20}, "as": "ma20"}],
    "entry_long": {"logic": "all", "conditions": [{"lhs": "close", "op": "crosses_above", "rhs": "ma20"}]},
    "exit_long": {"logic": "all", "conditions": [{"lhs": "close", "op": "crosses_below", "rhs": "ma20"}]},
    "entry_short": None,
    "exit_short": None,
    "position": {"mode": "fixed_pct", "value": 1.0, "max_position": 1.0},
    "params": {"allow_short": False, "only_long": True, "max_hold_bars": None},
}, ensure_ascii=False, indent=2)

RULE_SYSTEM_PROMPT = f"""你是 QuantSim 量化系统的策略配置生成器。把用户的自然语言交易想法转成**声明式 JSON 配置**。

# 输出 JSON 结构（严格照抄字段名）
{_RULE_SCHEMA_DOC}

# 指标白名单（indicators[].name 只能取这些）
{json.dumps({k: {"label": v["label"], "params": v["params"], "outputs": v["outputs"]}
             for k, v in INDICATOR_WHITELIST.items()}, ensure_ascii=False, indent=1)}

# 可引用字段
- 内置：{BUILTIN_REFS}
- 指标输出：由 `as` 前缀按 outputs 模板展开（如 as="ma20" 的 sma → 输出 "ma20"；boll 的 as="bb" → "bb_up"/"bb_mid"/"bb_low"；macd 的 as="m" → "m"/"m_signal"/"m_hist"；donchian 的 as="dc" → "dc_up"/"dc_low"）

# 操作符
{OPERATORS}

# 仓位模式
{POSITION_MODES}

# 规则
1. 只输出 JSON 对象，不要 markdown 代码块、不要解释。
2. 条件里的 lhs/rhs 引用必须先定义（先写在 indicators 里，或用内置字段）。
3. `op` 为 between/not_between 时，rhs 写成两元素数组 [下限, 上限]。
4. `op` 为 crosses_above/crosses_below 时，rhs 必须是对已定义输出的引用字符串。
5. allow_short 与 only_long 不能同时为 true。
"""

# ====================== 思路模式（只聊想法，不产出策略） ======================

IDEA_SYSTEM_PROMPT = """你是 QuantSim 量化系统的策略研究顾问。用户正在构思量化策略，你的职责是**帮他发散思路、提出新想法**，而不是直接写代码。

# 输出格式（严格遵守）
1. 先用 1~3 句中文点评用户当前的想法：亮点是什么、风险在哪里（要具体，别空话）。
2. 然后输出 2~4 条**新的策略想法**，每条是一个 JSON 对象，字段：
   - "title"：不超过 14 个字的短标题
   - "logic"：40~110 字，说清「用什么指标/条件、什么时候买、什么时候卖、怎么控仓」
   - "scene"：不超过 30 字，适合什么行情/周期
   - "risk"：不超过 30 字，主要风险或失效场景
3. 最后一行单独输出 JSON 数组，把这几个对象包起来：
```json
[{"title": "...", "logic": "...", "scene": "...", "risk": "..."}]
```

# 硬性要求
- 想法必须**能落地**：只能用常见技术指标（均线/MACD/RSI/布林带/ATR/成交量/突破/动量/回归）与资金管理（仓位、止损）表达，不要依赖基本面或无法获取的数据。
- 每条想法的买卖条件必须**方向明确、可判断**，避免「适当」「择机」这类模糊词。
- 想法之间要有差异（趋势/震荡/过滤/风控不同角度），不要换参数重说同一件事。
- 如果用户已有策略，想法应是它的**补充或替代方案**，而不是复述。
- 绝对不要输出策略代码，也不要输出规则 JSON 配置。
"""

IDEA_USER_TEMPLATE = """请围绕下面这个方向帮我发散策略思路。

用户想法：{request}
{extra}"""

IDEA_REVISE_TEMPLATE = """我当前这版策略是这样的，请基于它给出改进或替代思路。

=== 当前策略代码 ===
```python
{code}
```

=== 我的想法 ===
{request}
{extra}"""

# 用户点某条思路的「按这个写」时附带的前缀，让模型直接写代码而不是再发散。
IDEA_PICK_PREFIX = "就按下面这个思路写策略，不要发散、不要给备选方案：\n"


def build_messages(mode: str, request: str, *, code: str | None = None,
                   history: list[dict[str, str]] | None = None,
                   context: dict[str, Any] | None = None,
                   intent: str = "write",
                   config: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """组装对话消息。

    - mode="code"：带骨架与契约的系统提示
    - mode="rule"：带 JSON schema 的系统提示
    - intent="ideas"：换「策略顾问」系统提示，只发散思路
    - code 非空 → 走「按意见修改」的分支
    - history → 多轮对话上下文（仅保留 user/assistant 文本）
    - config → 规则模式已生成的配置（改配置时带上下文，避免模型推倒重来）
    """
    intent = intent if intent in INTENTS else "write"
    if intent == "ideas":
        system = IDEA_SYSTEM_PROMPT
        extra = _context_block(context)
        if code:
            messages: list[dict[str, str]] = [{"role": "system", "content": system}]
            for m in (history or [])[-8:]:
                role = m.get("role")
                content = (m.get("content") or "").strip()
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": IDEA_REVISE_TEMPLATE.format(
                request=request, code=code, extra=extra)})
            return messages
        return [{"role": "system", "content": system},
                {"role": "user",
                 "content": IDEA_USER_TEMPLATE.format(request=request, extra=extra)}]

    system = CODE_SYSTEM_PROMPT if mode == "code" else RULE_SYSTEM_PROMPT
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for m in (history or [])[-8:]:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    extra = _context_block(context)
    if mode == "rule":
        extra = _append_config_block(extra, config)
    if code and mode == "code":
        messages.append({"role": "user",
                         "content": CODE_REVISE_TEMPLATE.format(request=request, code=code,
                                                                extra=extra)})
    elif code and mode == "rule":
        messages.append({"role": "user",
                         "content": RULE_REVISE_TEMPLATE.format(request=request,
                                                                config=json.dumps(config, ensure_ascii=False, indent=2),
                                                                extra=extra)})
    else:
        messages.append({"role": "user",
                         "content": CODE_USER_TEMPLATE.format(request=request, extra=extra)})
    return messages


def _append_config_block(extra: str, config: dict[str, Any] | None) -> str:
    """规则模式下把当前配置附带给模型，让「改一处」不至于整份重写。"""
    if not config:
        return extra
    block = ("\n# 当前配置\n这是已生成的配置，请在此基础上按用户要求**最小化修改**"
             "（不要改变无关的指标、条件与参数）：\n```json\n"
             + json.dumps(config, ensure_ascii=False, indent=2) + "\n```")
    return (extra or "") + block


def _context_block(context: dict[str, Any] | None) -> str:
    """可选的行情/标的上下文，帮助模型设定合适的窗口与阈值。"""
    if not context:
        return ""
    lines = []
    if context.get("symbol"):
        lines.append(f"- 标的：{context['symbol']}")
    if context.get("timeframe"):
        lines.append(f"- 周期：{context['timeframe']}")
    if context.get("bars_count"):
        lines.append(f"- 可用历史K线数：{context['bars_count']}（参数窗口别超过它的 1/3）")
    if context.get("notes"):
        lines.append(f"- 补充说明：{context['notes']}")
    return ("\n# 行情上下文\n" + "\n".join(lines) + "\n") if lines else ""


# ====================== 输出解析 ======================

_FENCE = re.compile(r"```(?:python|py|json)?\s*(.*?)```", re.S | re.I)


def extract_code(text: str) -> str:
    """从模型输出里抽出 Python 代码（优先代码块，否则找 class 开头的一段）。"""
    text = text or ""
    blocks = [b.strip() for b in _FENCE.findall(text) if b.strip()]
    code_blocks = [b for b in blocks if "def generate_signals" in b or "class " in b]
    if code_blocks:
        return max(code_blocks, key=len)
    if blocks:
        return max(blocks, key=len)
    idx = text.find("class ")
    if idx >= 0:
        return text[idx:].strip()
    return text.strip()


def extract_json(text: str) -> dict[str, Any]:
    """从模型输出里抽出 JSON 对象。"""
    text = (text or "").strip()
    blocks = [b.strip() for b in _FENCE.findall(text) if b.strip()]
    candidates = blocks + [text]
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:  # noqa: BLE001
            pass
        start, end = cand.find("{"), cand.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(cand[start:end + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:  # noqa: BLE001
                continue
    raise ValueError("模型输出中没有找到合法 JSON")


def explanation_of(text: str, code: str = "", mode: str = "code") -> str:
    """提取模型给出的中文说明（去掉代码块本身）。"""
    body = (text or "").replace(code, "") if code else (text or "")
    body = _FENCE.sub("", body).strip()
    return body[:600]


def parse_ideas(text: str) -> list[dict[str, str]]:
    """从思路模式的输出里解析出想法列表。

    容错：JSON 数组在任意围栏里、单个对象、甚至裸文本都能尽量救回来。
    每条只保留 title/logic/scene/risk 四个字符串字段并做长度截断。
    """
    text = text or ""
    candidates: list[Any] = []
    for block in _FENCE.findall(text):
        block = block.strip()
        if not block:
            continue
        start, end = block.find("["), block.rfind("]")
        if start >= 0 and end > start:
            try:
                obj = json.loads(block[start:end + 1])
                if isinstance(obj, list):
                    candidates.append(obj)
            except Exception:  # noqa: BLE001
                pass
    start, end = text.find("["), text.rfind("]")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, list):
                candidates.append(obj)
        except Exception:  # noqa: BLE001
            pass
    # 找最大的那个候选（模型可能在说明文字里也提到方括号）
    best: list[Any] = []
    for cand in candidates:
        if len(cand) > len(best):
            best = cand

    ideas: list[dict[str, str]] = []
    for item in best if isinstance(best, list) else []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("name") or "").strip()
        logic = str(item.get("logic") or item.get("idea") or item.get("detail") or "").strip()
        if not title and not logic:
            continue
        ideas.append({
            "title": title[:40],
            "logic": logic[:600],
            "scene": str(item.get("scene") or "").strip()[:60],
            "risk": str(item.get("risk") or "").strip()[:60],
        })
        if len(ideas) >= 6:
            break
    return ideas


def ideas_text(text: str) -> str:
    """思路模式给用户看的说明文字（去掉 JSON 数组本身）。"""
    body = text or ""
    start, end = body.find("["), body.rfind("]")
    if start >= 0 and end > start:
        body = body[:start] + body[end + 1:]
    return _FENCE.sub("", body).strip()[:1200]


def rule_messages_json_default() -> dict[str, Any]:
    """规则模式的完整默认配置（供 UI 预览）。"""
    return RuleStrategyConfig(name="未命名策略", params=RuleParams()).model_dump(by_alias=True)
