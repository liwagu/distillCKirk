"""输出守卫：在句子进入 TTS 之前做确定性检查。

为什么不靠提示词：这些约束在 §0/§1/§2 里已经写了三遍，模型每次换个形式绕回来
（禁了 "You're not as attractive" 就改说 "if you're single by 30, you've got…"）。
5000 token 的系统提示下，细粒度否定约束的遵守率本来就不可靠。
**确定性的规则属于代码，不属于提示词** —— 和"回合计数不放 prompt"是同一个道理。

逐句检查，违规的整句丢弃。回复本来就是多句，丢一句不影响可听性，
而且比重新生成快得多（重新生成要再付一次 TTFT）。
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger("voxck.guard")

# 1) 把人生判词甩到用户脸上。包含直陈和假设两种形态。
_PERSONAL = re.compile(
    r"\byou(?:'re|\s+are|\s+were|'ve\s+got|\s+have\s+got|'ll|\s+will)\b"
    r"[^.!?]{0,80}?"
    r"\b(?:attractive|peak|prime|unhappy|lonely|empty|fulfilled|regret|"
    r"running\s+out|too\s+late|never\s+(?:have|having)\s+kids|childless|"
    r"biological\s+clock|fertility|\d{1,3}\s*%)",
    re.I)
# "if you're in your early thirties…" 这类把用户代入的假设
_HYPO = re.compile(
    r"\bif\s+you(?:'re|\s+are|'ve|\s+have)\b[^.!?]{0,100}?"
    r"\b(?:thirt|twent|single|career|kids|children|married|unmarried)",
    re.I)
# 2) 脚手架泄漏
_TAGS = re.compile(r"\bG\d{2}\b|\[[HM]\]|§\s*\d|\bthe ledger\b|\bcite-id\b", re.I)


def check(sentence: str) -> str | None:
    """返回违规原因；干净则返回 None。"""
    if _TAGS.search(sentence):
        return "标签泄漏"
    if _PERSONAL.search(sentence):
        return "第二人称人生判词"
    if _HYPO.search(sentence):
        return "把用户代入的假设"
    return None


def clean(sentence: str) -> tuple[str | None, str | None]:
    """(可朗读的句子, 丢弃原因)。违规则返回 (None, 原因)。"""
    why = check(sentence)
    if why:
        log.warning("丢弃违规句 [%s]: %s", why, " ".join(sentence.split())[:110])
        return None, why
    return sentence, None


class WordBudget:
    """回合词数上限。超了就在句子边界截断，绝不截在半句。"""

    def __init__(self, ceiling: int = 90):
        self.ceiling = ceiling
        self.used = 0

    def allow(self, sentence: str) -> bool:
        n = len(sentence.split())
        if self.used and self.used + n > self.ceiling:
            return False          # 已经说过话了，这句会超 → 停在这里
        self.used += n
        return True


_RESTATEMENT = re.compile(
    # 明确指向上一句/上一轮；单个 explain 或 repeat 可能只是新问题或抱怨。
    r"(?:^|\bplease\s+|\b(?:can|could|would|will)\s+you\s+)"
    r"(?:repeat|rephrase|restate)\s+(?:that|it|this|your\s+(?:point|answer|statement|position|claim)|"
    r"(?:the\s+)?(?:previous|last)\s+(?:point|answer|statement|sentence)|"
    r"the\s+(?:point|answer|statement|sentence)|what\s+you\s+(?:said|mean))\b|"
    r"(?:^|\bplease\s+|\b(?:can|could|would|will)\s+you\s+)quote\b|"
    r"\b(?:explain|clarify)\s+(?:that|it|this|your\s+(?:point|answer)|"
    r"(?:the\s+)?(?:previous|last)\s+(?:point|answer))(?:\s+again)?\b|"
    r"\bsay\s+(?:that|it|this|your\s+(?:point|answer))\s+again\b|"
    r"\b(?:didn['’]?t|did\s+not|don['’]?t|do\s+not)\s+(?:follow|understand|catch)\b|"
    r"\bwhat\s+do\s+you\s+mean\b|^\s*again[?!.\s]*$",
    re.I)
_NO_RESTATEMENT = re.compile(r"\b(?:don't|dont|do\s+not|stop|never|quit)\s+"
                            r"(?:\w+\s+){0,5}(?:repeat(?:ing)?|rephras(?:e|ing)|"
                            r"restat(?:e|ing)|quot(?:e|ing))\b", re.I)


def allows_restatement(user_text: str) -> bool:
    """明确请求重说/解释时允许复述；抱怨重复不会解除复读检查。"""
    if _NO_RESTATEMENT.search(user_text):
        return False
    return bool(_RESTATEMENT.search(user_text) or any(
        phrase in user_text for phrase in ("再说一遍", "解释一下", "没听懂", "什么意思")))


class RepeatGuard:
    """整场会话内的复读检测。

    提示词里写"不要重复"不可靠 —— 模型背下示范台词后会一字不差地重复
    （实测它把 §0 的两句示范当剧本，连说两次 "You walked up to the table. Argue."）。
    复读是最快让人出戏的东西，所以在代码里拦。

    比对时归一化（小写、去标点、压空白），并对近似重复也拦：
    两句词集合重合度超过阈值即视为同一句换个说法。
    """

    def __init__(self, overlap: float = 0.8, min_words: int = 3,
                 max_drops_per_turn: int = 8):
        self.seen: list[set[str]] = []
        self.exact: set[str] = set()
        self.overlap = overlap
        self.min_words = min_words
        # 每轮丢弃上限，纯粹是失控保护。编排层可在全被丢光时重生成一次，
        # 所以这里可以放宽 —— 之前设 2 太紧，
        # 撞上限后剩余句子不再检查，整段重复的回复照样放行。
        self.max_drops = max_drops_per_turn
        self._turn_drops = 0

    def new_turn(self) -> None:
        self._turn_drops = 0

    @staticmethod
    def _norm(s: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9\s]+", "", s.lower()).split())

    def is_exact_repeat(self, sentence: str, min_words: int = 6) -> bool:
        """首句只拦已出现过的完整长句；短回应和近似措辞不增加首音等待。"""
        n = self._norm(sentence)
        return len(n.split()) >= min_words and n in self.exact

    def is_repeat(self, sentence: str) -> bool:
        if self._turn_drops >= self.max_drops:
            return False            # 本轮已丢够，剩下的放行，避免整轮静音
        n = self._norm(sentence)
        if not n:
            return False
        words = set(n.split())
        # 太短的句子（"No." "Right."）本来就会自然重复，不拦
        if len(words) < self.min_words:
            return False
        hit = n in self.exact or any(
            len(words & prev) / max(len(words), len(prev)) >= self.overlap
            for prev in self.seen)
        if hit:
            self._turn_drops += 1
        return hit

    def add(self, sentence: str) -> None:
        n = self._norm(sentence)
        if not n:
            return
        self.exact.add(n)
        self.seen.append(set(n.split()))


# TTS 会把 markdown 符号念出来或读出怪音。人设里写了"不要 markdown"，
# 模型照样用 *强调*（实测 "*can't*" "*forces*"）—— 又一条确定性规则，归代码管。
_MD = [
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),     # **bold**
    (re.compile(r"(?<!\w)\*(.+?)\*(?!\w)"), r"\1"),   # *emphasis*
    (re.compile(r"(?<!\w)_(.+?)_(?!\w)"), r"\1"),     # _emphasis_
    (re.compile(r"`+([^`]+)`+"), r"\1"),       # `code`
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),    # [text](url)
    (re.compile(r"^\s{0,3}#{1,6}\s+", re.M), ""),     # # 标题
    (re.compile(r"^\s{0,3}[-*+]\s+", re.M), ""),      # - 列表
    (re.compile(r"\s{2,}"), " "),
]


def for_speech(text: str) -> str:
    """把一句话整理成适合朗读的形式。"""
    for pat, rep in _MD:
        text = pat.sub(rep, text)
    # 破折号读起来是停顿，保留；但连续的要压掉
    text = re.sub(r"—{2,}", "—", text)
    return text.strip()
