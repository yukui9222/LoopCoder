"""L1-L3 程序化上下文压缩 pipeline。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from loopcoder.context.archive import ArchiveIntegrityError, ToolResultArchive
from loopcoder.context.checkpoint import CheckpointIndex
from loopcoder.context.content.build import BuildOutputRouteCompressor
from loopcoder.context.content.code import SourceCodeRouteCompressor
from loopcoder.context.content.compressors import PlainTextRouteCompressor, compact_old_task_part
from loopcoder.context.content.detector import (
    is_already_compacted,
    is_old_task_part,
)
from loopcoder.context.content.diff import GitDiffRouteCompressor
from loopcoder.context.content.html import HtmlRouteCompressor
from loopcoder.context.content.json import JsonRouteCompressor
from loopcoder.context.content.router import RouteCompactRouter, RouteContentType
from loopcoder.context.content.search import SearchResultsRouteCompressor
from loopcoder.context.identity import session_view_fingerprint
from loopcoder.context.models import AgentMessage, MessagePart, SessionView, latest_user_message_id, utc_now_iso
from loopcoder.context.token_budget import estimate_text_tokens
from loopcoder.context.tool_lifecycle import (
    ToolResultLifecycle,
    ToolResultLifecycleRecord,
    index_tool_result_lifecycles,
)
from loopcoder.context.versions import COMPACTION_STRATEGY_VERSION, CONTEXT_EVENT_SCHEMA_VERSION

CompactionLevel = Literal["l1", "l2", "l3"]


@dataclass(slots=True)
class CompactionRequest:
    view: SessionView
    active_task_hash: str | None
    target_tokens: int
    current_turn: int
    estimate_tokens: Callable[[SessionView], int]
    consumed_tool_result_part_ids: frozenset[str]
    enabled_levels: tuple[CompactionLevel, ...] = ("l1", "l2", "l3")
    required_levels: tuple[CompactionLevel, ...] = ()
    l2_result_target_tokens: int | None = None
    force_route_current_text: bool = False
    force_old_task_compaction: bool = False


@dataclass(slots=True)
class CompactionEvent:
    input_fingerprint: str
    before_tokens: int
    after_tokens: int
    levels_attempted: list[str]
    stopped_at: str
    changed_parts: int
    reason: str = "programmatic_compaction"
    target_tokens: int = 0
    source_part_ids: list[str] = field(default_factory=list)
    output_part_ids: list[str] = field(default_factory=list)
    replacements: list[dict[str, object]] = field(default_factory=list)
    checkpoint_id: str | None = None
    strategy_version: str = COMPACTION_STRATEGY_VERSION
    event_version: str = CONTEXT_EVENT_SCHEMA_VERSION
    llm_used: bool = False
    success: bool = True
    error: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    noop: bool = False
    deduped: bool = False
    lifecycle_counts: dict[str, int] = field(default_factory=dict)
    level_metrics: dict[str, dict[str, int]] = field(default_factory=dict)
    archive_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CompactionResult:
    view: SessionView
    event: CompactionEvent


@dataclass(slots=True)
class CompactionPipeline:
    root: str | Path
    large_tool_result_tokens: int = 1200
    cold_turn_distance: int = 8
    cold_preview_chars: int = 160
    _seen_noop_fingerprints: set[str] = field(default_factory=set)

    def compact(self, request: CompactionRequest) -> CompactionResult:
        view = _clone_view(request.view)
        #给输入的会话消息计算一个指纹，用来识别重复的无变化压缩
        input_fingerprint = session_view_fingerprint(request.view)
        #估算压缩前的大小
        before_tokens = request.estimate_tokens(view)
        # 判断工具结果属于 fresh、stale、derived 等哪种状态，供后续筛选使用
        """
        从旧到新扫描，遇到新的文件读取，就检查此前的读取是否应变成 superseded（已被后面的读取取代）。
        遇到成功的文件修改记录，就检查此前同一文件的读取是否应变成 stale（可能过时的旧读取）。
        扫描结束后，再检查哪些旧派生输出应变成 duplicate（与后面的输出完全重复）。
        然后返回分类结果，供 L1、L2、L3 使用。
        """
        lifecycle_records = index_tool_result_lifecycles(
            _effective_tail_messages(view),
            current_turn=request.current_turn,
        )
        lifecycle_counts = _lifecycle_counts(lifecycle_records)
        required_levels = set(request.required_levels).intersection(request.enabled_levels)
        per_result_target = _per_result_target(
            request.l2_result_target_tokens,
            fallback=self.large_tool_result_tokens,
        )
        #检查是否仍有陈旧、重复或单条过大的结果需要清理
        has_l3_mandatory_candidates = _has_l3_mandatory_candidates(
            _effective_tail_messages(view),
            lifecycle_records=lifecycle_records,
            current_turn=request.current_turn,
            consumed_tool_result_part_ids=request.consumed_tool_result_part_ids,
        )
        #是否存在可处理的 derived 结果，其单条大小仍超过目标
        has_l3_per_result_pressure = _has_l3_per_result_pressure(
            _effective_tail_messages(view),
            lifecycle_records=lifecycle_records,
            current_turn=request.current_turn,
            per_result_target=per_result_target,
            consumed_tool_result_part_ids=request.consumed_tool_result_part_ids,
        )

        """
        总 token 已经达标
        且没有强制清理旧任务
        且没有必须尝试的层
        且没有已启用 L3 需要尝试清理的陈旧、覆盖、重复结果
        且没有已启用 L2/L3 需要处理的单条过大结果
        才直接返回
        """
        if (
            before_tokens <= request.target_tokens
            and not request.force_old_task_compaction
            and not required_levels
            and not ("l3" in request.enabled_levels and has_l3_mandatory_candidates)
            and not ({"l2", "l3"}.intersection(request.enabled_levels) and has_l3_per_result_pressure)
        ):
            deduped = input_fingerprint in self._seen_noop_fingerprints
            self._seen_noop_fingerprints.add(input_fingerprint)
            return CompactionResult(
                view=view,
                event=CompactionEvent(
                    input_fingerprint=input_fingerprint,
                    before_tokens=before_tokens,
                    after_tokens=before_tokens,
                    levels_attempted=[],
                    stopped_at="already_within_budget",
                    changed_parts=0,
                    reason="already_within_budget",
                    target_tokens=request.target_tokens,
                    noop=True,
                    deduped=deduped,
                    lifecycle_counts=lifecycle_counts,
                ),
            )

        levels_attempted: list[str] = []
        replacements: list[dict[str, object]] = []
        level_metrics: dict[str, dict[str, int]] = {}
        stopped_at = "not_reached"

        #L2 看到的是经过 L1 处理的视图，L3 看到的是经过 L1、L2 处理的视图
        for level_index, level in enumerate(request.enabled_levels):
            levels_attempted.append(level)
            before_level_tokens = request.estimate_tokens(view)
            level_replacements = self._apply_level(
                view,
                request=request,
                level=level,
                lifecycle_records=lifecycle_records,
            )
            replacements.extend(level_replacements)
            after_level_tokens = request.estimate_tokens(view)
            level_metrics[level] = {
                "before_tokens": before_level_tokens,
                "after_tokens": after_level_tokens,
                "saved_tokens": max(0, before_level_tokens - after_level_tokens),
                "changed_parts": len(level_replacements),
            }
            remaining_levels = request.enabled_levels[level_index + 1 :]
            if (
                after_level_tokens <= request.target_tokens
                and not required_levels.intersection(remaining_levels)
                and not (
                    "l3" in remaining_levels
                    and (
                        _has_l3_mandatory_candidates(
                            _effective_tail_messages(view),
                            lifecycle_records=lifecycle_records,
                            current_turn=request.current_turn,
                            consumed_tool_result_part_ids=request.consumed_tool_result_part_ids,
                        )
                        or _has_l3_per_result_pressure(
                            _effective_tail_messages(view),
                            lifecycle_records=lifecycle_records,
                            current_turn=request.current_turn,
                            per_result_target=per_result_target,
                            consumed_tool_result_part_ids=request.consumed_tool_result_part_ids,
                        )
                    )
                )
            ):
                stopped_at = level
                break

        after_tokens = request.estimate_tokens(view)
        changed_parts = len(replacements)
        noop = changed_parts == 0
        deduped = noop and input_fingerprint in self._seen_noop_fingerprints
        if noop:
            self._seen_noop_fingerprints.add(input_fingerprint)

        return CompactionResult(
            view=view,
            event=CompactionEvent(
                input_fingerprint=input_fingerprint,
                before_tokens=before_tokens,
                after_tokens=after_tokens,
                levels_attempted=levels_attempted,
                stopped_at=stopped_at,
                changed_parts=changed_parts,
                reason=stopped_at,
                target_tokens=request.target_tokens,
                source_part_ids=[str(replacement["source_part_id"]) for replacement in replacements],
                output_part_ids=[str(replacement["replacement_part"]["id"]) for replacement in replacements],
                replacements=replacements,
                noop=noop,
                deduped=deduped,
                lifecycle_counts=lifecycle_counts,
                level_metrics=level_metrics,
                archive_ids=_archive_ids_from_replacements(replacements),
            ),
        )

    def _apply_level(
        self,
        view: SessionView,
        *,
        request: CompactionRequest,
        level: CompactionLevel,
        lifecycle_records: dict[tuple[str, str], ToolResultLifecycleRecord],
    ) -> list[dict[str, object]]:
        if level == "l1":
            return self._apply_l1(
                view,
                active_task_hash=request.active_task_hash,
                current_turn=request.current_turn,
                force_old_task_compaction=request.force_old_task_compaction,
            )
        if level == "l2":
            return self._apply_l2(
                view,
                request=request,
                lifecycle_records=lifecycle_records,
            )
        if level == "l3":
            return self._apply_l3(
                view,
                request=request,
                active_task_hash=request.active_task_hash,
                current_turn=request.current_turn,
                lifecycle_records=lifecycle_records,
            )
        return []

    def _apply_l1(
        self,
        view: SessionView,
        *,
        active_task_hash: str | None,
        current_turn: int,
        force_old_task_compaction: bool,
    ) -> list[dict[str, object]]:
        """
        L1 会遍历当前有效会话中的 user、assistant 消息，找出明确属于旧任务、尚未压缩的普通文本片段；
        如果这些片段距离当前已经至少过去 cold_turn_distance 轮，就将它们的 content 替换成 ""，标记为 trimmed，并记录替换操作。

        Trim only old-task ordinary dialogue that is safe to forget.

        L1 eligibility is partly a property of the enclosing message.  In
        particular, text in an assistant tool-call message must not be trimmed
        independently: it is part of the provider-visible tool transaction.
        """

        changed: list[dict[str, object]] = []
        #只让程序化压缩处理 latest checkpoint 之后的历史记录
        tail_messages = _effective_tail_messages(view)
        latest_user_id = latest_user_message_id(tail_messages)
        for message in tail_messages:
            """
            先保护整条消息
            遇到以下情况直接跳过：
            - 消息角色不是 user 或 assistant。
            - 是最新一条用户消息。
            - 是包含 tool_call 的 assistant 消息。它里面的普通文本也一起保留，以保护完整的工具调用过程。
            """
            if message.role not in {"user", "assistant"}:
                continue
            if message.id == latest_user_id:
                continue
            if message.role == "assistant" and any(part.kind == "tool_call" for part in message.parts):
                continue
            for index, part in enumerate(message.parts):
                #要求这个片段是尚未压缩的普通 text，并且它的 task_hash 与当前任务不同
                if not is_old_task_part(part, active_task_hash=active_task_hash):
                    continue
                #超过 8 轮的消息才压缩
                if not force_old_task_compaction and not _is_cold_old_task_part(
                    part,
                    current_turn=current_turn,
                    cold_turn_distance=self.cold_turn_distance,
                ):
                    continue
                compacted = compact_old_task_part(part)
                if _replace_l1_trimmed(message.parts, index, compacted):
                    changed.append(_replacement_event(message_id=message.id, source=part, replacement=compacted))
        return changed


    """
    L2 会遍历当前有效会话中的工具结果，筛选出模型已经看过、生命周期为 derived、尚未压缩且不处于归档检索保护期的结果，
    通过 RouteCompactRouter 识别内容类型，再交给对应的 xxxRouteCompressor 按规则保留关键结构、删减冗余内容。

    测试、构建日志：BuildOutputRouteCompressor
    优先保留错误、堆栈、错误附近的上下文、警告和执行汇总，删减大量普通日志。
    **例子：**上千行测试输出 → 保留 test_refund FAILED、对应的 AssertionError、文件位置和 1 failed, 99 passed，省略多数正常执行日志。

    搜索结果：SearchResultsRouteCompressor
    按文件分组，保留路径、行号和部分匹配内容；每个文件优先保留首尾匹配，再挑选含 error、TODO 等关键词的匹配。默认最多展示 15 个文件，每个文件 5 条。
    例子：payment.py 有 30 条匹配 → 保留首尾匹配，以及 payment.py:80: # TODO: fix refund 等评分较高的匹配，标明其余条数被省略。

    Git diff：GitDiffRouteCompressor
    保留文件头、@@ 修改位置、增删行，删减未改动的普通上下文。文件数量也有上限。
    **例子：**一个 diff 包含 100 行未改动代码和两行修改 → 保留文件路径、修改位置、-timeout = 10、+timeout = 30，省略大部分未改动代码。

    JSON：JsonRouteCompressor
    对象优先选择 status、error 等重要字段；数组挑选首尾及关键词评分较高的元素，同时截断长字符串、缩减嵌套内容，并记录省略数量。
    **例子：**一个有 80 个字段的响应 → 保留 status: "failed"、error: "timeout" 等字段，将长描述截短，标明省略了哪些字段。数组则保留部分元素，而非完整列表。

    HTML：HtmlRouteCompressor
    提取页面标题、标题层级、可见文本和链接，去掉 HTML 标签及脚本、样式等内容，再限制文本块和链接数量。
    **例子：**一份包含大量 CSS、JavaScript 的接口文档页面 → 提取“支付接口”“请求参数”“POST /pay”和文档链接，省略页面样式和脚本。

    源码形状的内容：SourceCodeRouteCompressor
    保留 import、类和函数签名、类型定义、少量函数体预览，以及 TODO、raise 等重要行，省略大量实现细节。
    **例子：**200 行 Python 内容 → 保留 import requests、class PaymentClient、def refund(...)、函数体前几行和异常抛出行，形成代码结构预览。L2 中 fresh 源码读取受到保护，不会进入这里。

    普通文本：PlainTextRouteCompressor
    直接保留文本开头和结尾，并附上片段 ID、token 数等信息。默认首尾各取 160 个字符。
    **例子：**工具返回一万字符的普通说明 → 保留开头“本次操作包含……”和结尾“处理结果为……”，省略中间内容。
    """
    def _apply_l2(
        self,
        view: SessionView,
        *,
        request: CompactionRequest,
        lifecycle_records: dict[tuple[str, str], ToolResultLifecycleRecord],
    ) -> list[dict[str, object]]:
        changed: list[dict[str, object]] = []
        #负责保存原文
        archive = ToolResultArchive(self.root)
        router = _make_route_router(preview_chars=self.cold_preview_chars)
        for message in _effective_tail_messages(view):
            for index, part in enumerate(message.parts):
                lifecycle = lifecycle_records.get((message.id, part.id))
                """
                它要求：
                - 是模型已经看过的 tool_result。
                - 生命周期是 DERIVED。
                - 尚未压缩，也没有处于归档检索的保护期。
                """
                if not _should_route_compact_l2_part(
                    part,
                    lifecycle=lifecycle,
                    current_turn=request.current_turn,
                    consumed_tool_result_part_ids=request.consumed_tool_result_part_ids,
                ):
                    continue

                compacted = router.compact_part(part)
                if compacted is None:
                    continue
                try:
                    # This is backing, not L3 eviction: archive the exact raw
                    # bytes before the route result becomes visible in the view.
                    # 保存压缩前的内容
                    record = archive.store_original(view.session_id, part)
                except (ArchiveIntegrityError, OSError, ValueError):
                    continue

                assert lifecycle is not None
                compacted.metadata.update(
                    {
                        "archive_id": record.archive_id,
                        "original_content_sha256": record.content_sha256,
                        "original_tokens": record.original_tokens,
                        "replacement_tokens": estimate_text_tokens(compacted.content),
                        "lifecycle": lifecycle.lifecycle.value,
                        "lifecycle_reason": lifecycle.reason,
                        "compaction_state": "l2_route_compacted",
                        "compacted_by": _l2_compacted_by(compacted.metadata.get("compacted_by")),
                    }
                )
                if _replace_if_smaller(message.parts, index, compacted):
                    changed.append(_replacement_event(message_id=message.id, source=part, replacement=compacted))
        return changed

    """
    L3 会从当前有效历史中，筛选模型已经看过、未受保护的工具结果，按“重复 → 被后续读取覆盖 → 已过时 → 单条过大的派生结果 → 其他派生结果”的顺序处理，同一优先级先处理较大的结果。**前三类和单条过大的派生结果，即使总上下文已满足预算，也会继续尝试清理；其他派生结果则在达到预算目标后停止。处理时，先保存完整原文，或复用 L2 已建立的原文归档，再生成包含归档 ID、工具状态、生命周期原因和检索提示的占位符。只有归档成功且占位符严格更小，才替换正文并记录操作；否则保留现有内容。fresh 源码和尚未被模型看过的工具结果受到保护，工具调用与结果的配对也保持完整。
    """
    def _apply_l3(
        self,
        view: SessionView,
        *,
        request: CompactionRequest,
        active_task_hash: str | None,
        current_turn: int,
        lifecycle_records: dict[tuple[str, str], ToolResultLifecycleRecord],
    ) -> list[dict[str, object]]:
        changed: list[dict[str, object]] = []
        archive = ToolResultArchive(self.root)
        del active_task_hash
        #必须是已被模型看过的工具结果，状态为原始内容或 L2 压缩结果，且不处于检索保护期
        candidates = _l3_candidates(
            _effective_tail_messages(view),
            lifecycle_records=lifecycle_records,
            current_turn=current_turn,
            target_tokens=request.target_tokens,
            per_result_target=_per_result_target(
                request.l2_result_target_tokens,
                fallback=self.large_tool_result_tokens,
            ),
            consumed_tool_result_part_ids=request.consumed_tool_result_part_ids,
        )
        for candidate in candidates:
            if not candidate.mandatory and not candidate.over_per_result_target and request.estimate_tokens(view) <= request.target_tokens:
                break

            part = candidate.message.parts[candidate.part_index]
            # A preceding candidate may have transformed this part in future
            # refactors.  Re-check before touching durable backing.
            if not _can_archive_l3_part(
                part,
                lifecycle=candidate.lifecycle,
                current_turn=current_turn,
                consumed_tool_result_part_ids=request.consumed_tool_result_part_ids,
            ):
                continue
            try:
                #已经经过 L2 的结果，会复用 L2 保存的原文归档；尚未归档的结果，则调用 store_original() 保存当前原文
                record = _l3_backing_record(archive, view.session_id, part)
                compacted = archive.make_placeholder(
                    part,
                    record,
                    lifecycle=candidate.lifecycle.lifecycle.value,
                    summary=_lifecycle_summary(part, candidate.lifecycle),
                    key_errors=_lifecycle_key_errors(part),
                )
            except (ArchiveIntegrityError, OSError, ValueError):
                # Archive backing is an all-or-nothing safety boundary: if
                # persistence or validation fails, retain the current part.
                continue
            compacted.metadata.update(
                {
                    "lifecycle": candidate.lifecycle.lifecycle.value,
                    "lifecycle_reason": candidate.lifecycle.reason,
                    "replacement_tokens": estimate_text_tokens(compacted.content),
                }
            )
            if _replace_if_smaller(candidate.message.parts, candidate.part_index, compacted):
                changed.append(
                    _replacement_event(
                        message_id=candidate.message.id,
                        source=part,
                        replacement=compacted,
                    )
                )
        return changed


def _clone_view(view: SessionView) -> SessionView:
    return SessionView(
        session_id=view.session_id,
        messages=[AgentMessage.from_dict(message.to_dict()) for message in view.messages],
        checkpoints=list(view.checkpoints),
        metadata=dict(view.metadata),
    )


def _effective_tail_messages(view: SessionView) -> list[AgentMessage]:
    """只让程序化压缩处理 latest checkpoint 之后的真实 tail。

    checkpoint 覆盖过的旧历史已经由 summary 表达；L1-L3 如果继续扫描旧 raw message，
    会和 ContextBuilder/L4 的 effective context 边界不一致。
    """

    checkpoint = CheckpointIndex(view.checkpoints).latest()
    if checkpoint is None:
        return view.messages

    for index, message in enumerate(view.messages):
        if message.id == checkpoint.tail_start_message_id:
            return view.messages[index:]
    raise ValueError(f"latest checkpoint tail_start_message_id not found: {checkpoint.tail_start_message_id}")


def _replacement_event(*, message_id: str, source: MessagePart, replacement: MessagePart) -> dict[str, object]:
    return {
        "message_id": message_id,
        "source_part_id": source.id,
        "replacement_part": replacement.to_dict(),
    }


def _replace_l1_trimmed(parts: list[MessagePart], index: int, trimmed: MessagePart) -> bool:
    """Apply L1 even when the resulting empty content has zero tokens."""

    if parts[index].content == trimmed.content and parts[index].metadata == trimmed.metadata:
        return False
    parts[index] = trimmed
    return True


def _replace_if_smaller(parts: list[MessagePart], index: int, compacted: MessagePart) -> bool:
    original = parts[index]
    if estimate_text_tokens(compacted.content) >= estimate_text_tokens(original.content):
        return False
    parts[index] = compacted
    return True


def _is_cold_old_task_part(
    part: MessagePart,
    *,
    current_turn: int,
    cold_turn_distance: int,
) -> bool:
    created_turn = part.metadata.get("created_turn")
    return isinstance(created_turn, int) and not isinstance(created_turn, bool) and current_turn - created_turn >= cold_turn_distance


def _archive_ids_from_replacements(replacements: list[dict[str, object]]) -> list[str]:
    archive_ids: list[str] = []
    for replacement in replacements:
        replacement_part = replacement.get("replacement_part")
        if not isinstance(replacement_part, dict):
            continue
        metadata = replacement_part.get("metadata")
        if not isinstance(metadata, dict):
            continue
        archive_id = metadata.get("archive_id")
        if isinstance(archive_id, str) and archive_id and archive_id not in archive_ids:
            archive_ids.append(archive_id)
    return archive_ids


def _lifecycle_counts(
    lifecycle_records: dict[tuple[str, str], ToolResultLifecycleRecord],
) -> dict[str, int]:
    counts = {lifecycle.value: 0 for lifecycle in ToolResultLifecycle}
    for record in lifecycle_records.values():
        counts[record.lifecycle.value] += 1
    return counts


def _make_route_router(*, preview_chars: int) -> RouteCompactRouter:
    json_compressor = JsonRouteCompressor()
    return RouteCompactRouter(
        compressors={
            RouteContentType.BUILD_OUTPUT: BuildOutputRouteCompressor(),
            RouteContentType.GIT_DIFF: GitDiffRouteCompressor(),
            RouteContentType.HTML: HtmlRouteCompressor(),
            RouteContentType.JSON_ARRAY: json_compressor,
            RouteContentType.JSON_OBJECT: json_compressor,
            RouteContentType.SEARCH_RESULTS: SearchResultsRouteCompressor(),
            RouteContentType.SOURCE_CODE: SourceCodeRouteCompressor(),
            RouteContentType.PLAIN_TEXT: PlainTextRouteCompressor(),
        },
        preview_chars=preview_chars,
    )


def _l2_compacted_by(value: object) -> str:
    """Translate the existing route labels at the L2 ownership boundary.

    Content compressors deliberately remain independently usable.  The
    pipeline is where their output acquires its L2 semantic label.
    """

    label = str(value or "l2_route")
    if label.startswith("l3_"):
        return f"l2_{label[3:]}"
    return label


def _should_route_compact_l2_part(
    part: MessagePart,
    *,
    lifecycle: ToolResultLifecycleRecord | None,
    current_turn: int,
    consumed_tool_result_part_ids: frozenset[str],
) -> bool:
    if not _is_consumed_tool_result(
        part,
        consumed_tool_result_part_ids=consumed_tool_result_part_ids,
    ):
        return False
    if lifecycle is None or is_already_compacted(part):
        return False
    if _is_retrieval_protected(part, current_turn=current_turn):
        return False
    return lifecycle.lifecycle is ToolResultLifecycle.DERIVED


def _is_retrieval_protected(part: MessagePart, *, current_turn: int) -> bool:
    metadata = part.metadata
    retrieval_metadata: dict[str, object] = metadata
    if metadata.get("archive_retrieval") is not True:
        nested_data = metadata.get("data")
        if not isinstance(nested_data, dict) or nested_data.get("archive_retrieval") is not True:
            return False
        retrieval_metadata = nested_data
    protected_until_turn = retrieval_metadata.get("compaction_protected_until_turn")
    return isinstance(protected_until_turn, int) and not isinstance(protected_until_turn, bool) and protected_until_turn >= current_turn


def _lifecycle_summary(part: MessagePart, lifecycle: ToolResultLifecycleRecord) -> str:
    tool_name = str(part.metadata.get("tool_name") or "tool").replace("\n", " ").strip() or "tool"
    return f"{tool_name} result is {lifecycle.lifecycle.value}: {lifecycle.reason}."


def _lifecycle_key_errors(part: MessagePart) -> tuple[str, ...]:
    error = part.metadata.get("error")
    return (error,) if isinstance(error, str) and error.strip() else ()


@dataclass(frozen=True, slots=True)
class _L3Candidate:
    message: AgentMessage
    part_index: int
    lifecycle: ToolResultLifecycleRecord
    mandatory: bool
    over_per_result_target: bool
    priority: int
    tokens: int
    created_turn: int
    tail_index: int


def _has_l3_mandatory_candidates(
    messages: list[AgentMessage],
    *,
    lifecycle_records: dict[tuple[str, str], ToolResultLifecycleRecord],
    current_turn: int,
    consumed_tool_result_part_ids: frozenset[str],
) -> bool:
    for message in messages:
        for part in message.parts:
            lifecycle = lifecycle_records.get((message.id, part.id))
            if _is_l3_mandatory(lifecycle) and _can_archive_l3_part(
                part,
                lifecycle=lifecycle,
                current_turn=current_turn,
                consumed_tool_result_part_ids=consumed_tool_result_part_ids,
            ):
                return True
    return False


def _has_l3_per_result_pressure(
    messages: list[AgentMessage],
    *,
    lifecycle_records: dict[tuple[str, str], ToolResultLifecycleRecord],
    current_turn: int,
    per_result_target: int | None,
    consumed_tool_result_part_ids: frozenset[str],
) -> bool:
    """Whether an eligible derived result needs an L2/L3 pass below budget."""

    if per_result_target is None:
        return False
    for message in messages:
        for part in message.parts:
            lifecycle = lifecycle_records.get((message.id, part.id))
            if (
                lifecycle is not None
                and lifecycle.lifecycle is ToolResultLifecycle.DERIVED
                and _can_archive_l3_part(
                    part,
                    lifecycle=lifecycle,
                    current_turn=current_turn,
                    consumed_tool_result_part_ids=consumed_tool_result_part_ids,
                )
                and estimate_text_tokens(part.content) > per_result_target
            ):
                return True
    return False


def _per_result_target(value: object, *, fallback: int) -> int | None:
    """Resolve a positive per-result budget without treating bool as int."""

    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    if isinstance(fallback, int) and not isinstance(fallback, bool) and fallback > 0:
        return fallback
    return None


def _l3_candidates(
    messages: list[AgentMessage],
    *,
    lifecycle_records: dict[tuple[str, str], ToolResultLifecycleRecord],
    current_turn: int,
    target_tokens: int,
    per_result_target: int | None,
    consumed_tool_result_part_ids: frozenset[str],
) -> list[_L3Candidate]:
    """Return deterministic tool-result-only L3 candidates.

    Mandatory lifecycle cleanup is selected regardless of the overall target.
    Derived output is optional: it is selected when an individual result still
    exceeds its L2 budget or when the current context remains above target.
    """

    del target_tokens  # Selection below-budget is decided during application.
    candidates: list[_L3Candidate] = []
    tail_index = 0
    for message in messages:
        for part_index, part in enumerate(message.parts):
            lifecycle = lifecycle_records.get((message.id, part.id))
            if lifecycle is None or not _can_archive_l3_part(
                part,
                lifecycle=lifecycle,
                current_turn=current_turn,
                consumed_tool_result_part_ids=consumed_tool_result_part_ids,
            ):
                tail_index += 1
                continue

            tokens = estimate_text_tokens(part.content)
            mandatory = _is_l3_mandatory(lifecycle)
            over_per_result_target = per_result_target is not None and lifecycle.lifecycle is ToolResultLifecycle.DERIVED and tokens > per_result_target
            if mandatory:
                priority = _l3_priority(lifecycle.lifecycle)
            elif over_per_result_target:
                priority = _l3_priority(lifecycle.lifecycle, over_per_result_target=True)
            elif lifecycle.lifecycle is ToolResultLifecycle.DERIVED:
                priority = _l3_priority(lifecycle.lifecycle)
            else:
                tail_index += 1
                continue

            created_turn = part.metadata.get("created_turn")
            candidates.append(
                _L3Candidate(
                    message=message,
                    part_index=part_index,
                    lifecycle=lifecycle,
                    mandatory=mandatory,
                    over_per_result_target=over_per_result_target,
                    priority=priority,
                    tokens=tokens,
                    created_turn=created_turn if isinstance(created_turn, int) and not isinstance(created_turn, bool) else 0,
                    tail_index=tail_index,
                )
            )
            tail_index += 1

    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.priority,
            -candidate.tokens,
            candidate.created_turn,
            candidate.tail_index,
        ),
    )


def _can_archive_l3_part(
    part: MessagePart,
    *,
    lifecycle: ToolResultLifecycleRecord | None,
    current_turn: int,
    consumed_tool_result_part_ids: frozenset[str],
) -> bool:
    if not _is_consumed_tool_result(
        part,
        consumed_tool_result_part_ids=consumed_tool_result_part_ids,
    ):
        return False
    if lifecycle is None:
        return False
    # L3 may turn a raw result or its L2 projection into a placeholder.  It
    # must not consume a pinned/retrieved result or replay a legacy/terminal
    # compaction projection whose backing is not this L2 flow's raw record.
    state = str(part.metadata.get("compaction_state") or "raw")
    if state not in {"raw", "l2_route_compacted"}:
        return False
    if _is_retrieval_protected(part, current_turn=current_turn):
        return False
    return lifecycle.lifecycle in {
        ToolResultLifecycle.STALE,
        ToolResultLifecycle.SUPERSEDED,
        ToolResultLifecycle.DUPLICATE,
        ToolResultLifecycle.DERIVED,
    }


def _is_consumed_tool_result(
    part: MessagePart,
    *,
    consumed_tool_result_part_ids: frozenset[str],
) -> bool:
    return part.kind == "tool_result" and part.id in consumed_tool_result_part_ids


def _is_l3_mandatory(lifecycle: ToolResultLifecycleRecord | None) -> bool:
    return lifecycle is not None and lifecycle.lifecycle in {
        ToolResultLifecycle.DUPLICATE,
        ToolResultLifecycle.SUPERSEDED,
        ToolResultLifecycle.STALE,
    }


def _l3_priority(lifecycle: ToolResultLifecycle, *, over_per_result_target: bool = False) -> int:
    if lifecycle is ToolResultLifecycle.DUPLICATE:
        return 0
    if lifecycle is ToolResultLifecycle.SUPERSEDED:
        return 1
    if lifecycle is ToolResultLifecycle.STALE:
        return 2
    if over_per_result_target:
        return 3
    return 4


def _l3_backing_record(
    archive: ToolResultArchive,
    session_id: str,
    part: MessagePart,
):
    """Return raw backing for a candidate without archiving L2 text as raw.

    L2 retains its original archive id and payload.  A later L3 projection
    must use exactly that backing so `retrieve_archive` always returns the
    pre-route result rather than a compact derivative.
    """

    archive_id = part.metadata.get("archive_id")
    if isinstance(archive_id, str) and archive_id:
        record, _raw = archive.read(session_id, archive_id)
        return record
    return archive.store_original(session_id, part)
