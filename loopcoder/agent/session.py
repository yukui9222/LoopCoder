"""Agent 会话运行时。

- 更新会话内容：调用 append_*() 写入 JSONL。
- 读取更新后的最新内容：调用 rebuild_view()，重放日志，得到新的 view。
已有的 view 不会自动更新；如果日志没有变化，就可以继续使用已有 view。

AgentSession
    ├─ 用 writer 把消息写入 JSONL
    └─ 用 rebuild_view() 读取 JSONL → 返回一个 SessionView

AgentSession
    ↓ rebuild_view()
SessionView（整理好的会话数据）
    ↓ ContextBuilder
发给模型的消息

程序重新启动，要恢复昨天的会话：用 resume()，复用writer、工具注册表，只是重建 SessionView。
会话正在运行，只想获取最新内容：用 rebuild_view()，包含AgentSession、writer、工具注册表。

append_tool_result、append_assistant_response、append_user_message这几个函数说白了就是把消息写入 JSONL 日志，然后返回一个 MessageID

这一层连接 context store、runtime state、system prompt cache 和 session scoped tools。
它不负责调用模型，也不执行压缩；这些动作由更外层的 agent loop 或 context manager 编排。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock

from loopcoder.agent.prompt_inputs import (
    DEFAULT_PERMISSION_POLICY,
    build_system_prompt_inputs,
    read_agents_md,
)
from loopcoder.agent.tool_flow import assistant_response_to_parts, tool_result_to_part
from loopcoder.context.identity import new_message_id
from loopcoder.context.runtime_replay import replay_runtime_state
from loopcoder.context.runtime_state import SessionRuntimeState
from loopcoder.context.store import JsonlSessionStore
from loopcoder.context.system_prompt import PromptPrefixCache, SystemPromptBuilder
from loopcoder.context.task_boundary import observation_from_tool_result_data
from loopcoder.context.writer import SessionEventWriter
from loopcoder.permissions.grants import FilePermissionGrantStore, PermissionGrantStore
from loopcoder.permissions.manager import PermissionManager
from loopcoder.permissions.policy import DefaultPermissionPolicy
from loopcoder.permissions.types import PermissionDecisionKind, PermissionMode
from loopcoder.providers.types import ChatResponse, ProviderCapabilities, ToolCall
from loopcoder.runtime.user_input import UserInputRequest
from loopcoder.permissions.types import PermissionDecision, PermissionRequest
from loopcoder.tools.permission_registry import PermissionAwareToolRegistry
from loopcoder.tools.review import PrewriteReview, build_prewrite_review, supports_prewrite_review
from loopcoder.tools.session_registry import ToolRegistryLike, create_session_tool_registry
from loopcoder.tools.types import Tool, ToolResult, make_error_result
from loopcoder.context.models import AgentMessage, MessagePart, SessionView
from loopcoder.input.attachments import UserAttachment, prepare_attachments_for_session
from loopcoder.utils.sandbox_access import SandboxAccess, SandboxAccessMode
from loopcoder.skills.discovery import discover_all_skills
from loopcoder.skills.catalog import render_skill_catalog
from loopcoder.skills.models import SkillCatalog

DEFAULT_BASE_RULES = "你是 LoopCoder，一个本地 AI coding agent。请遵守项目规则并优先保持上下文可恢复。"


@dataclass(slots=True)
class PendingPermissionExecution:
    """等待用户确认后才能继续的工具调用。

    这类状态不能相信 UI 回传的 payload。agent 只接受 request id 和用户选择，
    原始 tool_call 与规范化后的 permission request 都保存在本地运行时对象里。
    """

    request_id: str
    tool_call: ToolCall
    permission_request: PermissionRequest
    prewrite_review: PrewriteReview | None = None
    review_only: bool = False
    skipped_tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(slots=True)
class ToolPermissionPreflight:
    """工具权限预检结果。"""

    request: PermissionRequest
    decision: PermissionDecision


@dataclass(slots=True)
class AgentSession:
    """单个会话的运行时容器。

    `SessionRuntimeState` 和 `PromptPrefixCache` 都是运行期对象，不写成自然语言消息。
    真正可 resume 的会话事实通过 `JsonlSessionStore` 追加事件保存。
    """

    session_id: str
    store: JsonlSessionStore
    runtime_state: SessionRuntimeState
    tool_registry: ToolRegistryLike
    writer: SessionEventWriter
    agents_md: str = ""
    skill_catalog: SkillCatalog = field(default_factory=SkillCatalog)
    base_rules: str = DEFAULT_BASE_RULES
    prompt_cache: PromptPrefixCache = field(default_factory=PromptPrefixCache)
    prompt_builder: SystemPromptBuilder = field(default_factory=SystemPromptBuilder)
    provider_capability_overrides: dict[str, object] = field(default_factory=dict)
    permission_manager: PermissionManager | None = None
    permission_policy: dict[str, object] = field(default_factory=lambda: dict(DEFAULT_PERMISSION_POLICY))
    sandbox_access: SandboxAccess = field(default_factory=SandboxAccess)
    known_message_ids: set[str] = field(default_factory=set)
    turn_counter: int = 0
    mode: str = "default"
    benchmark_task: str = ""
    require_prewrite_review: bool = True
    pending_permission_execution: PendingPermissionExecution | None = None
    _tool_result_lock: RLock = field(default_factory=RLock, repr=False)
    _tool_result_message_ids: dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def create(
        cls,
        *,
        store: JsonlSessionStore,
        session_id: str,
        agents_md: str = "",
        skill_catalog: SkillCatalog | None = None,
        tools: list[Tool] | None = None,
        permission_manager: PermissionManager | None = None,
        sandbox_access: SandboxAccess | None = None,
    ) -> "AgentSession":
        """
        把会话所需的组件组装好，返回一个可以使用的 AgentSession
        创建全新 session，并初始化 session-scoped 工具。

        这里会立即写入 `session_created` 事件。后续所有可恢复事实都追加到同一个
        JSONL 日志中；运行时对象只是方便当前进程快速访问这些事实。
        """

        runtime_state = SessionRuntimeState(session_id=session_id)
        known_message_ids: set[str] = set()
        writer = SessionEventWriter(store=store, session_id=session_id)
        """
        这个函数就是把外部传入的工具和自动加入的工具
        （如 task_create/update/revise/list）装进一个注册表，
        让 Agent 按工具名找到并执行它们
        """
        registry = create_session_tool_registry(
            session_id=session_id,
            runtime_state=runtime_state,
            tools=tools,
            known_message_ids=known_message_ids,
            task_boundary_required_stable_count=_task_boundary_required_stable_count(permission_manager),
            permission_manager=permission_manager,
            archive_root=store.root,
            current_turn=lambda: writer.current_turn,
            store=store,
            writer=writer,
            skill_catalog=(skill_catalog or SkillCatalog()).resolved(),
        )
        """
        创建 AgentSession 对象，把会话状态、工具注册表、日志写入器等组件放进去。
        """
        session = cls(
            session_id=session_id,
            store=store,
            runtime_state=runtime_state,
            tool_registry=registry,
            writer=writer,
            agents_md=agents_md,
            skill_catalog=(skill_catalog or SkillCatalog()).resolved(),
            known_message_ids=known_message_ids,
            permission_manager=permission_manager,
            sandbox_access=sandbox_access or SandboxAccess(),
            turn_counter=0,
            mode=permission_manager.mode.value if permission_manager is not None else "default",
        )
        session._sync_sandbox_access_with_mode()
        session.append_session_created()
        return session

    @classmethod
    def from_project(
        cls,
        *,
        store: JsonlSessionStore,
        session_id: str,
        project_root: str | Path,
        tools: list[Tool] | None = None,
        permission_manager: PermissionManager | None = None,
        sandbox_access: SandboxAccess | None = None,
    ) -> "AgentSession":
        """
        根据项目目录准备好 AGENTS.md、skill 目录和权限管理器，再调用 create() 创建一个新的 AgentSession

        从项目根目录创建 session。

        这一层负责读取项目级 `AGENTS.md`，并创建默认权限管理器。这样 app/UI 不需要知道
        AGENTS.md、permission grant 文件放在哪里，也不会把这些初始化细节散落到 widget。
        """

        agents_md = read_agents_md(project_root)
        skill_catalog = discover_all_skills(project_root)
        permission_manager = permission_manager or create_project_permission_manager(
            project_root,
            grants=FilePermissionGrantStore(store.root / "permissions.json"),
        )
        return cls.create(
            store=store,
            session_id=session_id,
            agents_md=agents_md,
            skill_catalog=skill_catalog,
            tools=tools,
            permission_manager=permission_manager,
            sandbox_access=sandbox_access,
        )

    @classmethod
    def resume(
        cls,
        *,
        store: JsonlSessionStore,
        session_id: str,
        agents_md: str = "",
        skill_catalog: SkillCatalog | None = None,
        tools: list[Tool] | None = None,
        permission_manager: PermissionManager | None = None,
        sandbox_access: SandboxAccess | None = None,
    ) -> "AgentSession":
        """
        读取旧会话的 JSONL，恢复已保存的状态，再把 writer、工具注册表等组件装好，返回一个能继续使用的 AgentSession
        从 JSONL 会话日志恢复运行期 session。

        `rebuild_session_view()` 恢复可投影的消息和 checkpoint；`replay_runtime_state()`
        恢复 task hash、compact 熔断和最近压缩事实。这里还要把历史 message id 注入
        `known_message_ids`，否则恢复后的 task_boundary 工具会拒绝模型引用旧消息。
        """

        runtime_state = replay_runtime_state(store, session_id)
        view = store.rebuild_session_view(session_id)
        known_message_ids = {message.id for message in view.messages}
        turn_counter = _infer_turn_counter(view.messages)
        writer = SessionEventWriter(store=store, session_id=session_id, current_turn=turn_counter)
        registry = create_session_tool_registry(
            session_id=session_id,
            runtime_state=runtime_state,
            tools=tools,
            known_message_ids=known_message_ids,
            task_boundary_required_stable_count=_task_boundary_required_stable_count(permission_manager),
            permission_manager=permission_manager,
            archive_root=store.root,
            current_turn=lambda: writer.current_turn,
            store=store,
            writer=writer,
            skill_catalog=(skill_catalog or SkillCatalog()).resolved(),
        )
        session = cls(
            session_id=session_id,
            store=store,
            runtime_state=runtime_state,
            tool_registry=registry,
            writer=writer,
            agents_md=agents_md,
            skill_catalog=(skill_catalog or SkillCatalog()).resolved(),
            known_message_ids=known_message_ids,
            permission_manager=permission_manager,
            sandbox_access=sandbox_access or SandboxAccess(),
            turn_counter=turn_counter,
            mode=permission_manager.mode.value if permission_manager is not None else "default",
            _tool_result_message_ids=_tool_result_message_ids_from_view(view),
        )
        session._sync_sandbox_access_with_mode()
        return session

    def restore_pending_permission_execution(self) -> PendingPermissionExecution | None:
        """从 append-only 历史中重建未完成的权限确认。

        只有最后一个 assistant tool_call 批次仍缺少 tool_result 时才尝试重建。
        即使 grant 已经存在，也只恢复 pending，不自动执行工具，避免 resume 阶段
        产生隐式副作用或留下悬空 tool_call。
        """

        pending = self._pending_tool_calls_from_tail()
        if len(pending) != 1:
            return None

        tool_call, skipped_tool_calls, persisted_review_only = pending[0]
        preflight = self.preflight_tool_call_permission(tool_call)
        if preflight is None:
            return None

        restored = PendingPermissionExecution(
            request_id=preflight.request.id,
            tool_call=tool_call,
            permission_request=preflight.request,
            prewrite_review=(
                build_prewrite_review(
                    self.permission_manager.policy.project_root,
                    tool_call,
                    access=self.sandbox_access,
                )
                if self.permission_manager is not None and supports_prewrite_review(tool_call.name)
                else None
            ),
            review_only=(persisted_review_only if persisted_review_only is not None else preflight.decision.kind == PermissionDecisionKind.ALLOW),
            skipped_tool_calls=skipped_tool_calls,
        )
        self.pending_permission_execution = restored
        return restored

    def persist_pending_permission_kind(self, *, tool_call_id: str, review_only: bool) -> None:
        view = self.rebuild_view()
        for message in reversed(view.messages):
            if message.role != "assistant":
                continue
            part = next(
                (item for item in message.parts if item.kind == "tool_call" and str(item.metadata.get("tool_call_id") or "") == tool_call_id),
                None,
            )
            if part is not None:
                self.writer.append_message_part_metadata_updated(
                    message_id=message.id,
                    part_id=part.id,
                    metadata={"prewrite_review_only": review_only},
                )
            return

    def pending_permission_input_request(
        self,
        pending: PendingPermissionExecution | None = None,
    ) -> UserInputRequest | None:
        pending = pending or self.pending_permission_execution
        if pending is None or self.permission_manager is None:
            return None
        confirmation = (
            self.permission_manager.build_prewrite_review_confirmation(pending.permission_request) if pending.review_only else self.permission_manager.build_confirmation(pending.permission_request)
        )
        if pending.prewrite_review is not None:
            confirmation.payload["prewrite_review"] = pending.prewrite_review.to_payload()
        confirmation.payload["pending_tool_call"] = {
            "id": pending.tool_call.id,
            "name": pending.tool_call.name,
            "arguments": deepcopy(pending.tool_call.arguments),
        }
        return confirmation

    def append_session_created(self) -> None:
        self.writer.append_session_created()

    def build_system_prefix(
        self,
        *,
        provider_name: str,
        provider_model: str = "",
        provider_capabilities: ProviderCapabilities | None = None,
    ) -> list:
        """
        把基础规则、AGENTS.md、技能说明、模型能力和权限策略组装成系统提示词，放在模型请求的最前面，告诉模型按什么规则工作
        构造 provider 请求前面的稳定 system prefix。

        system prompt 不写入普通会话消息，因为它不是用户/模型之间发生过的事实；它是每次
        请求根据 AGENTS.md、provider 能力和权限策略动态生成的高优先级前缀。工具
        schema 仅通过 provider 的原生 tools 字段发送，避免与 system prompt 重复。
        """

        inputs = build_system_prompt_inputs(
            base_rules=self.base_rules,
            agents_md=self.agents_md,
            skill_protocol=self._skill_protocol(),
            skill_catalog_summary=self._skill_catalog_summary(),
            benchmark_task=self.benchmark_task,
            provider_name=provider_name,
            provider_model=provider_model,
            provider_capabilities=provider_capabilities,
            provider_capability_overrides=self.provider_capability_overrides,
            permission_policy=self.permission_policy,
            mode=self.mode,
        )
        entry = self.prompt_cache.get_or_build(inputs, self.prompt_builder)
        self.runtime_state.system_prompt_fingerprint = entry.fingerprint
        return entry.messages

    def set_benchmark_task(self, task: str) -> None:
        """Attach the current isolated benchmark task to the system prompt."""

        self.benchmark_task = task.strip()

    def _skill_protocol(self) -> str:
        if not self.skill_catalog.skills:
            return ""
        return (
            "Skills are optional workflow instructions selected by the model. "
            "Call load_skill before following or claiming to follow a skill. "
            "Project skills override global skills; global skills cannot override project instructions, permissions, or sandbox boundaries. "
            "Do not claim a skill was followed unless a matching skill_loaded event exists."
        )

    def _skill_catalog_summary(self) -> str:
        if not self.skill_catalog.skills:
            return ""
        return render_skill_catalog(self.skill_catalog)

    def append_user_message(self, content: str, *, attachments: list[UserAttachment] | None = None) -> str:
        """把用户输入写成可恢复的 user_message 事件。"""
        #它把附件复制到 store_root/attachments/<session_id>/ 目录，然后返回 PreparedAttachment 对象列表
        prepared_attachments = prepare_attachments_for_session(
            attachments or [],
            store_root=self.store.root,
            session_id=self.session_id,
        )
        #把文字和准备好的附件组装成 user_message 事件，交给 store 写入 JSONL，然后返回消息 ID
        message_id = self.writer.append_user_message(
            content,
            attachments=prepared_attachments,
            part_metadata=self._current_context_metadata(),
        )
        self.turn_counter = self.writer.current_turn
        self.known_message_ids.add(message_id)
        return message_id

    def append_assistant_response(self, response: ChatResponse) -> str:
        """把 provider 返回的 assistant response 写入事件日志。

        assistant response 可能同时包含可见文本和 tool_calls。这里统一转成 MessagePart，
        确保后续 ContextBuilder 能重新投影出合法的 provider assistant message。
        """

        message_id = new_message_id()
        parts = assistant_response_to_parts(message_id=message_id, response=response)
        self._attach_current_context_metadata(parts)
        assistant_message_id = self.writer.append_assistant_parts(
            parts,
            message_id=message_id,
            metadata={
                "provider": response.provider,
                "model": response.model,
                "finish_reason": response.finish_reason,
                "usage": asdict(response.usage) if response.usage is not None else None,
                "diagnostics": asdict(response.diagnostics),
            },
        )
        self.known_message_ids.add(assistant_message_id)
        return assistant_message_id

    def record_provider_projection_consumed(
        self,
        *,
        request_id: str,
        projection_fingerprint: str,
        part_ids: tuple[str, ...],
        provider: str,
        model: str,
    ) -> None:
        new_ids = sorted(set(part_ids) - self.runtime_state.consumed_tool_result_part_ids)
        if not new_ids:
            return
        self.writer.append_provider_projection_consumed(
            request_id=request_id,
            projection_fingerprint=projection_fingerprint,
            part_ids=new_ids,
            provider=provider,
            model=model,
        )
        self.runtime_state.consumed_tool_result_part_ids.update(new_ids)

    def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """通过当前 session 的工具注册表执行一次模型请求的工具调用。"""

        return self.tool_registry.execute(tool_call.name, tool_call.arguments)

    def preflight_tool_call_permission(self, tool_call: ToolCall) -> ToolPermissionPreflight | None:
        """对工具调用做权限预检，但不执行工具。

        只有权限 wrapper 支持这个能力；无权限声明的工具返回 `None`，由旧执行路径
        直接处理。这样权限系统接入不会污染普通工具的执行模型。
        """

        registry = self.tool_registry
        if not isinstance(registry, PermissionAwareToolRegistry):
            return None
        preflight = registry.preflight(tool_call.name, tool_call.arguments)
        if preflight is None:
            return None
        _, _, request, decision = preflight
        return ToolPermissionPreflight(request=request, decision=decision)

    def execute_tool_call_after_permission_confirmation(self, tool_call: ToolCall) -> ToolResult:
        """执行已经通过用户确认的 pending tool_call。"""

        if isinstance(self.tool_registry, PermissionAwareToolRegistry):
            return self.tool_registry.execute_without_permission_check(
                tool_call.name,
                tool_call.arguments,
            )
        return self.tool_registry.execute(tool_call.name, tool_call.arguments)

    def set_permission_mode(self, mode: PermissionMode | str) -> PermissionMode:
        """切换当前 session 的权限策略模式。"""

        resolved = PermissionMode(str(mode))
        self.mode = resolved.value
        if self.permission_manager is not None:
            self.permission_manager.mode = resolved
        self._sync_sandbox_access_with_mode()
        return resolved

    def _sync_sandbox_access_with_mode(self) -> None:
        if self.mode == PermissionMode.BYPASS.value:
            self.sandbox_access.mode = SandboxAccessMode.UNRESTRICTED
            self.permission_policy["path_access"] = "unrestricted"
            self.permission_policy["read"] = "allow"
            self.permission_policy["write"] = "allow"
            self.permission_policy["delete"] = "allow"
            self.permission_policy["shell"] = "allow"
            self.permission_policy["network"] = "allow"
            return

        self.sandbox_access.mode = SandboxAccessMode.PROJECT
        self.permission_policy["path_access"] = DEFAULT_PERMISSION_POLICY["path_access"]
        self.permission_policy["read"] = DEFAULT_PERMISSION_POLICY["read"]
        self.permission_policy["write"] = DEFAULT_PERMISSION_POLICY["write"]
        self.permission_policy["delete"] = DEFAULT_PERMISSION_POLICY["delete"]
        self.permission_policy["shell"] = DEFAULT_PERMISSION_POLICY["shell"]
        self.permission_policy["network"] = DEFAULT_PERMISSION_POLICY["network"]

    def append_tool_result(self, *, tool_call: ToolCall, result: ToolResult) -> str:
        """
        把工具执行结果保存成一条 tool 消息，写入 JSONL，并返回这条消息的 ID

        把工具执行结果写成 role=tool 事实。

        这一步是 tool calling 闭环的关键：模型下一次调用时会看到这个 tool_result，并基于
        工具输出生成后续回答。工具结果不直接替代 assistant 回复。
        """

        with self._tool_result_lock:
            existing_message_id = self._tool_result_message_ids.get(tool_call.id)
            if existing_message_id is not None:
                return existing_message_id

            message_id = new_message_id()
            part = tool_result_to_part(message_id=message_id, tool_call=tool_call, result=result)
            self._attach_current_context_metadata([part])
            tool_message_id = self.writer.append_tool_result_part(
                part,
                message_id=message_id,
            )
            self._tool_result_message_ids[tool_call.id] = tool_message_id
            self.known_message_ids.add(tool_message_id)
            self._append_task_boundary_observation_if_present(tool_call=tool_call, result=result)
            return tool_message_id

    def append_interrupted_tool_results(self) -> list[ToolCall]:
        """为会话尾部尚未闭合的工具调用写入中断结果。"""

        with self._tool_result_lock:
            pending = self._pending_tool_calls_from_tail()
            if len(pending) != 1:
                return []

            first, remaining, _ = pending[0]
            tool_calls = [first, *remaining]
            for tool_call in tool_calls:
                self.append_tool_result(
                    tool_call=tool_call,
                    result=make_error_result(
                        tool_call.name,
                        "工具执行被用户中断；结果未知，操作可能尚未执行、部分执行，或已在后台继续。",
                        interrupted=True,
                        execution_outcome="unknown",
                    ),
                )
            return tool_calls

    def append_background_notification(
        self,
        *,
        content: str,
        job_id: str,
        tool_name: str,
        status: str,
        task_id: str | None = None,
        observed_revision: int | None = None,
    ) -> str:
        """把一条后台完成通知写成可 resume 的独立事件。"""

        message_id = self.writer.append_background_notification(
            content=content,
            job_id=job_id,
            tool_name=tool_name,
            status=status,
            task_id=task_id,
            observed_revision=observed_revision,
        )
        self.known_message_ids.add(message_id)
        return message_id

    @property
    def current_turn(self) -> int:
        return self.writer.current_turn

    def rebuild_view(self):
        """从 append-only JSONL 重建当前 SessionView。"""

        return self.store.rebuild_session_view(self.session_id)

    def _append_task_boundary_observation_if_present(self, *, tool_call: ToolCall, result: ToolResult) -> None:
        if tool_call.name != "task_boundary" or not result.ok:
            return
        observation = observation_from_tool_result_data(result.data)
        if observation is not None:
            self.writer.append_task_boundary_observation(observation)

    def _current_context_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {}
        if self.runtime_state.active_task_hash:
            metadata["task_hash"] = self.runtime_state.active_task_hash
        return metadata

    def _attach_current_context_metadata(self, parts: list[MessagePart]) -> None:
        """给本轮新写入的 parts 附加当前任务上下文元数据。"""

        metadata = self._current_context_metadata()
        for part in parts:
            part.metadata.update(metadata)

    def _pending_tool_calls_from_tail(self) -> list[tuple[ToolCall, list[ToolCall], bool | None]]:
        messages = self.rebuild_view().messages
        if not messages:
            return []

        assistant_index = None
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].role == "assistant":
                assistant_index = index
                break
        if assistant_index is None:
            return []

        assistant = messages[assistant_index]
        tool_call_parts = [part for part in assistant.parts if part.kind == "tool_call"]
        tool_calls = [_tool_call_from_part(part) for part in tool_call_parts]
        if not tool_calls:
            return []

        completed_ids: set[str] = set()
        for message in messages[assistant_index + 1 :]:
            if message.role != "tool":
                return []
            for part in message.parts:
                if part.kind == "tool_result" and part.metadata.get("tool_call_id"):
                    completed_ids.add(str(part.metadata["tool_call_id"]))

        pending_calls = [tool_call for tool_call in tool_calls if tool_call.id not in completed_ids]
        if not pending_calls:
            return []
        first_pending = pending_calls[0]
        source_part = next(part for part in tool_call_parts if str(part.metadata.get("tool_call_id") or "") == first_pending.id)
        persisted_review_only = source_part.metadata.get("prewrite_review_only")
        return [
            (
                first_pending,
                pending_calls[1:],
                persisted_review_only if isinstance(persisted_review_only, bool) else None,
            )
        ]


def _tool_call_from_part(part: MessagePart) -> ToolCall:
    arguments = deepcopy(part.metadata.get("arguments", {}))
    return ToolCall(
        id=str(part.metadata["tool_call_id"]),
        name=str(part.metadata["tool_name"]),
        arguments=arguments,
    )


def _tool_result_message_ids_from_view(view: SessionView) -> dict[str, str]:
    """Index existing tool results so resumed sessions preserve idempotent settlement."""

    result: dict[str, str] = {}
    for message in view.messages:
        if message.role != "tool":
            continue
        for part in message.parts:
            if part.kind != "tool_result":
                continue
            tool_call_id = part.metadata.get("tool_call_id")
            if tool_call_id:
                result.setdefault(str(tool_call_id), message.id)
    return result


def _infer_turn_counter(messages: list[AgentMessage]) -> int:
    """从已恢复的消息里推断下一轮 turn 编号。"""

    return sum(1 for message in messages if message.role == "user")


def create_project_permission_manager(
    project_root: str | Path,
    *,
    grants: PermissionGrantStore | None = None,
    mode: PermissionMode = PermissionMode.STANDARD,
) -> PermissionManager:
    return PermissionManager(policy=DefaultPermissionPolicy(project_root), grants=grants, mode=mode)


def _task_boundary_required_stable_count(permission_manager: PermissionManager | None) -> int:
    if permission_manager is not None and permission_manager.mode == PermissionMode.BYPASS:
        return 1
    return 2
