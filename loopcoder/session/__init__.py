"""用户可见 session 能力的边界模块。

`loopcoder.context` 仍然负责底层事件日志、上下文投影、checkpoint 和压缩；
本包后续只承载 catalog、resume 编排、只读 transcript 和 share export 等用户入口。
"""

from loopcoder.session.errors import (
    SessionCorruptError,
    SessionEmptyError,
    SessionError,
    SessionInvalidIdError,
    SessionNotFoundError,
)
from loopcoder.session.models import (
    RedactionOptions,
    SessionRecord,
    ShareOptions,
    Transcript,
    TranscriptEntry,
)
from loopcoder.session.share import SessionShareService
from loopcoder.session.transcript import TranscriptBuilder

__all__ = [
    "RedactionOptions",
    "SessionCorruptError",
    "SessionEmptyError",
    "SessionError",
    "SessionInvalidIdError",
    "SessionNotFoundError",
    "SessionRecord",
    "SessionShareService",
    "ShareOptions",
    "Transcript",
    "TranscriptBuilder",
    "TranscriptEntry",
]
