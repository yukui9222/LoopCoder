"""权限系统公共入口。

第一阶段只暴露纯策略和内存授权匹配，不直接绑定工具执行层。
"""

from loopcoder.permissions.grants import FilePermissionGrantStore, PermissionGrantStore
from loopcoder.permissions.manager import PermissionManager
from loopcoder.permissions.policy import DefaultPermissionPolicy
from loopcoder.permissions.types import (
    PermissionAction,
    PermissionConfirmationChoice,
    PermissionDecision,
    PermissionDecisionKind,
    PermissionGrant,
    PermissionMode,
    PermissionPersistence,
    PermissionRequest,
    PermissionScopeType,
)

__all__ = [
    "DefaultPermissionPolicy",
    "FilePermissionGrantStore",
    "PermissionAction",
    "PermissionConfirmationChoice",
    "PermissionDecision",
    "PermissionDecisionKind",
    "PermissionGrant",
    "PermissionGrantStore",
    "PermissionManager",
    "PermissionMode",
    "PermissionPersistence",
    "PermissionRequest",
    "PermissionScopeType",
]
