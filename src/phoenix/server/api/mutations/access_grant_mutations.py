from typing import Optional, cast

import strawberry
from sqlalchemy import ColumnElement, delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from strawberry import UNSET
from strawberry.relay import GlobalID
from strawberry.types import Info

from phoenix.db import models
from phoenix.server.access import (
    DEFAULT_PERMISSION_SET,
    OBJECT_TYPE_DATASET,
    OBJECT_TYPE_PROJECT,
    OBJECT_TYPE_PROMPT,
    Permission,
    can_access,
)
from phoenix.server.api.auth import IsLocked, IsNotReadOnly, IsNotViewer
from phoenix.server.api.context import Context
from phoenix.server.api.exceptions import BadRequest, NotFound
from phoenix.server.api.queries import Query
from phoenix.server.api.types.AccessSubjectKind import AccessSubjectKind
from phoenix.server.api.types.Dataset import Dataset
from phoenix.server.api.types.node import from_global_id_with_expected_type
from phoenix.server.api.types.PermissionSet import PermissionSet
from phoenix.server.api.types.Project import Project
from phoenix.server.api.types.Prompt import Prompt
from phoenix.server.api.types.User import User
from phoenix.server.api.types.UserGroup import UserGroup


@strawberry.input(one_of=True)
class AccessGrantSubjectInput:
    user_id: Optional[GlobalID] = UNSET
    user_group_id: Optional[GlobalID] = UNSET
    is_everyone: Optional[bool] = UNSET


@strawberry.input(one_of=True)
class AccessGrantObjectInput:
    project_id: Optional[GlobalID] = UNSET
    dataset_id: Optional[GlobalID] = UNSET
    prompt_id: Optional[GlobalID] = UNSET


@strawberry.input
class AccessGrantInput:
    subject: AccessGrantSubjectInput
    object: AccessGrantObjectInput
    permission_set_id: Optional[GlobalID] = UNSET


@strawberry.type
class AccessGrantMutationPayload:
    query: Query


async def _resolve_project_rowid(session: AsyncSession, project_id: GlobalID) -> int:
    try:
        rowid = from_global_id_with_expected_type(project_id, Project.__name__)
    except ValueError:
        raise NotFound(f"Unknown project: {project_id}") from None
    exists = await session.scalar(select(models.Project.id).where(models.Project.id == rowid))
    if exists is None:
        raise NotFound(f"Unknown project: {project_id}")
    return rowid


async def _resolve_prompt_rowid(session: AsyncSession, prompt_id: GlobalID) -> int:
    try:
        rowid = from_global_id_with_expected_type(prompt_id, Prompt.__name__)
    except ValueError:
        raise NotFound(f"Unknown prompt: {prompt_id}") from None
    exists = await session.scalar(select(models.Prompt.id).where(models.Prompt.id == rowid))
    if exists is None:
        raise NotFound(f"Unknown prompt: {prompt_id}")
    return rowid


async def _resolve_dataset_rowid(session: AsyncSession, dataset_id: GlobalID) -> int:
    try:
        rowid = from_global_id_with_expected_type(dataset_id, Dataset.__name__)
    except ValueError:
        raise NotFound(f"Unknown dataset: {dataset_id}") from None
    exists = await session.scalar(select(models.Dataset.id).where(models.Dataset.id == rowid))
    if exists is None:
        raise NotFound(f"Unknown dataset: {dataset_id}")
    return rowid


async def _resolve_object_rowid(
    session: AsyncSession, object: AccessGrantObjectInput
) -> tuple[str, int]:
    if object.dataset_id is None:
        raise BadRequest("datasetId must not be null")
    if object.dataset_id is not UNSET:
        return OBJECT_TYPE_DATASET, await _resolve_dataset_rowid(session, object.dataset_id)
    if object.project_id is None:
        raise BadRequest("projectId must not be null")
    if object.project_id is not UNSET:
        return OBJECT_TYPE_PROJECT, await _resolve_project_rowid(session, object.project_id)
    if object.prompt_id is None:
        raise BadRequest("promptId must not be null")
    if object.prompt_id is not UNSET:
        return OBJECT_TYPE_PROMPT, await _resolve_prompt_rowid(session, object.prompt_id)
    raise BadRequest("An object is required")


async def _resolve_permission_set_id(
    session: AsyncSession, permission_set_id: Optional[GlobalID]
) -> Optional[int]:
    """The permission set to attach to a grant — viewer (visibility), editor (mutate), or manager
    (manage access); the oracle enforces each tier at its permission level. An explicit id is
    validated; absent, it defaults to the built-in "Resource Viewer"."""
    if permission_set_id is not None and permission_set_id is not UNSET:
        try:
            permission_set_rowid = from_global_id_with_expected_type(
                permission_set_id, PermissionSet.__name__
            )
        except ValueError:
            raise NotFound(f"Unknown permission set: {permission_set_id}") from None
        role_id: Optional[int] = await session.scalar(
            select(models.PermissionSet.id).where(models.PermissionSet.id == permission_set_rowid)
        )
        if role_id is None:
            raise NotFound(f"Unknown permission set: {permission_set_id}")
        return role_id
    return cast(
        Optional[int],
        await session.scalar(
            select(models.PermissionSet.id).where(
                models.PermissionSet.name == DEFAULT_PERMISSION_SET
            )
        ),
    )


async def _assert_can_manage_object(
    info: Info[Context, None], session: AsyncSession, object_type: str, object_rowid: int
) -> None:
    """The caller must hold OBJ_MANAGE_ACCESS on the object. Unauthorized is surfaced as
    not-found (indistinguishable from an object the caller cannot see). A no-op when auth is
    disabled."""
    user_id = info.context.user_id
    if user_id is None:
        return
    if not await can_access(
        session,
        user_id=user_id,
        object_type=object_type,
        object_id=object_rowid,
        enabled=True,
        permission=Permission.OBJ_MANAGE_ACCESS,
    ):
        raise NotFound("Unknown access object")


async def _resolve_subject_rowid(
    session: AsyncSession, subject: AccessGrantSubjectInput
) -> tuple[AccessSubjectKind, Optional[int]]:
    """Validate and resolve the grant subject. EVERYONE carries no id; USER and GROUP must
    reference an existing row by Relay id."""
    if subject.user_id is None:
        raise BadRequest("userId must not be null")
    if subject.user_id is not UNSET:
        user_id = subject.user_id
        try:
            rowid = from_global_id_with_expected_type(user_id, User.__name__)
        except ValueError:
            raise NotFound(f"Unknown user: {user_id}") from None
        exists = await session.scalar(select(models.User.id).where(models.User.id == rowid))
        if exists is None:
            raise NotFound(f"Unknown user: {user_id}")
        return AccessSubjectKind.USER, rowid
    if subject.user_group_id is None:
        raise BadRequest("userGroupId must not be null")
    if subject.user_group_id is not UNSET:
        user_group_id = subject.user_group_id
        try:
            rowid = from_global_id_with_expected_type(user_group_id, UserGroup.__name__)
        except ValueError:
            raise NotFound(f"Unknown group: {user_group_id}") from None
        exists = await session.scalar(
            select(models.UserGroup.id).where(models.UserGroup.id == rowid)
        )
        if exists is None:
            raise NotFound(f"Unknown group: {user_group_id}")
        return AccessSubjectKind.GROUP, rowid
    if subject.is_everyone is not UNSET:
        if subject.is_everyone is not True:
            raise BadRequest("isEveryone must be true when provided")
        return AccessSubjectKind.EVERYONE, None
    raise BadRequest("A subject is required")


def _subject_id_clause(subject_id: Optional[int]) -> "ColumnElement[bool]":
    """Match a grant's subject id, treating EVERYONE's null id correctly (``== None`` would
    never match a NULL column)."""
    if subject_id is None:
        return models.AccessGrant.subject_id.is_(None)
    return models.AccessGrant.subject_id == subject_id


@strawberry.type
class AccessGrantMutationMixin:
    @strawberry.mutation(permission_classes=[IsNotReadOnly, IsNotViewer, IsLocked])  # type: ignore
    async def grant_access(
        self, info: Info[Context, None], input: AccessGrantInput
    ) -> AccessGrantMutationPayload:
        """Grant a user or group an permission set on an access-controlled object. The model is
        allow-only — the grant only *adds* a grantee to an object that is otherwise admin-only.
        Authoring requires OBJ_MANAGE_ACCESS on the target object. Idempotent:
        re-granting the same subject updates the role."""
        async with info.context.db() as session:
            object_type, object_rowid = await _resolve_object_rowid(session, input.object)
            await _assert_can_manage_object(info, session, object_type, object_rowid)
            subject_kind, subject_rowid = await _resolve_subject_rowid(session, input.subject)
            role_id = await _resolve_permission_set_id(session, input.permission_set_id)
            existing = await session.scalar(
                select(models.AccessGrant).where(
                    models.AccessGrant.subject_kind == subject_kind.value,
                    _subject_id_clause(subject_rowid),
                    models.AccessGrant.object_type == object_type,
                    models.AccessGrant.object_id == object_rowid,
                    models.AccessGrant.selector_kind == "ids",
                    models.AccessGrant.effect == "allow",
                )
            )
            if existing is None:
                session.add(
                    models.AccessGrant(
                        subject_kind=subject_kind.value,
                        subject_id=subject_rowid,
                        role_id=role_id,
                        object_type=object_type,
                        object_id=object_rowid,
                        selector_kind="ids",
                        effect="allow",
                    )
                )
            else:
                existing.role_id = role_id
        return AccessGrantMutationPayload(query=Query())

    @strawberry.mutation(permission_classes=[IsNotReadOnly, IsNotViewer, IsLocked])  # type: ignore
    async def revoke_access(
        self, info: Info[Context, None], input: AccessGrantInput
    ) -> AccessGrantMutationPayload:
        """Remove a subject's grant on an access-controlled object. Requires OBJ_MANAGE_ACCESS
        (or admin privileges)."""
        async with info.context.db() as session:
            object_type, object_rowid = await _resolve_object_rowid(session, input.object)
            await _assert_can_manage_object(info, session, object_type, object_rowid)
            subject_kind, subject_rowid = await _resolve_subject_rowid(session, input.subject)
            await session.execute(
                delete(models.AccessGrant).where(
                    models.AccessGrant.subject_kind == subject_kind.value,
                    _subject_id_clause(subject_rowid),
                    models.AccessGrant.object_type == object_type,
                    models.AccessGrant.object_id == object_rowid,
                    models.AccessGrant.selector_kind == "ids",
                )
            )
        return AccessGrantMutationPayload(query=Query())
