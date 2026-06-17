"""The access-control admin API (the REST complement to the GraphQL grant mutations).

A deliberately small surface. The grantable ``object_type`` enum is exactly the three
access *roots* — project / dataset / prompt. Containment children (spans, traces,
annotations) and eval artifacts (experiments, runs) are absent *by design*: their access is
inherited from a parent, so they are never granted directly. The model is **allow-only**:
there is no ``effect`` field, so the schema cannot express deny. Type-wide authoring is
**admin-only**; per-object authoring requires ``OBJ_MANAGE_ACCESS`` on that object (open when
auth is disabled).

**Id convention:** every id on the wire is a Relay **GlobalID string**, matching the rest
of the v1 API — objects (``Project``/``Dataset``/``Prompt``), users (``User``), and the
access-control entities (``UserGroup``/``Role``/``ServiceAccount``/``AccessGrant``/
``PermissionSet``). The latter are not GraphQL nodes, so their GlobalIDs are opaque type-tagged
tokens that do not resolve through ``node()`` — but a client uses one id format everywhere.
"""

from enum import Enum
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload
from starlette.requests import Request
from strawberry.relay import GlobalID

from phoenix.config import get_env_access_control_enabled
from phoenix.db import models
from phoenix.server.access import (
    DEFAULT_PERMISSION_SET,
    OBJECT_TYPE_DATASET,
    OBJECT_TYPE_PROJECT,
    OBJECT_TYPE_PROMPT,
    Permission,
    SubjectKind,
    can_access,
    subjects_for,
)
from phoenix.server.api.types.node import from_global_id_with_expected_type
from phoenix.server.authorization import require_admin

from .models import V1RoutesBaseModel
from .utils import ResponseBody, add_errors_to_responses, request_user_id

# Org-wide access administration — authoring type-wide grants, managing groups, reading
# enforcement. Admin-only (open when auth is disabled).
router = APIRouter(tags=["access"], dependencies=[Depends(require_admin)])

# Per-object access administration — granting/revoking on, and auditing, a specific object.
# Not blanket admin: each endpoint authorizes "admin OR a subject holding OBJ_MANAGE_ACCESS"
# in-body, so a resource manager can share that object without being a global administrator.
owner_router = APIRouter(tags=["access"])


async def _assert_can_manage_object(
    request: Request,
    session: AsyncSession,
    object_type: str,
    object_id: int,
    not_found_detail: str,
) -> None:
    """Authorize a per-object access-admin action: the caller must hold OBJ_MANAGE_ACCESS on
    the object. Unauthorized is surfaced as 404, indistinguishable from a missing object. A
    no-op when auth is disabled (access control presupposes auth)."""
    user_id = request_user_id(request)
    if user_id is None:
        return
    if not await can_access(
        session,
        user_id=user_id,
        object_type=object_type,
        object_id=object_id,
        enabled=True,
        permission=Permission.OBJ_MANAGE_ACCESS,
    ):
        raise HTTPException(404, not_found_detail)


_SELECTOR_IDS = "ids"
_SELECTOR_ALL = "all"
_EFFECT_ALLOW = "allow"
# Admin-managed groups live under their own provider so the login-time IdP reconcile
# (which is provider-scoped) never touches them. This is how a deployment with no identity
# provider gets groups at all: admins create them in-product instead of syncing from an IdP.
_LOCAL_PROVIDER = "local"

# GlobalID type tags for the access-control entities.
_USER = "User"
_GROUP = "UserGroup"
_ROLE = "Role"
_SERVICE_ACCOUNT = "ServiceAccount"
_GRANT = "AccessGrant"
_PERMISSION_SET = "PermissionSet"


class GrantObjectType(str, Enum):
    """The grantable access roots. Containment children and eval artifacts are
    intentionally absent: their access is inherited from a parent, never granted directly."""

    project = OBJECT_TYPE_PROJECT
    dataset = OBJECT_TYPE_DATASET
    prompt = OBJECT_TYPE_PROMPT


class GrantSubjectKind(str, Enum):
    """Who a grant targets. ``everyone`` is the only kind with no id — a deliberate
    'make this object visible to all' grant, not a seeded baseline."""

    user = SubjectKind.USER.value
    group = SubjectKind.GROUP.value
    role = SubjectKind.ROLE.value
    service_account = SubjectKind.SERVICE_ACCOUNT.value
    everyone = SubjectKind.EVERYONE.value


_GLOBAL_ID_TYPE_NAME = {
    GrantObjectType.project: "Project",
    GrantObjectType.dataset: "Dataset",
    GrantObjectType.prompt: "Prompt",
}
_OBJECT_MODEL = {
    GrantObjectType.project: models.Project,
    GrantObjectType.dataset: models.Dataset,
    GrantObjectType.prompt: models.Prompt,
}
_SUBJECT_TYPE_NAME = {
    GrantSubjectKind.user: _USER,
    GrantSubjectKind.group: _GROUP,
    GrantSubjectKind.role: _ROLE,
    GrantSubjectKind.service_account: _SERVICE_ACCOUNT,
}
# The row a grant subject must reference, by kind. `everyone` has no row; `service_account`
# is reserved but has no identity lifecycle yet, so it is rejected rather than resolved.
_SUBJECT_MODEL = {
    GrantSubjectKind.user: models.User,
    GrantSubjectKind.group: models.UserGroup,
    GrantSubjectKind.role: models.UserRole,
}


def _encode(type_name: str, rowid: int) -> str:
    return str(GlobalID(type_name, str(rowid)))


def _decode(type_name: str, gid: str) -> int:
    try:
        return from_global_id_with_expected_type(GlobalID.from_id(gid), type_name)
    except Exception:
        raise HTTPException(422, f"Invalid {type_name} id: {gid!r}")


class Subject(V1RoutesBaseModel):
    kind: GrantSubjectKind
    # GlobalID of the user / group / role / service account. Null only for `everyone`.
    id: Optional[str] = None


class GrantCreate(V1RoutesBaseModel):
    subject: Subject
    object_type: GrantObjectType
    # GlobalID of the object. Omit for a deliberate type-wide grant ("all projects").
    object_id: Optional[str] = None
    # Permission set conferred — viewer / editor / manager. Omit → the view-only default.
    role: Optional[str] = None


class Grant(V1RoutesBaseModel):
    id: str  # GlobalID
    subject: Subject
    object_type: GrantObjectType
    object_id: Optional[str]  # GlobalID, or null for a type-wide grant
    role: Optional[str]


class PermissionSetData(V1RoutesBaseModel):
    id: str  # GlobalID
    name: str
    is_built_in: bool
    permissions: list[str]


class Enforcement(V1RoutesBaseModel):
    enabled: bool
    source: str


class GroupCreate(V1RoutesBaseModel):
    name: str


class MemberCreate(V1RoutesBaseModel):
    user_id: str  # GlobalID


class GroupData(V1RoutesBaseModel):
    id: str  # GlobalID
    name: str
    member_user_ids: list[str]  # GlobalIDs


class GrantResponseBody(ResponseBody[Grant]):
    pass


class GrantsResponseBody(ResponseBody[list[Grant]]):
    pass


class SubjectsResponseBody(ResponseBody[list[Subject]]):
    pass


class PermissionSetsResponseBody(ResponseBody[list[PermissionSetData]]):
    pass


class EnforcementResponseBody(ResponseBody[Enforcement]):
    pass


class GroupResponseBody(ResponseBody[GroupData]):
    pass


class GroupsResponseBody(ResponseBody[list[GroupData]]):
    pass


def _decode_subject_id(kind: GrantSubjectKind, gid: Optional[str]) -> Optional[int]:
    """The subject's row id, decoding its GlobalID and validating it matches the kind.
    ``everyone`` carries no id; every other kind requires one."""
    if kind is GrantSubjectKind.everyone:
        if gid is not None:
            raise HTTPException(422, "subject.id must be omitted for kind=everyone")
        return None
    if gid is None:
        raise HTTPException(422, f"subject.id is required for kind={kind.value}")
    return _decode(_SUBJECT_TYPE_NAME[kind], gid)


async def _resolve_subject_rowid(
    session: AsyncSession, kind: GrantSubjectKind, gid: Optional[str]
) -> Optional[int]:
    """Decode *and validate* a grant subject. ``everyone`` carries no id. ``service_account``
    is reserved but not yet grantable — there is no service-account identity lifecycle, so a
    grant naming one would point at a subject that can never exist; reject it. Every other kind
    must reference a row that exists: a grant to a deleted user/group/role is a client error,
    not a silently-stored dangling subject (the GlobalID shape alone says nothing about
    existence, which is why decoding is not enough)."""
    if kind is GrantSubjectKind.service_account:
        raise HTTPException(422, "service_account subjects are reserved but not yet supported")
    subject_id = _decode_subject_id(kind, gid)
    if subject_id is None:  # everyone
        return None
    model = _SUBJECT_MODEL[kind]
    exists = await session.scalar(select(model.id).where(model.id == subject_id))
    if exists is None:
        raise HTTPException(404, f"{kind.value} {gid} not found")
    return subject_id


def _encode_subject(kind_str: str, rowid: Optional[int]) -> Subject:
    kind = GrantSubjectKind(kind_str)
    if kind is GrantSubjectKind.everyone or rowid is None:
        return Subject(kind=kind, id=None)
    return Subject(kind=kind, id=_encode(_SUBJECT_TYPE_NAME[kind], rowid))


async def _resolve_object_rowid(
    session: AsyncSession, object_type: GrantObjectType, object_id: str
) -> int:
    """Resolve an object GlobalID to its row id, validating type and existence
    (404 = absent — admins see all, so there is no unauthorized case here)."""
    rowid = _decode(_GLOBAL_ID_TYPE_NAME[object_type], object_id)
    exists = await session.scalar(
        select(_OBJECT_MODEL[object_type].id).where(_OBJECT_MODEL[object_type].id == rowid)
    )
    if exists is None:
        raise HTTPException(404, f"{object_type.value} {object_id} not found")
    return rowid


async def _resolve_role_id(session: AsyncSession, role: Optional[str]) -> Optional[int]:
    """The permission set a grant confers — viewer (visibility), editor (mutate), or manager
    (manage access). The oracle enforces these tiers per permission. Absent ⇒ the view-only
    default; a *named* role that does not exist is a client error."""
    name = role or DEFAULT_PERMISSION_SET
    role_id = await session.scalar(
        select(models.PermissionSet.id).where(models.PermissionSet.name == name)
    )
    if role_id is None and role is not None:
        raise HTTPException(422, f"Unknown permission set: {role}")
    return role_id


async def _role_names(session: AsyncSession) -> dict[int, str]:
    rows = await session.execute(select(models.PermissionSet.id, models.PermissionSet.name))
    return {id_: name for id_, name in rows}


def _object_global_id(object_type: GrantObjectType, rowid: Optional[int]) -> Optional[str]:
    if rowid is None:
        return None
    return _encode(_GLOBAL_ID_TYPE_NAME[object_type], rowid)


def _to_grant(row: models.AccessGrant, role_names: dict[int, str]) -> Grant:
    object_type = GrantObjectType(row.object_type)
    return Grant(
        id=_encode(_GRANT, row.id),
        subject=_encode_subject(row.subject_kind, row.subject_id),
        object_type=object_type,
        object_id=_object_global_id(object_type, row.object_id),
        role=role_names.get(row.role_id) if row.role_id is not None else None,
    )


@owner_router.post(
    "/access/grants",
    operation_id="createAccessGrant",
    summary="Author an access grant (allow-only, idempotent)",
    responses=add_errors_to_responses([403, 404, 422]),
)
async def create_access_grant(request: Request, body: GrantCreate) -> GrantResponseBody:
    """Grant a subject access to an object (or, with ``object_id`` omitted, to every
    object of that type). Monotonic and allow-only: grants only ever *add* access.
    Idempotent — re-granting the same subject+object updates only the role. Authoring a grant
    on a specific object needs OBJ_MANAGE_ACCESS on it; a type-wide grant is admin-only."""
    selector = _SELECTOR_ALL if body.object_id is None else _SELECTOR_IDS
    async with request.app.state.db() as session:
        subject_id = await _resolve_subject_rowid(session, body.subject.kind, body.subject.id)
        if body.object_id is None:
            # A type-wide ("all of a type") grant names no owned object — admin only.
            require_admin(request)
            object_rowid = None
        else:
            object_rowid = await _resolve_object_rowid(session, body.object_type, body.object_id)
            await _assert_can_manage_object(
                request,
                session,
                body.object_type.value,
                object_rowid,
                f"{body.object_type.value} {body.object_id} not found",
            )
        role_id = await _resolve_role_id(session, body.role)
        grant = await session.scalar(
            select(models.AccessGrant).where(
                models.AccessGrant.subject_kind == body.subject.kind.value,
                models.AccessGrant.subject_id == subject_id,
                models.AccessGrant.object_type == body.object_type.value,
                models.AccessGrant.object_id == object_rowid,
                models.AccessGrant.selector_kind == selector,
                models.AccessGrant.effect == _EFFECT_ALLOW,
            )
        )
        if grant is None:
            grant = models.AccessGrant(
                subject_kind=body.subject.kind.value,
                subject_id=subject_id,
                role_id=role_id,
                object_type=body.object_type.value,
                object_id=object_rowid,
                selector_kind=selector,
                effect=_EFFECT_ALLOW,
            )
            session.add(grant)
        else:
            grant.role_id = role_id
        await session.flush()
        names = await _role_names(session)
        data = _to_grant(grant, names)
    return GrantResponseBody(data=data)


@router.get(
    "/access/grants",
    operation_id="listAccessGrants",
    summary="List access grants",
    responses=add_errors_to_responses([403, 422]),
)
async def list_access_grants(
    request: Request,
    object_type: Optional[GrantObjectType] = Query(default=None),
    object_id: Optional[str] = Query(
        default=None, description="Filter to one object (requires object_type)."
    ),
    subject_kind: Optional[GrantSubjectKind] = Query(default=None),
    subject_id: Optional[str] = Query(
        default=None, description="Filter to one subject (requires subject_kind)."
    ),
) -> GrantsResponseBody:
    if object_id is not None and object_type is None:
        raise HTTPException(422, "object_type is required when object_id is given")
    if subject_id is not None and subject_kind is None:
        raise HTTPException(422, "subject_kind is required when subject_id is given")
    grantable = [t.value for t in GrantObjectType]
    async with request.app.state.db() as session:
        stmt = select(models.AccessGrant).where(
            models.AccessGrant.effect == _EFFECT_ALLOW,
            models.AccessGrant.object_type.in_(grantable),
        )
        if object_type is not None:
            stmt = stmt.where(models.AccessGrant.object_type == object_type.value)
        if object_id is not None and object_type is not None:
            rowid = await _resolve_object_rowid(session, object_type, object_id)
            stmt = stmt.where(models.AccessGrant.object_id == rowid)
        if subject_kind is not None:
            stmt = stmt.where(models.AccessGrant.subject_kind == subject_kind.value)
        if subject_id is not None and subject_kind is not None:
            stmt = stmt.where(
                models.AccessGrant.subject_id == _decode_subject_id(subject_kind, subject_id)
            )
        rows = (await session.scalars(stmt.order_by(models.AccessGrant.id.desc()))).all()
        names = await _role_names(session)
        data = [_to_grant(r, names) for r in rows]
    return GrantsResponseBody(data=data)


@owner_router.delete(
    "/access/grants/{grant_id}",
    operation_id="deleteAccessGrant",
    summary="Revoke an access grant",
    status_code=204,
    responses=add_errors_to_responses([403, 404, 422]),
)
async def delete_access_grant(
    request: Request, grant_id: str = Path(description="The access grant GlobalID")
) -> None:
    rowid = _decode(_GRANT, grant_id)
    async with request.app.state.db() as session:
        grant = await session.scalar(
            select(models.AccessGrant).where(models.AccessGrant.id == rowid)
        )
        if grant is None:
            raise HTTPException(404, f"Grant {grant_id} not found")
        # Revoking on a specific object needs OBJ_MANAGE_ACCESS on it; a
        # type-wide grant is admin-only.
        if grant.selector_kind == _SELECTOR_ALL or grant.object_id is None:
            require_admin(request)
        else:
            await _assert_can_manage_object(
                request, session, grant.object_type, grant.object_id, f"Grant {grant_id} not found"
            )
        await session.delete(grant)


@owner_router.get(
    "/access/objects/{object_type}/{object_id}/subjects",
    operation_id="listObjectSubjects",
    summary='Who can access this object ("who can see X?")',
    responses=add_errors_to_responses([403, 404, 422]),
)
async def list_object_subjects(
    request: Request,
    object_type: GrantObjectType = Path(),
    object_id: str = Path(description="The object GlobalID"),
) -> SubjectsResponseBody:
    """The audit read: the subjects currently granted access to the object (plus its
    creator). Needs OBJ_MANAGE_ACCESS on the object. Administrators have
    implicit access and are not enumerated."""
    async with request.app.state.db() as session:
        rowid = await _resolve_object_rowid(session, object_type, object_id)
        await _assert_can_manage_object(
            request, session, object_type.value, rowid, f"{object_type.value} {object_id} not found"
        )
        resolved = await subjects_for(session, object_type.value, rowid)
    data = [
        _encode_subject(s.kind.value, None if s.kind is SubjectKind.EVERYONE else s.id)
        for s in resolved
    ]
    return SubjectsResponseBody(data=data)


@router.get(
    "/access/object-roles",
    operation_id="listPermissionSets",
    summary="List the permission sets a grant may confer",
    responses=add_errors_to_responses([403]),
)
async def list_permission_sets(request: Request) -> PermissionSetsResponseBody:
    """The permission sets a grant may confer — viewer (visibility), editor (mutate), manager
    (manage access). The oracle enforces each tier at its permission level."""
    async with request.app.state.db() as session:
        roles = (
            (
                await session.scalars(
                    select(models.PermissionSet)
                    .options(joinedload(models.PermissionSet.permissions))
                    .order_by(models.PermissionSet.is_built_in.desc(), models.PermissionSet.name)
                )
            )
            .unique()
            .all()
        )
    data = [
        PermissionSetData(
            id=_encode(_PERMISSION_SET, r.id),
            name=r.name,
            is_built_in=r.is_built_in,
            permissions=[p.permission for p in r.permissions],
        )
        for r in roles
    ]
    return PermissionSetsResponseBody(data=data)


@router.get(
    "/access/enforcement",
    operation_id="getAccessEnforcement",
    summary="Whether access control is enforcing",
    responses=add_errors_to_responses([403]),
)
async def get_access_enforcement() -> EnforcementResponseBody:
    """The DB-latched activation state — the source of truth for whether enforcement is on.
    Read-only: there is intentionally no enable/disable endpoint (enabling is the one-way env
    switch; disabling is a deliberate ops action)."""
    return EnforcementResponseBody(
        data=Enforcement(enabled=get_env_access_control_enabled(), source="db-latch")
    )


# --- local (admin-managed) groups ---------------------------------------------------
#
# A deployment with only basic auth has no identity provider to sync groups from, so without
# these endpoints it could grant per-user only. They touch *only* `provider="local"` groups;
# IdP-synced groups (`oauth2:*`, `ldap`) are owned by the login-time reconcile and are not
# mutable here.


async def _local_group_or_404(session: AsyncSession, group_rowid: int) -> models.UserGroup:
    group = await session.scalar(
        select(models.UserGroup).where(
            models.UserGroup.id == group_rowid,
            models.UserGroup.provider == _LOCAL_PROVIDER,
        )
    )
    if group is None:
        raise HTTPException(404, f"Local group {_encode(_GROUP, group_rowid)} not found")
    return group


async def _group_data(session: AsyncSession, group: models.UserGroup) -> GroupData:
    member_ids = list(
        await session.scalars(
            select(models.UserGroupMembership.user_id).where(
                models.UserGroupMembership.user_group_id == group.id
            )
        )
    )
    return GroupData(
        id=_encode(_GROUP, group.id),
        name=group.display_name or group.group_key,
        member_user_ids=[_encode(_USER, uid) for uid in member_ids],
    )


@router.post(
    "/access/groups",
    operation_id="createLocalGroup",
    summary="Create a local (admin-managed) group",
    responses=add_errors_to_responses([403, 409, 422]),
)
async def create_local_group(request: Request, body: GroupCreate) -> GroupResponseBody:
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "name must be non-empty")
    async with request.app.state.db() as session:
        existing = await session.scalar(
            select(models.UserGroup.id).where(
                models.UserGroup.provider == _LOCAL_PROVIDER,
                models.UserGroup.group_key == name,
            )
        )
        if existing is not None:
            raise HTTPException(409, f"A local group named {name!r} already exists")
        group = models.UserGroup(provider=_LOCAL_PROVIDER, group_key=name, display_name=name)
        session.add(group)
        await session.flush()
        data = await _group_data(session, group)
    return GroupResponseBody(data=data)


@router.get(
    "/access/groups",
    operation_id="listLocalGroups",
    summary="List local (admin-managed) groups",
    responses=add_errors_to_responses([403]),
)
async def list_local_groups(request: Request) -> GroupsResponseBody:
    async with request.app.state.db() as session:
        groups = (
            await session.scalars(
                select(models.UserGroup)
                .where(models.UserGroup.provider == _LOCAL_PROVIDER)
                .order_by(models.UserGroup.id)
            )
        ).all()
        data = [await _group_data(session, g) for g in groups]
    return GroupsResponseBody(data=data)


@router.delete(
    "/access/groups/{group_id}",
    operation_id="deleteLocalGroup",
    summary="Delete a local group (and sweep its grants)",
    status_code=204,
    responses=add_errors_to_responses([403, 404, 422]),
)
async def delete_local_group(
    request: Request, group_id: str = Path(description="The local group GlobalID")
) -> None:
    group_rowid = _decode(_GROUP, group_id)
    async with request.app.state.db() as session:
        await _local_group_or_404(session, group_rowid)
        # A group's acl rows carry no foreign key, so deleting the group does not cascade
        # them. Remove them in the same transaction, before the id can be reused — otherwise a
        # future group assigned this id would silently inherit the stale grants.
        await session.execute(
            delete(models.AccessGrant).where(
                models.AccessGrant.subject_kind == SubjectKind.GROUP.value,
                models.AccessGrant.subject_id == group_rowid,
            )
        )
        # Memberships cascade via their FK; delete the group itself.
        await session.execute(delete(models.UserGroup).where(models.UserGroup.id == group_rowid))


@router.post(
    "/access/groups/{group_id}/members",
    operation_id="addLocalGroupMember",
    summary="Add a user to a local group",
    responses=add_errors_to_responses([403, 404, 422]),
)
async def add_local_group_member(
    request: Request,
    body: MemberCreate,
    group_id: str = Path(description="The local group GlobalID"),
) -> GroupResponseBody:
    group_rowid = _decode(_GROUP, group_id)
    user_rowid = _decode(_USER, body.user_id)
    async with request.app.state.db() as session:
        group = await _local_group_or_404(session, group_rowid)
        user_exists = await session.scalar(
            select(models.User.id).where(models.User.id == user_rowid)
        )
        if user_exists is None:
            raise HTTPException(404, f"User {body.user_id} not found")
        existing = await session.scalar(
            select(models.UserGroupMembership.user_id).where(
                models.UserGroupMembership.user_group_id == group_rowid,
                models.UserGroupMembership.user_id == user_rowid,
            )
        )
        if existing is None:
            session.add(models.UserGroupMembership(user_group_id=group_rowid, user_id=user_rowid))
            await session.flush()
        data = await _group_data(session, group)
    return GroupResponseBody(data=data)


@router.delete(
    "/access/groups/{group_id}/members/{user_id}",
    operation_id="removeLocalGroupMember",
    summary="Remove a user from a local group",
    status_code=204,
    responses=add_errors_to_responses([403, 404, 422]),
)
async def remove_local_group_member(
    request: Request,
    group_id: str = Path(description="The local group GlobalID"),
    user_id: str = Path(description="The user GlobalID to remove"),
) -> None:
    group_rowid = _decode(_GROUP, group_id)
    user_rowid = _decode(_USER, user_id)
    async with request.app.state.db() as session:
        await _local_group_or_404(session, group_rowid)
        deleted = await session.scalar(
            delete(models.UserGroupMembership)
            .where(
                models.UserGroupMembership.user_group_id == group_rowid,
                models.UserGroupMembership.user_id == user_rowid,
            )
            .returning(models.UserGroupMembership.user_id)
        )
        if deleted is None:
            raise HTTPException(404, f"User {user_id} is not a member of group {group_id}")
