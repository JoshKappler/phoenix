from __future__ import annotations

from asyncio import get_running_loop
from dataclasses import dataclass, field
from functools import cached_property, partial
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, cast

from pydantic import SecretStr
from sqlalchemy import true
from sqlalchemy.orm import InstrumentedAttribute
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse
from strawberry.fastapi import BaseContext

from phoenix.auth import compute_password_hash
from phoenix.db import models
from phoenix.server.access import (
    AccessScope,
    Permission,
    access_predicate,
    accessible_scope,
    permissions_for_user_id,
)
from phoenix.server.api.dataloaders import CacheForDataLoaders, DataLoaders, build_data_loaders
from phoenix.server.bearer_auth import PhoenixSystemUser, PhoenixUser
from phoenix.server.types import UserId

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql.elements import ColumnElement

    from phoenix.server.daemons.experiment_runner import ExperimentRunner
    from phoenix.server.daemons.span_cost_calculator import SpanCostCalculator
    from phoenix.server.daemons.system_settings import SystemSettings
    from phoenix.server.dml_event import DmlEvent
    from phoenix.server.email.types import EmailSender
    from phoenix.server.sandbox.session_manager import SandboxSessionManager
    from phoenix.server.types import (
        CanGetLastUpdatedAt,
        CanPutItem,
        DbSessionFactory,
        TokenStore,
    )


class _NoOp:
    def get(self, *args: Any, **kwargs: Any) -> Any: ...
    def put(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass
class Context(BaseContext):
    db: DbSessionFactory
    data_loaders: DataLoaders
    settings: SystemSettings
    cache_for_dataloaders: Optional[CacheForDataLoaders]
    span_cost_calculator: SpanCostCalculator
    experiment_runner: ExperimentRunner
    sandbox_session_manager: SandboxSessionManager
    encrypt: Callable[[bytes], bytes]
    decrypt: Callable[[bytes], bytes]
    last_updated_at: CanGetLastUpdatedAt = _NoOp()
    event_queue: CanPutItem[DmlEvent] = _NoOp()
    allowed_provider_names: Optional[frozenset[str]] = None
    read_only: bool = False
    locked: bool = False
    auth_enabled: bool = False
    access_control_enabled: bool = False
    secret: Optional[SecretStr] = None
    token_store: Optional[TokenStore] = None
    email_sender: Optional[EmailSender] = None
    _permissions_cache: Optional[frozenset[Permission]] = field(
        default=None, init=False, repr=False, compare=False
    )
    _access_scope_cache: dict[str, AccessScope] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def get_secret(self) -> SecretStr:
        """A type-safe way to get the application secret. Throws an error if the secret is not set.

        Returns:
            str: the phoenix secret
        """
        if self.secret is None:
            raise ValueError(
                "Application secret not set."
                " Please set the PHOENIX_SECRET environment variable and re-deploy the application."
            )
        return self.secret

    def get_request(self) -> StarletteRequest:
        """
        A type-safe way to get the request object. Throws an error if the request is not set.
        """
        if not isinstance(request := self.request, StarletteRequest):
            raise ValueError("no request is set")
        return request

    def get_response(self) -> StarletteResponse:
        """
        A type-safe way to get the response object. Throws an error if the response is not set.
        """
        if (response := self.response) is None:
            raise ValueError("no response is set")
        return response

    async def is_valid_password(self, password: SecretStr, user: models.User) -> bool:
        return (
            (hash_ := user.password_hash) is not None
            and (salt := user.password_salt) is not None
            and hash_ == await self.hash_password(password, salt)
        )

    @staticmethod
    async def hash_password(password: SecretStr, salt: bytes) -> bytes:
        compute = partial(compute_password_hash, password=password, salt=salt)
        return await get_running_loop().run_in_executor(None, compute)

    async def log_out(self, user_id: int) -> None:
        assert self.token_store is not None
        await self.token_store.log_out(UserId(user_id))

    @cached_property
    def user(self) -> PhoenixUser:
        return cast(PhoenixUser, self.get_request().user)

    @cached_property
    def user_id(self) -> Optional[int]:
        try:
            return int(self.user.identity)
        except Exception:
            return None

    async def actor_permissions(self) -> frozenset[Permission]:
        """The current actor's permissions, resolved live from the database and
        cached for this request. Reading from the database (rather than the role
        carried in the token) is what makes a role-permission edit take effect on
        the next request, and makes a deleted user's API keys go dead."""
        if self._permissions_cache is None:
            self._permissions_cache = await self._resolve_permissions()
        return self._permissions_cache

    async def _resolve_permissions(self) -> frozenset[Permission]:
        user = self.user
        # The admin-secret system actor always holds every permission, independent
        # of any database row.
        if isinstance(user, PhoenixSystemUser):
            return frozenset(Permission)
        if (user_id := self.user_id) is None:
            return frozenset()
        async with self.db() as session:
            return await permissions_for_user_id(session, user_id)

    async def access_scope(self, session: "AsyncSession", object_type: str) -> AccessScope:
        """Which objects of ``object_type`` the current actor may access, for use
        as a list filter (``scope.apply(column)``) or a point check
        (``scope.allows(id)``). When access control is disabled, the scope is
        everything — so resolvers can apply it unconditionally with no behavior
        change. Cached per object type for the request."""
        if object_type not in self._access_scope_cache:
            user_id = self.user_id
            if not self.access_control_enabled or user_id is None:
                self._access_scope_cache[object_type] = AccessScope(True, True, frozenset())
            else:
                self._access_scope_cache[object_type] = await accessible_scope(
                    session, user_id=user_id, object_type=object_type, enabled=True
                )
        return self._access_scope_cache[object_type]

    async def access_filter(
        self,
        session: "AsyncSession",
        object_type: str,
        id_column: Union["ColumnElement[int]", InstrumentedAttribute[int]],
    ) -> "ColumnElement[bool]":
        """SQL predicate for list/count queries.

        Point checks use :meth:`access_scope` so resolvers can call ``allows(id)``; list
        queries should use this method to keep access expansion inside SQL.
        """
        user_id = self.user_id
        if not self.access_control_enabled or user_id is None:
            return true()
        return await access_predicate(
            session,
            user_id=user_id,
            object_type=object_type,
            id_column=id_column,
            enabled=True,
        )


def build_context(
    *,
    db: DbSessionFactory,
    settings: SystemSettings,
    span_cost_calculator: SpanCostCalculator,
    experiment_runner: ExperimentRunner,
    sandbox_session_manager: SandboxSessionManager,
    encrypt: Callable[[bytes], bytes],
    decrypt: Callable[[bytes], bytes],
    cache_for_dataloaders: Optional[CacheForDataLoaders] = None,
    last_updated_at: CanGetLastUpdatedAt = _NoOp(),
    event_queue: CanPutItem[DmlEvent] = _NoOp(),
    allowed_provider_names: Optional[frozenset[str]] = None,
    read_only: bool = False,
    auth_enabled: bool = False,
    access_control_enabled: bool = False,
    secret: Optional[SecretStr] = None,
    token_store: Optional[TokenStore] = None,
    email_sender: Optional[EmailSender] = None,
    request: Optional[StarletteRequest] = None,
) -> Context:
    """Build a GraphQL Context object."""

    context = Context(
        db=db,
        settings=settings,
        data_loaders=build_data_loaders(db, cache_for_dataloaders),
        cache_for_dataloaders=cache_for_dataloaders,
        span_cost_calculator=span_cost_calculator,
        experiment_runner=experiment_runner,
        sandbox_session_manager=sandbox_session_manager,
        encrypt=encrypt,
        decrypt=decrypt,
        last_updated_at=last_updated_at,
        event_queue=event_queue,
        allowed_provider_names=allowed_provider_names,
        read_only=read_only,
        auth_enabled=auth_enabled,
        access_control_enabled=access_control_enabled,
        secret=secret,
        token_store=token_store,
        email_sender=email_sender,
    )
    if request is not None:
        context.request = request
    return context
