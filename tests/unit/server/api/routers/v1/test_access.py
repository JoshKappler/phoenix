from secrets import token_bytes, token_hex

import httpx
import pytest
from sqlalchemy import insert, select
from starlette.types import ASGIApp
from strawberry.relay import GlobalID

from phoenix.db import models
from phoenix.server.types import DbSessionFactory

_ADMIN_SECRET = "unit-test-admin-secret-0123456789abcdef0123456789"


@pytest.fixture(autouse=True)
def _enable_access_control(monkeypatch: pytest.MonkeyPatch) -> None:
    # Access control requires authentication (the boot guard refuses access-on + auth-off),
    # so these tests run an auth- and access-enabled app; the client below acts as an admin.
    monkeypatch.setenv("PHOENIX_ENABLE_AUTH", "true")
    monkeypatch.setenv("PHOENIX_ACCESS_CONTROL_ENABLED", "true")
    monkeypatch.setenv("PHOENIX_SECRET", "unit-test-phoenix-secret-0123456789abcdef0123456789")
    monkeypatch.setenv("PHOENIX_ADMIN_SECRET", _ADMIN_SECRET)


@pytest.fixture
def httpx_client(asgi_app: ASGIApp) -> httpx.AsyncClient:
    # Authenticate as an administrator (SYSTEM) via the admin-secret bearer token.
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=asgi_app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {_ADMIN_SECRET}"},
    )


def _gid(type_name: str, rowid: int) -> str:
    return str(GlobalID(type_name, str(rowid)))


def _rowid(gid: str) -> int:
    return int(GlobalID.from_id(gid).node_id)


async def _project(db: DbSessionFactory, name: str) -> int:
    async with db() as session:
        rowid = await session.scalar(
            insert(models.Project).values(name=name).returning(models.Project.id)
        )
    assert rowid is not None
    return rowid


async def _user(db: DbSessionFactory) -> int:
    async with db() as session:
        role = models.UserRole(name=f"r{token_hex(2)}")
        session.add(role)
        await session.flush()
        user = models.LocalUser(
            user_role_id=role.id,
            username=token_hex(8),
            email=f"{token_hex(8)}@x.test",
            reset_password=False,
            password_salt=token_bytes(32),
            password_hash=token_bytes(32),
        )
        session.add(user)
        await session.flush()
        return user.id


async def _group(db: DbSessionFactory, key: str) -> int:
    async with db() as session:
        gid = await session.scalar(
            insert(models.UserGroup)
            .values(provider="test", group_key=key, display_name=key)
            .returning(models.UserGroup.id)
        )
    assert gid is not None
    return gid


class TestAccessGrants:
    async def test_grant_lifecycle(
        self, httpx_client: httpx.AsyncClient, db: DbSessionFactory
    ) -> None:
        project_id = await _project(db, f"p-{token_hex(4)}")
        group_id = await _group(db, f"team-{token_hex(4)}")
        object_gid = _gid("Project", project_id)
        group_gid = _gid("UserGroup", group_id)

        # Author a grant: the group may read the project. All ids are GlobalIDs.
        resp = await httpx_client.post(
            "v1/access/grants",
            json={
                "subject": {"kind": "group", "id": group_gid},
                "object_type": "project",
                "object_id": object_gid,
            },
        )
        assert resp.status_code == 200, resp.text
        grant = resp.json()["data"]
        assert grant["subject"] == {"kind": "group", "id": group_gid}
        assert grant["object_id"] == object_gid
        grant_gid = grant["id"]
        assert _rowid(grant_gid)  # it's a real AccessGrant GlobalID

        # Re-granting is idempotent — same id.
        resp2 = await httpx_client.post(
            "v1/access/grants",
            json={
                "subject": {"kind": "group", "id": group_gid},
                "object_type": "project",
                "object_id": object_gid,
            },
        )
        assert resp2.json()["data"]["id"] == grant_gid

        # List, filtered to the object.
        resp = await httpx_client.get(
            "v1/access/grants", params={"object_type": "project", "object_id": object_gid}
        )
        assert [g["id"] for g in resp.json()["data"]] == [grant_gid]

        # The audit read: who can see the project?
        resp = await httpx_client.get(f"v1/access/objects/project/{object_gid}/subjects")
        assert resp.status_code == 200, resp.text
        assert {"kind": "group", "id": group_gid} in resp.json()["data"]

        # Revoke, then it's gone.
        resp = await httpx_client.delete(f"v1/access/grants/{grant_gid}")
        assert resp.status_code == 204, resp.text
        resp = await httpx_client.get(
            "v1/access/grants", params={"object_type": "project", "object_id": object_gid}
        )
        assert resp.json()["data"] == []
        async with db() as session:
            assert (
                await session.scalar(
                    select(models.AccessGrant.id).where(models.AccessGrant.id == _rowid(grant_gid))
                )
            ) is None

    async def test_everyone_grant_makes_object_public(
        self, httpx_client: httpx.AsyncClient, db: DbSessionFactory
    ) -> None:
        project_id = await _project(db, f"p-{token_hex(4)}")
        resp = await httpx_client.post(
            "v1/access/grants",
            json={
                "subject": {"kind": "everyone"},
                "object_type": "project",
                "object_id": _gid("Project", project_id),
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["subject"] == {"kind": "everyone", "id": None}

    async def test_validation(self, httpx_client: httpx.AsyncClient, db: DbSessionFactory) -> None:
        project_id = await _project(db, f"p-{token_hex(4)}")
        object_gid = _gid("Project", project_id)
        a_user = _gid("User", 1)

        # everyone must not carry an id.
        resp = await httpx_client.post(
            "v1/access/grants",
            json={"subject": {"kind": "everyone", "id": a_user}, "object_type": "project"},
        )
        assert resp.status_code == 422, resp.text

        # a user subject must carry an id.
        resp = await httpx_client.post(
            "v1/access/grants",
            json={"subject": {"kind": "user"}, "object_type": "project", "object_id": object_gid},
        )
        assert resp.status_code == 422, resp.text

        # a subject id of the wrong type is rejected (a Project GlobalID where a User is expected).
        resp = await httpx_client.post(
            "v1/access/grants",
            json={
                "subject": {"kind": "user", "id": object_gid},
                "object_type": "project",
                "object_id": object_gid,
            },
        )
        assert resp.status_code == 422, resp.text

        # an ungrantable object type is rejected by the enum.
        resp = await httpx_client.post(
            "v1/access/grants",
            json={"subject": {"kind": "user", "id": a_user}, "object_type": "span"},
        )
        assert resp.status_code == 422, resp.text

        # a grant on a nonexistent object is not-found.
        resp = await httpx_client.post(
            "v1/access/grants",
            json={
                "subject": {"kind": "user", "id": a_user},
                "object_type": "project",
                "object_id": _gid("Project", 999999),
            },
        )
        assert resp.status_code == 404, resp.text

    async def test_rejects_unresolvable_subjects_and_unknown_roles(
        self, httpx_client: httpx.AsyncClient, db: DbSessionFactory
    ) -> None:
        project_id = await _project(db, f"p-{token_hex(4)}")
        object_gid = _gid("Project", project_id)
        user_id = await _user(db)

        # A subject whose GlobalID is well-formed but references no row is not-found: the id
        # shape alone does not prove the user exists.
        resp = await httpx_client.post(
            "v1/access/grants",
            json={
                "subject": {"kind": "user", "id": _gid("User", 999999)},
                "object_type": "project",
                "object_id": object_gid,
            },
        )
        assert resp.status_code == 404, resp.text

        # service_account subjects are reserved but have no identity lifecycle yet, so a grant
        # naming one is refused rather than stored against a subject that can never exist.
        resp = await httpx_client.post(
            "v1/access/grants",
            json={
                "subject": {"kind": "service_account", "id": _gid("ServiceAccount", 1)},
                "object_type": "project",
                "object_id": object_gid,
            },
        )
        assert resp.status_code == 422, resp.text

        # A named permission set that does not exist is a client error.
        resp = await httpx_client.post(
            "v1/access/grants",
            json={
                "subject": {"kind": "user", "id": _gid("User", user_id)},
                "object_type": "project",
                "object_id": object_gid,
                "role": "No Such Role",
            },
        )
        assert resp.status_code == 422, resp.text

        # An existing subject with the view-only default (role omitted) succeeds.
        resp = await httpx_client.post(
            "v1/access/grants",
            json={
                "subject": {"kind": "user", "id": _gid("User", user_id)},
                "object_type": "project",
                "object_id": object_gid,
            },
        )
        assert resp.status_code == 200, resp.text

    async def test_permission_sets_and_enforcement(
        self, httpx_client: httpx.AsyncClient, db: DbSessionFactory
    ) -> None:
        async with db() as session:
            role_id = await session.scalar(
                insert(models.PermissionSet)
                .values(name=f"Role-{token_hex(4)}", is_built_in=True)
                .returning(models.PermissionSet.id)
            )
            assert role_id is not None
            session.add(models.PermissionSetItem(permission_set_id=role_id, permission="obj_view"))

        resp = await httpx_client.get("v1/access/object-roles")
        assert resp.status_code == 200, resp.text
        mine = [r for r in resp.json()["data"] if r["id"] == _gid("PermissionSet", role_id)]
        assert mine and mine[0]["permissions"] == ["obj_view"]

        # These tests run an access-control-enabled app (see the module fixture), so the
        # DB latch is on and enforcement reports enabled.
        resp = await httpx_client.get("v1/access/enforcement")
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"] == {"enabled": True, "source": "db-latch"}


class TestLocalGroups:
    async def test_group_and_member_lifecycle(
        self, httpx_client: httpx.AsyncClient, db: DbSessionFactory
    ) -> None:
        resp = await httpx_client.post("v1/access/groups", json={"name": f"team-{token_hex(4)}"})
        assert resp.status_code == 200, resp.text
        group = resp.json()["data"]
        assert group["member_user_ids"] == []
        group_gid = group["id"]
        assert _rowid(group_gid)  # a real UserGroup GlobalID

        user_gid = _gid("User", await _user(db))
        resp = await httpx_client.post(
            f"v1/access/groups/{group_gid}/members", json={"user_id": user_gid}
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["member_user_ids"] == [user_gid]

        # Adding the same member again is idempotent.
        resp = await httpx_client.post(
            f"v1/access/groups/{group_gid}/members", json={"user_id": user_gid}
        )
        assert resp.json()["data"]["member_user_ids"] == [user_gid]

        resp = await httpx_client.get("v1/access/groups")
        assert group_gid in [g["id"] for g in resp.json()["data"]]

        resp = await httpx_client.delete(f"v1/access/groups/{group_gid}/members/{user_gid}")
        assert resp.status_code == 204, resp.text
        resp = await httpx_client.get("v1/access/groups")
        mine = next(g for g in resp.json()["data"] if g["id"] == group_gid)
        assert mine["member_user_ids"] == []

    async def test_delete_group_sweeps_its_grants(
        self, httpx_client: httpx.AsyncClient, db: DbSessionFactory
    ) -> None:
        resp = await httpx_client.post("v1/access/groups", json={"name": f"team-{token_hex(4)}"})
        group_gid = resp.json()["data"]["id"]
        project_id = await _project(db, f"p-{token_hex(4)}")
        resp = await httpx_client.post(
            "v1/access/grants",
            json={
                "subject": {"kind": "group", "id": group_gid},
                "object_type": "project",
                "object_id": _gid("Project", project_id),
            },
        )
        assert resp.status_code == 200, resp.text
        grant_gid = resp.json()["data"]["id"]
        user_gid = _gid("User", await _user(db))
        await httpx_client.post(f"v1/access/groups/{group_gid}/members", json={"user_id": user_gid})

        resp = await httpx_client.delete(f"v1/access/groups/{group_gid}")
        assert resp.status_code == 204, resp.text

        group_rowid = _rowid(group_gid)
        async with db() as session:
            # The group's grant was swept on delete (no FK on acls.subject_id, so it
            # cannot cascade) — the grant row is gone ...
            assert (
                await session.scalar(
                    select(models.AccessGrant.id).where(models.AccessGrant.id == _rowid(grant_gid))
                )
            ) is None
            # ... and the group + its memberships are gone.
            assert (
                await session.scalar(
                    select(models.UserGroup.id).where(models.UserGroup.id == group_rowid)
                )
            ) is None
            assert (
                await session.scalar(
                    select(models.UserGroupMembership.user_group_id).where(
                        models.UserGroupMembership.user_group_id == group_rowid
                    )
                )
            ) is None

    async def test_validation_and_idp_groups_untouchable(
        self, httpx_client: httpx.AsyncClient, db: DbSessionFactory
    ) -> None:
        name = f"team-{token_hex(4)}"
        assert (await httpx_client.post("v1/access/groups", json={"name": name})).status_code == 200
        # duplicate name → 409
        dup = await httpx_client.post("v1/access/groups", json={"name": name})
        assert dup.status_code == 409, dup.text
        # empty name → 422
        empty = await httpx_client.post("v1/access/groups", json={"name": "  "})
        assert empty.status_code == 422, empty.text

        # An IdP-synced (non-local) group is neither listed nor mutable here.
        idp_gid = _gid("UserGroup", await _group(db, f"k-{token_hex(4)}"))  # provider="test"
        assert (await httpx_client.delete(f"v1/access/groups/{idp_gid}")).status_code == 404
        r = await httpx_client.post(
            f"v1/access/groups/{idp_gid}/members", json={"user_id": _gid("User", 1)}
        )
        assert r.status_code == 404, r.text
        resp = await httpx_client.get("v1/access/groups")
        assert idp_gid not in [g["id"] for g in resp.json()["data"]]
