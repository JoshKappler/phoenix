from secrets import token_bytes, token_hex

import pytest
from strawberry.relay import GlobalID

from phoenix.db import models
from phoenix.db.types.identifier import Identifier
from phoenix.server.access import OBJECT_TYPE_DATASET, OBJECT_TYPE_PROJECT, OBJECT_TYPE_PROMPT
from phoenix.server.api.exceptions import NotFound
from phoenix.server.api.mutations.access_grant_mutations import (
    AccessGrantObjectInput,
    AccessGrantSubjectInput,
    _resolve_permission_set_id,
    _resolve_object_rowid,
    _resolve_subject_rowid,
)
from phoenix.server.api.types.AccessSubjectKind import AccessSubjectKind
from phoenix.server.types import DbSessionFactory


class TestResolveSubjectRowid:
    async def test_accepts_existing_user_and_group(self, db: DbSessionFactory) -> None:
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
            group = models.UserGroup(
                provider="test",
                group_key=f"team-{token_hex(4)}",
                display_name="Team",
            )
            session.add_all([user, group])
            await session.flush()

            assert await _resolve_subject_rowid(
                session,
                AccessGrantSubjectInput(user_id=GlobalID("User", str(user.id))),
            ) == (AccessSubjectKind.USER, user.id)
            assert await _resolve_subject_rowid(
                session,
                AccessGrantSubjectInput(user_group_id=GlobalID("UserGroup", str(group.id))),
            ) == (AccessSubjectKind.GROUP, group.id)
            assert await _resolve_subject_rowid(
                session,
                AccessGrantSubjectInput(is_everyone=True),
            ) == (AccessSubjectKind.EVERYONE, None)

    async def test_rejects_missing_subject(self, db: DbSessionFactory) -> None:
        async with db() as session:
            with pytest.raises(NotFound, match="Unknown user"):
                await _resolve_subject_rowid(
                    session,
                    AccessGrantSubjectInput(user_id=GlobalID("User", "999999")),
                )
            with pytest.raises(NotFound, match="Unknown group"):
                await _resolve_subject_rowid(
                    session,
                    AccessGrantSubjectInput(user_group_id=GlobalID("UserGroup", "999999")),
                )


class TestResolveObjectRowid:
    async def test_accepts_existing_project_dataset_and_prompt(self, db: DbSessionFactory) -> None:
        async with db() as session:
            project = models.Project(name=f"project-{token_hex(4)}")
            dataset = models.Dataset(name=f"dataset-{token_hex(4)}")
            prompt = models.Prompt(name=Identifier(f"prompt-{token_hex(4)}"))
            session.add_all([project, dataset, prompt])
            await session.flush()

            assert await _resolve_object_rowid(
                session,
                AccessGrantObjectInput(dataset_id=GlobalID("Dataset", str(dataset.id))),
            ) == (OBJECT_TYPE_DATASET, dataset.id)
            assert await _resolve_object_rowid(
                session,
                AccessGrantObjectInput(project_id=GlobalID("Project", str(project.id))),
            ) == (OBJECT_TYPE_PROJECT, project.id)
            assert await _resolve_object_rowid(
                session,
                AccessGrantObjectInput(prompt_id=GlobalID("Prompt", str(prompt.id))),
            ) == (OBJECT_TYPE_PROMPT, prompt.id)

    async def test_rejects_missing_object(self, db: DbSessionFactory) -> None:
        async with db() as session:
            with pytest.raises(NotFound, match="Unknown dataset"):
                await _resolve_object_rowid(
                    session,
                    AccessGrantObjectInput(dataset_id=GlobalID("Dataset", "999999")),
                )
            with pytest.raises(NotFound, match="Unknown project"):
                await _resolve_object_rowid(
                    session,
                    AccessGrantObjectInput(project_id=GlobalID("Project", "999999")),
                )
            with pytest.raises(NotFound, match="Unknown prompt"):
                await _resolve_object_rowid(
                    session,
                    AccessGrantObjectInput(prompt_id=GlobalID("Prompt", "999999")),
                )


class TestResolvePermissionSetId:
    async def test_accepts_existing_permission_set_global_id(self, db: DbSessionFactory) -> None:
        async with db() as session:
            role = models.PermissionSet(name=f"Role-{token_hex(4)}", is_built_in=False)
            session.add(role)
            await session.flush()

            assert (
                await _resolve_permission_set_id(
                    session,
                    GlobalID("PermissionSet", str(role.id)),
                )
                == role.id
            )

    async def test_rejects_missing_permission_set_global_id(self, db: DbSessionFactory) -> None:
        async with db() as session:
            with pytest.raises(NotFound, match="Unknown permission set"):
                await _resolve_permission_set_id(
                    session,
                    GlobalID("PermissionSet", "999999"),
                )
