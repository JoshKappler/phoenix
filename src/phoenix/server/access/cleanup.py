from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from phoenix.db import models


async def delete_object_grants(session: AsyncSession, object_type: str, object_id: int) -> None:
    """Remove grants that name one concrete object.

    ACL object references are polymorphic (object_type, object_id), so the database
    cannot enforce an ON DELETE CASCADE to projects/datasets/prompts. Type-wide grants
    intentionally remain: deleting one project must not remove "all projects" access.
    """
    await session.execute(
        delete(models.AccessGrant).where(
            models.AccessGrant.object_type == object_type,
            models.AccessGrant.object_id == object_id,
            models.AccessGrant.selector_kind == "ids",
        )
    )
