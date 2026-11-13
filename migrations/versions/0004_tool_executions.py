"""tool_executions: idempotency keys for write operations.

A write (create a ticket, update it, add a comment) can be sent with an idempotency key.
The key is recorded in the same transaction as the write, together with its result, so
running the same write again with the same key returns the saved result instead of writing
twice. The agent uses "<thread_id>:<tool_call_id>" as the key: if a run crashes after the
write but before LangGraph saved its checkpoint, resuming re-runs the tool step, and the
ticket is still created only once.

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "tool_executions",
        sa.Column("key", sa.String(200), primary_key=True),
        sa.Column("tool", sa.String(64), nullable=False),
        # Filled in before the transaction commits, so a committed row always has a result.
        sa.Column("result", JSONB),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )


def downgrade() -> None:
    op.drop_table("tool_executions")
