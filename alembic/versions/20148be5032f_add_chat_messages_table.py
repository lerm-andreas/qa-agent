"""add chat_messages table

Revision ID: 20148be5032f
Revises: bdd6591e7f9d
Create Date: 2026-06-21 21:10:23.186849

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20148be5032f'
down_revision: Union[str, Sequence[str], None] = 'bdd6591e7f9d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # NOTE: autogenerate also emitted op.drop_index('ix_chunks_embedding_hnsw', ...)
    # here — removed manually. That HNSW index was created via raw SQL in
    # bdd6591e7f9d and isn't declared in the model's __table_args__, so autogenerate
    # can't see it and assumes it should be dropped. Leave it alone.
    op.create_table('chat_messages',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('session_id', sa.String(length=64), nullable=False),
    sa.Column('role', sa.String(length=16), nullable=False),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_chat_messages_session_id'), 'chat_messages', ['session_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    # the matching op.create_index('ix_chunks_embedding_hnsw', ...) was removed too
    op.drop_index(op.f('ix_chat_messages_session_id'), table_name='chat_messages')
    op.drop_table('chat_messages')
