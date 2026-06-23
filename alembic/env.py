from logging.config import fileConfig

from alembic import context

# --- project wiring ---
# load Base.metadata so autogenerate can compare models vs DB schema
from rag.database import Base, engine
import rag.models  # noqa: F401 — registers Document + DocumentChunk + ChatMessage on Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# autogenerate compares this metadata against the live DB schema
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Generate SQL script without a live DB connection."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live DB connection."""
    # reuse the engine from rag.database — no second engine needed
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
