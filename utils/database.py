import os

from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    Identity,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

READER_DATABASE_URL = os.environ.get("READER_DATABASE_URL")
WRITER_DATABASE_URL = os.environ.get("WRITER_DATABASE_URL")

metadata = MetaData(naming_convention={
    "ix": "%(column_0_label)s_idx",
    "uq": "%(table_name)s_%(column_0_name)s_key",
    "ck": "%(table_name)s_%(constraint_name)s_check",
    "fk": "%(table_name)s_%(column_0_name)s_fkey",
    "pk": "%(table_name)s_pkey",
})

def get_db(is_writer: bool = False):
    engine = create_engine(
        url=WRITER_DATABASE_URL if is_writer else READER_DATABASE_URL,
        echo=False,
        executemany_mode="values_plus_batch",
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    try:
        return session
    finally:
        session.close()


def fetch_one(cursor: Any) -> dict[str, Any] | None:
    return cursor.fetchone()._asdict() if cursor.rowcount > 0 else None


def fetch_all(cursor: Any) -> list[dict[str, Any]]:
    return [r._asdict() for r in cursor.fetchall()]


Base = declarative_base()

user_cv = Table(
    "user_cv",
    metadata,
    Column("id", Integer, Identity(), primary_key=True),
    Column("summary", String, nullable=True),
    Column("name", String, nullable=True),
    Column("year_of_birth", Integer, nullable=True),
    Column("skills", JSON, nullable=True),
    Column("experiences", JSON, nullable=True),
    Column("year_of_experience", Integer, nullable=True),
    Column("educations", JSON, nullable=True),
    Column("awards", JSON, nullable=True),
    Column("qualifications", JSON, nullable=True),
    Column("cv_file_path", String, nullable=False),
    Column("created_at", DateTime, server_default=func.now(), nullable=False),
    Column("updated_at", DateTime, server_default=func.now(), onupdate=func.now()),
)