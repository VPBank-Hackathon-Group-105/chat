import os

from typing import Any

from dotenv import load_dotenv
load_dotenv()


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

DATABASE_URL = os.environ.get("DATABASE_URL")

engine = create_engine(DATABASE_URL)
metadata = MetaData(naming_convention={
    "ix": "%(column_0_label)s_idx",
    "uq": "%(table_name)s_%(column_0_name)s_key",
    "ck": "%(table_name)s_%(constraint_name)s_check",
    "fk": "%(table_name)s_%(column_0_name)s_fkey",
    "pk": "%(table_name)s_pkey",
})

engine = create_engine(
    url=DATABASE_URL,
    echo=False,
    executemany_mode="values_plus_batch",
    executemany_values_page_size=10000,
    executemany_batch_page_size=500,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def fetch_one(cursor: Any) -> dict[str, Any] | None:
    return cursor.fetchone()._asdict() if cursor.rowcount > 0 else None


def fetch_all(cursor: Any) -> list[dict[str, Any]]:
    return [r._asdict() for r in cursor.fetchall()]


Base = declarative_base()

user_cv = Table(
    "user_cv",
    metadata,
    Column("id", Integer, Identity(), primary_key=True),
    Column("name", String, nullable=True),
    Column("YOB", Integer, nullable=True),
    Column("cv_file_path", String, nullable=False),
    Column("skills", String, nullable=True),
    Column("experience", String, nullable=True),
    Column("education", String, nullable=True),
    Column("award/qualification", String, nullable=True),
)
