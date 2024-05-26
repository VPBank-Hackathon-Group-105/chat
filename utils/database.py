import os
import pandas as pd
import boto3

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
from sqlalchemy import insert, update, select, and_

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

def insert_new_cv_user(db, s3_file_path):
    try:
        query = (
            insert(user_cv)
            .values(
                {
                    "cv_file_path": s3_file_path,
                }
            )
            .returning(user_cv)
        )
        result = fetch_one(db.execute(query))
        db.commit()
    except:
        db.rollback()
        return None

    return result

def update_cv_user(db, entities, cv_user_id):
    try:
        query = (
            update(user_cv)
            .values({
                "name": entities.get("name", None),
                "summary": entities.get("summary", None),
                "year_of_birth": entities.get("year_of_birth", None),
                "skills": entities.get("skills", None),
                "experiences": entities.get("experiences", None),
                "year_of_experience": entities.get("year_of_experience", None),
                "educations": entities.get("educations", None),
                "awards": entities.get("awards", None),
                "qualifications": entities.get("qualifications", None),
            })
            .where(user_cv.c.id == cv_user_id)
            .returning(user_cv)
        )
        result = fetch_one(db.execute(query))
        db.commit()
    except:
        db.rollback()
        return None

    return result

def get_cv_users(per_page: int, page: int):
    engine = create_engine(
        url=READER_DATABASE_URL,
        echo=False,
        executemany_mode="values_plus_batch",
    )

    query = (
        select(user_cv)
        .with_only_columns(
            user_cv.c.id,
            user_cv.c.name,
            user_cv.c.summary,
            user_cv.c.year_of_birth,
            user_cv.c.skills,
            user_cv.c.experiences,
            user_cv.c.year_of_experience,
            user_cv.c.educations,
            user_cv.c.awards,
            user_cv.c.qualifications,
            user_cv.c.cv_file_path,
        )
    )
    query = query.order_by(user_cv.c.updated_at.desc())
    query = query.limit(per_page).offset((page - 1) * per_page)

    df = pd.read_sql(sql=query, con=engine)

    s3_client = boto3.client('s3')

    # Just specify folder_name:
    def get_presigned_url(file_path):
        url = s3_client.generate_presigned_url(
            ClientMethod='get_object',
            Params={'Bucket': os.environ.get("BUCKET_NAME"), 'Key': file_path},
            ExpiresIn=60,
        )

        return url

    df["cv_file_path"] = df["cv_file_path"].apply(get_presigned_url)

    return df