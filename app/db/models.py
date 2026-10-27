from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
    )
    name: Mapped[str] = mapped_column(
        String(100),
        unique=True,
    )
    description: Mapped[str] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    __table_args__ = (
        UniqueConstraint("dataset_id", "version"),
        CheckConstraint("version > 0", name="positive_version"),
    )

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
    )
    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("datasets.id"),
    )
    version: Mapped[int] = mapped_column(Integer)

    content_hash: Mapped[str] = mapped_column(String(64))
    manifest: Mapped[dict[str, Any]] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class TaskVersion(Base):
    __tablename__ = "task_versions"

    __table_args__ = (
        UniqueConstraint("dataset_version_id", "task_key"),
        CheckConstraint("version > 0", name="positive_version"),
        CheckConstraint(
            "split IN ('dev', 'validation', 'test')",
            name="valid_split",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
    )
    dataset_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("dataset_versions.id"),
    )

    task_key: Mapped[str] = mapped_column(String(100))
    version: Mapped[int] = mapped_column(Integer)
    category: Mapped[str] = mapped_column(String(50))
    scenario_family: Mapped[str] = mapped_column(String(100))
    split: Mapped[str] = mapped_column(String(20))

    agent_input: Mapped[dict[str, Any]] = mapped_column(JSONB)
    environment_fixture: Mapped[dict[str, Any]] = mapped_column(JSONB)
    evaluation_spec: Mapped[dict[str, Any]] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )