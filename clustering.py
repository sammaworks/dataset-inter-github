from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from sqlalchemy import Integer, String, DateTime, Index, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database.database import Base


class RepoInfo(Base):
    __tablename__ = "repo_info"

    # PK
    repo_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True, index=True
    )

    # Business keys
    repo_url: Mapped[str] = mapped_column(String, nullable=False)
    local_path: Mapped[str] = mapped_column(String, nullable=False)

    # Other columns
    latest_repo_snapshot: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("repo_url", "local_path", name="repo_info_url_path_uc"),
        Index("ix_repo_info_url", "repo_url"),
        Index("ix_repo_info_local_path", "local_path"),
    )

    # Relationships
    sessions: Mapped[List["Session"]] = relationship(
        back_populates="repo", cascade="all, delete-orphan", passive_deletes=True
    )
    histories: Mapped[List["RepoHistory"]] = relationship(
        back_populates="repo", cascade="all, delete-orphan", passive_deletes=True
    )






from __future__ import annotations

from datetime import datetime
from typing import List

from sqlalchemy import Integer, DateTime, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database.database import Base


class Session(Base):
    __tablename__ = "sessions"

    # PK
    session_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True, index=True
    )

    # FK → repo_info
    repo_id: Mapped[int] = mapped_column(
        ForeignKey("repo_info.repo_id", ondelete="CASCADE"), index=True, nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    repo: Mapped["RepoInfo"] = relationship(back_populates="sessions")

    commands: Mapped[List["CommandHistory"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", passive_deletes=True
    )






from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    Integer,
    String,
    DateTime,
    PrimaryKeyConstraint,
    ForeignKey,
    Index,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database.database import Base


class RepoHistory(Base):
    __tablename__ = "repo_history"

    # Composite PK (repo_id, commit_hash)
    repo_id: Mapped[int] = mapped_column(
        ForeignKey("repo_info.repo_id", ondelete="CASCADE"), index=True, nullable=False
    )
    commit_hash: Mapped[str] = mapped_column(String, nullable=False)

    repo_snapshot: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        PrimaryKeyConstraint("repo_id", "commit_hash", name="repo_history_pk"),
        Index("ix_repo_history_commit", "commit_hash"),
    )

    # Relationships
    repo: Mapped["RepoInfo"] = relationship(back_populates="histories")

    commands: Mapped[List["CommandHistory"]] = relationship(
        back_populates="repo_history", cascade="all, delete-orphan", passive_deletes=True
    )








from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Integer,
    String,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database.database import Base


class CommandHistory(Base):
    __tablename__ = "command_history"

    # PK
    command_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True, index=True
    )

    # FK → sessions
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.session_id", ondelete="CASCADE"), index=True, nullable=False
    )

    # Command data
    command: Mapped[str] = mapped_column(String, nullable=False)
    output: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Composite FK → repo_history (repo_id, commit_hash)
    repo_id: Mapped[int] = mapped_column(nullable=False)
    commit_hash: Mapped[str] = mapped_column(String, nullable=False)

    executed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["repo_id", "commit_hash"],
            ["repo_history.repo_id", "repo_history.commit_hash"],
            ondelete="CASCADE",
            name="fk_command_repo_history",
        ),
        Index("ix_command_history_session_id", "session_id"),
        Index("ix_command_history_repo_commit", "repo_id", "commit_hash"),
    )

    # Relationships
    session: Mapped["Session"] = relationship(back_populates="commands")
    repo_history: Mapped["RepoHistory"] = relationship(back_populates="commands")






