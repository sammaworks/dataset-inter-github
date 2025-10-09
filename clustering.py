# models.py (SQLAlchemy 2.x style)
from __future__ import annotations
from typing import List, Optional
from sqlalchemy import (
    Integer, String, DateTime, Index, UniqueConstraint, PrimaryKeyConstraint,
    ForeignKey, ForeignKeyConstraint, func
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class RepoInfo(Base):
    __tablename__ = "repo_info"

    repo_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True)
    repo_url: Mapped[str] = mapped_column(String, nullable=False)
    local_path: Mapped[str] = mapped_column(String, nullable=False)

    latest_repo_snapshot: Mapped[Optional[str]] = mapped_column(String)
    created_at: Mapped = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("repo_url", "local_path", name="repo_info_url_path_uc"),
        Index("ix_repo_info_url", "repo_url"),
        Index("ix_repo_info_local_path", "local_path"),
    )

    sessions: Mapped[List["Session"]] = relationship(
        back_populates="repo", cascade="all, delete-orphan", passive_deletes=True
    )
    histories: Mapped[List["RepoHistory"]] = relationship(
        back_populates="repo", cascade="all, delete-orphan", passive_deletes=True
    )

class Session(Base):
    __tablename__ = "sessions"

    session_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True)
    repo_id: Mapped[int] = mapped_column(ForeignKey("repo_info.repo_id", ondelete="CASCADE"), index=True, nullable=False)

    created_at: Mapped = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    repo: Mapped["RepoInfo"] = relationship(back_populates="sessions")
    commands: Mapped[List["CommandHistory"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", passive_deletes=True
    )

class RepoHistory(Base):
    __tablename__ = "repo_history"

    repo_id: Mapped[int] = mapped_column(ForeignKey("repo_info.repo_id", ondelete="CASCADE"), index=True, nullable=False)
    commit_hash: Mapped[str] = mapped_column(String, nullable=False)

    repo_snapshot: Mapped[Optional[str]] = mapped_column(String)
    created_at: Mapped = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint("repo_id", "commit_hash", name="repo_history_pk"),
        Index("ix_repo_history_commit", "commit_hash"),
    )

    repo: Mapped["RepoInfo"] = relationship(back_populates="histories")
    commands: Mapped[List["CommandHistory"]] = relationship(
        back_populates="repo_history", cascade="all, delete-orphan", passive_deletes=True
    )

class CommandHistory(Base):
    __tablename__ = "command_history"

    command_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.session_id", ondelete="CASCADE"), index=True, nullable=False)

    command: Mapped[str] = mapped_column(String, nullable=False)
    output: Mapped[Optional[str]] = mapped_column(String)

    # exact repo state linkage
    repo_id: Mapped[int] = mapped_column(nullable=False)
    commit_hash: Mapped[str] = mapped_column(String, nullable=False)

    executed_at: Mapped = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

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

    session: Mapped["Session"] = relationship(back_populates="commands")
    repo_history: Mapped["RepoHistory"] = relationship(back_populates="commands")
