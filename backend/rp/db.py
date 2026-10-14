"""Relational schema. Postgres + pgvector; uniqueness constraints carry the idempotency guarantees."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine, text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from . import config

J = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Topic(Base):
    __tablename__ = "topics"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    # Each concept may list synonyms with "|", e.g. "quantization|quantisation|4-bit".
    concepts: Mapped[list] = mapped_column(J, default=list)
    exclusions: Mapped[list] = mapped_column(J, default=list)
    source_ids: Mapped[list] = mapped_column(J, default=list)
    relevance_threshold: Mapped[float] = mapped_column(Float, default=0.45)
    alert_threshold: Mapped[str] = mapped_column(String(10), default="MEDIUM")
    poll_minutes: Mapped[int] = mapped_column(Integer, default=0)  # 0 = manual only
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    embedding = mapped_column(Vector(config.EMBED_DIM), nullable=True)
    profile_version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Source(Base):
    __tablename__ = "sources"
    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(20))  # fixture | arxiv | openalex | rss
    name: Mapped[str] = mapped_column(String(200))
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    query: Mapped[str | None] = mapped_column(Text, nullable=True)
    role_default: Mapped[str] = mapped_column(String(12), default="UNKNOWN")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    cursor: Mapped[dict] = mapped_column(J, default=dict)
    last_checked: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_success: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    items_last_run: Mapped[int | None] = mapped_column(Integer, nullable=True)
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0)


class Document(Base):
    """A canonical work (one paper, one article), resolved by DOI / arXiv id / canonical URL."""
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(primary_key=True)
    canonical_key: Mapped[str] = mapped_column(String(500), unique=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))
    external_id: Mapped[str | None] = mapped_column(String(300), nullable=True)
    url: Mapped[str] = mapped_column(Text)
    doi: Mapped[str | None] = mapped_column(String(200), nullable=True)
    arxiv_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    title: Mapped[str] = mapped_column(Text)
    publisher: Mapped[str] = mapped_column(String(200), default="")
    authors: Mapped[list] = mapped_column(J, default=list)
    source_role: Mapped[str] = mapped_column(String(12), default="UNKNOWN")
    publication_type: Mapped[str] = mapped_column(String(20), default="unknown")
    cites_url: Mapped[str | None] = mapped_column(Text, nullable=True)  # the primary source a report is based on
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    current_version: Mapped[int] = mapped_column(Integer, default=1)
    synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    text_access_status: Mapped[str] = mapped_column(String(20), default="abstract")


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    __table_args__ = (UniqueConstraint("document_id", "content_hash"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(80))
    title: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    changed_fields: Mapped[list] = mapped_column(J, default=list)
    source_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    injection_flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    injection_excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding = mapped_column(Vector(config.EMBED_DIM), nullable=True)
    embed_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    run_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


class TopicDocument(Base):
    __tablename__ = "topic_documents"
    __table_args__ = (UniqueConstraint("topic_id", "document_id"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id", ondelete="CASCADE"))
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    relevance: Mapped[float] = mapped_column(Float)
    relevant: Mapped[bool] = mapped_column(Boolean)
    version_id: Mapped[int | None] = mapped_column(Integer, nullable=True)  # version the judgement was made on
    reason: Mapped[str] = mapped_column(Text, default="")
    concepts: Mapped[list] = mapped_column(J, default=list)


class Cluster(Base):
    """One underlying development, possibly reported by many documents."""
    __tablename__ = "clusters"
    id: Mapped[int] = mapped_column(primary_key=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(Text)
    entity: Mapped[str | None] = mapped_column(String(100), nullable=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    event_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ClusterDocument(Base):
    __tablename__ = "cluster_documents"
    __table_args__ = (UniqueConstraint("cluster_id", "document_id"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    cluster_id: Mapped[int] = mapped_column(ForeignKey("clusters.id", ondelete="CASCADE"))
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    relation: Mapped[str] = mapped_column(String(30))  # origin | duplicate_report
    reason: Mapped[str] = mapped_column(Text, default="")
    decided_by: Mapped[str] = mapped_column(String(20), default="rule")
    score: Mapped[float | None] = mapped_column(Float, nullable=True)


class Claim(Base):
    __tablename__ = "claims"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    version_id: Mapped[int] = mapped_column(ForeignKey("document_versions.id", ondelete="CASCADE"))
    entity: Mapped[str] = mapped_column(String(100))
    metric: Mapped[str] = mapped_column(String(100))
    metric_key: Mapped[str] = mapped_column(String(100))
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(20))
    context: Mapped[str] = mapped_column(String(100), default="")
    conditions: Mapped[dict] = mapped_column(J, default=dict)  # method, batch size, context length
    text: Mapped[str] = mapped_column(Text)
    span_start: Mapped[int] = mapped_column(Integer)
    span_end: Mapped[int] = mapped_column(Integer)
    extractor: Mapped[str] = mapped_column(String(30), default=config.EXTRACTOR_VERSION)


class Assessment(Base):
    __tablename__ = "assessments"
    id: Mapped[int] = mapped_column(primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(300), unique=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id", ondelete="CASCADE"))
    cluster_id: Mapped[int | None] = mapped_column(ForeignKey("clusters.id", ondelete="SET NULL"), nullable=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    version_id: Mapped[int] = mapped_column(ForeignKey("document_versions.id", ondelete="CASCADE"))
    run_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    classification: Mapped[str] = mapped_column(String(30))
    priority: Mapped[str] = mapped_column(String(10))
    confidence: Mapped[str] = mapped_column(String(10))
    headline: Mapped[str] = mapped_column(Text)
    what_changed: Mapped[str] = mapped_column(Text)
    why_it_matters: Mapped[str] = mapped_column(Text)
    narrator: Mapped[str] = mapped_column(String(20), default="template")
    scores: Mapped[dict] = mapped_column(J, default=dict)
    score_reasons: Mapped[list] = mapped_column(J, default=list)
    comparisons: Mapped[list] = mapped_column(J, default=list)
    uncertainty: Mapped[list] = mapped_column(J, default=list)
    evidence_gaps: Mapped[list] = mapped_column(J, default=list)
    flags: Mapped[list] = mapped_column(J, default=list)
    audit: Mapped[dict] = mapped_column(J, default=dict)
    read: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AssessmentEvidence(Base):
    __tablename__ = "assessment_evidence"
    id: Mapped[int] = mapped_column(primary_key=True)
    assessment_id: Mapped[int] = mapped_column(ForeignKey("assessments.id", ondelete="CASCADE"))
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    version_id: Mapped[int] = mapped_column(ForeignKey("document_versions.id", ondelete="CASCADE"))
    claim_id: Mapped[int | None] = mapped_column(ForeignKey("claims.id", ondelete="SET NULL"), nullable=True)
    role: Mapped[str] = mapped_column(String(20))  # supports | contradicts | previous | context
    excerpt: Mapped[str] = mapped_column(Text)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)


class Trend(Base):
    __tablename__ = "trends"
    __table_args__ = (UniqueConstraint("topic_id", "concept"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id", ondelete="CASCADE"))
    concept: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(30))
    window_days: Mapped[int] = mapped_column(Integer)
    current_count: Mapped[int] = mapped_column(Integer)
    baseline_count: Mapped[int] = mapped_column(Integer)
    baseline_windows: Mapped[int] = mapped_column(Integer)
    article_count: Mapped[int] = mapped_column(Integer, default=0)
    summary: Mapped[str] = mapped_column(Text)
    limitations: Mapped[list] = mapped_column(J, default=list)
    series: Mapped[list] = mapped_column(J, default=list)
    evidence: Mapped[list] = mapped_column(J, default=list)
    contradicting: Mapped[list] = mapped_column(J, default=list)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(300), unique=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id", ondelete="CASCADE"))
    assessment_id: Mapped[int] = mapped_column(ForeignKey("assessments.id", ondelete="CASCADE"))
    cluster_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    priority: Mapped[str] = mapped_column(String(10))
    headline: Mapped[str] = mapped_column(Text)
    read: Mapped[bool] = mapped_column(Boolean, default=False)
    suppressed_repeats: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Feedback(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(primary_key=True)
    assessment_id: Mapped[int] = mapped_column(ForeignKey("assessments.id", ondelete="CASCADE"))
    verdict: Mapped[str] = mapped_column(String(30))
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(20), default="QUEUED")
    phase: Mapped[str] = mapped_column(String(30), default="queued")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    counters: Mapped[dict] = mapped_column(J, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RunEvent(Base):
    __tablename__ = "run_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    agent: Mapped[str] = mapped_column(String(20))
    kind: Mapped[str] = mapped_column(String(20))  # phase | tool | note | summary
    message: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="done")
    ms: Mapped[int | None] = mapped_column(Integer, nullable=True)


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(30), default="manual")  # manual | scheduled
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id", ondelete="CASCADE"))
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(20), default="queued")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


engine = create_engine(config.DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


def init_db(drop: bool = False) -> None:
    with engine.begin() as c:
        c.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    if drop:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with engine.begin() as c:
        # Lexical side of hybrid retrieval: a stored tsvector + GIN index; HNSW for the dense side.
        c.execute(text(
            "ALTER TABLE document_versions ADD COLUMN IF NOT EXISTS tsv tsvector "
            "GENERATED ALWAYS AS (to_tsvector('english', coalesce(title,'') || ' ' || coalesce(text,''))) STORED"))
        c.execute(text("CREATE INDEX IF NOT EXISTS dv_tsv_idx ON document_versions USING gin (tsv)"))
        c.execute(text("CREATE INDEX IF NOT EXISTS dv_emb_idx ON document_versions USING hnsw (embedding vector_cosine_ops)"))


@contextmanager
def session() -> Session:
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
