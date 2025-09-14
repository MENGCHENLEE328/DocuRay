"""Shared dataclasses for candidates and anchors."""  # Author: Team DocuRay | Generated: TDD impl | Version: 0.1.0 | Modified: 2025-09-14

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


@dataclass(slots=True)
class SourceACL:
    """Minimal ACL summary for replay and enforcement."""
    tenant_id: str
    principal_id: str
    scopes: List[str]


@dataclass(slots=True)
class Anchor:
    """Anchor to locate evidence inside a source."""
    page: Optional[int] = None
    line: Optional[int] = None
    byte_range: Optional[Tuple[int, int]] = None
    table_cell: Optional[Tuple[int, int]] = None  # row, col
    symbol: Optional[str] = None  # code symbol


@dataclass(slots=True)
class Fragment:
    """A content chunk with anchors and score."""
    doc_id: str
    chunk_id: str
    score: float
    channel: str
    anchors: List[Anchor] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class CandidateResult:
    """Document-level candidate with aggregated evidence."""
    doc_id: str
    fused_score: float
    fragments: List[Fragment] = field(default_factory=list)
    acl: Optional[SourceACL] = None
    source: Optional[str] = None

