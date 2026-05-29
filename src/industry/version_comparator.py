"""Version comparison module for comparing document versions."""

import difflib
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ChangeType(str, Enum):
    """Types of changes between document versions."""

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


class SignificanceLevel(str, Enum):
    """Significance levels for changes."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Change(BaseModel):
    """A single change between document versions."""

    change_type: ChangeType = Field(..., description="Type of change")
    content: str = Field(..., description="Content of the change")
    old_content: Optional[str] = Field(
        None, description="Previous content (for modifications)"
    )
    location: Optional[str] = Field(
        None, description="Location or section of the change"
    )
    significance: SignificanceLevel = Field(
        default=SignificanceLevel.MEDIUM, description="Significance of this change"
    )


class ComparisonResult(BaseModel):
    """Result of comparing two document versions."""

    changes: List[Change] = Field(
        default_factory=list, description="All detected changes"
    )
    summary: str = Field(default="", description="Summary of changes")
    added_clauses: List[str] = Field(
        default_factory=list, description="Clauses added in new version"
    )
    removed_clauses: List[str] = Field(
        default_factory=list, description="Clauses removed in new version"
    )
    modified_clauses: List[str] = Field(
        default_factory=list, description="Clauses modified between versions"
    )
    significance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall significance score of changes",
    )
    total_additions: int = Field(default=0, description="Total lines added")
    total_removals: int = Field(default=0, description="Total lines removed")
    total_modifications: int = Field(default=0, description="Total modifications")


class VersionComparator:
    """Compares two versions of a document to identify changes.

    Provides text-level diffing, clause-level comparison, and
    significance assessment.
    """

    # Keywords that indicate high-significance changes
    _HIGH_SIGNIFICANCE_KEYWORDS = [
        "liability",
        "indemnify",
        "termination",
        "payment",
        "penalty",
        "breach",
        "damages",
        "warranty",
        "guarantee",
        "obligation",
        "confidential",
        "intellectual property",
        "non-compete",
    ]

    def compare(self, text_v1: str, text_v2: str) -> ComparisonResult:
        """Compare two document versions.

        Args:
            text_v1: Text of the first (older) version.
            text_v2: Text of the second (newer) version.

        Returns:
            ComparisonResult with detailed change information.
        """
        # Get text-level differences
        changes = self._diff_text(text_v1, text_v2)

        # Get clause-level differences
        clauses_v1 = self._extract_clauses(text_v1)
        clauses_v2 = self._extract_clauses(text_v2)
        added_clauses, removed_clauses, modified_clauses = self._diff_clauses(
            clauses_v1, clauses_v2
        )

        # Identify and categorize changes
        categorized = self._identify_changes(changes)

        # Assess significance
        significance_score = self._assess_significance(categorized)

        # Generate summary
        summary = self._generate_summary(categorized, added_clauses, removed_clauses)

        # Count totals
        total_additions = sum(
            1 for c in categorized if c.change_type == ChangeType.ADDED
        )
        total_removals = sum(
            1 for c in categorized if c.change_type == ChangeType.REMOVED
        )
        total_modifications = sum(
            1 for c in categorized if c.change_type == ChangeType.MODIFIED
        )

        return ComparisonResult(
            changes=categorized,
            summary=summary,
            added_clauses=added_clauses,
            removed_clauses=removed_clauses,
            modified_clauses=modified_clauses,
            significance_score=significance_score,
            total_additions=total_additions,
            total_removals=total_removals,
            total_modifications=total_modifications,
        )

    def _diff_text(self, text1: str, text2: str) -> List[tuple]:
        """Compute text-level differences using difflib.

        Args:
            text1: First text version.
            text2: Second text version.

        Returns:
            List of (tag, content) tuples from difflib.
        """
        lines1 = text1.splitlines(keepends=True)
        lines2 = text2.splitlines(keepends=True)

        differ = difflib.unified_diff(lines1, lines2, lineterm="")
        diffs = []

        for line in differ:
            if line.startswith("+++") or line.startswith("---"):
                continue
            elif line.startswith("@@"):
                continue
            elif line.startswith("+"):
                diffs.append(("added", line[1:].rstrip()))
            elif line.startswith("-"):
                diffs.append(("removed", line[1:].rstrip()))

        return diffs

    def _extract_clauses(self, text: str) -> List[str]:
        """Extract clause-like sections from text.

        Args:
            text: Document text.

        Returns:
            List of clause text strings.
        """
        import re

        # Split by numbered sections or headings
        pattern = r"(?:^|\n)(?:\d+\.?\s+|ARTICLE\s+\w+|Section\s+\d+)"
        parts = re.split(pattern, text, flags=re.IGNORECASE)

        clauses = []
        for part in parts:
            stripped = part.strip()
            if stripped and len(stripped) > 20:
                clauses.append(stripped)

        # If no structured clauses found, split by paragraphs
        if not clauses:
            paragraphs = text.split("\n\n")
            clauses = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 20]

        return clauses

    def _diff_clauses(
        self, clauses1: List[str], clauses2: List[str]
    ) -> tuple:
        """Compare clauses between two versions.

        Args:
            clauses1: Clauses from version 1.
            clauses2: Clauses from version 2.

        Returns:
            Tuple of (added_clauses, removed_clauses, modified_clauses).
        """
        added: List[str] = []
        removed: List[str] = []
        modified: List[str] = []

        # Use sequence matcher for clause-level comparison
        matcher = difflib.SequenceMatcher(None, clauses1, clauses2)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "insert":
                for clause in clauses2[j1:j2]:
                    added.append(clause[:100])
            elif tag == "delete":
                for clause in clauses1[i1:i2]:
                    removed.append(clause[:100])
            elif tag == "replace":
                for clause in clauses1[i1:i2]:
                    modified.append(f"Old: {clause[:50]}")
                for clause in clauses2[j1:j2]:
                    modified.append(f"New: {clause[:50]}")

        return added, removed, modified

    def _identify_changes(self, diffs: List[tuple]) -> List[Change]:
        """Categorize raw diffs into Change objects.

        Args:
            diffs: Raw diff tuples (tag, content).

        Returns:
            List of categorized Change objects.
        """
        changes: List[Change] = []

        i = 0
        while i < len(diffs):
            tag, content = diffs[i]

            if not content.strip():
                i += 1
                continue

            if tag == "added":
                significance = self._rate_significance(content)
                changes.append(
                    Change(
                        change_type=ChangeType.ADDED,
                        content=content,
                        significance=significance,
                    )
                )
            elif tag == "removed":
                # Check if next item is an addition (modification)
                if (
                    i + 1 < len(diffs)
                    and diffs[i + 1][0] == "added"
                    and diffs[i + 1][1].strip()
                ):
                    new_content = diffs[i + 1][1]
                    significance = self._rate_significance(content + " " + new_content)
                    changes.append(
                        Change(
                            change_type=ChangeType.MODIFIED,
                            content=new_content,
                            old_content=content,
                            significance=significance,
                        )
                    )
                    i += 1
                else:
                    significance = self._rate_significance(content)
                    changes.append(
                        Change(
                            change_type=ChangeType.REMOVED,
                            content=content,
                            significance=significance,
                        )
                    )

            i += 1

        return changes

    def _assess_significance(self, changes: List[Change]) -> float:
        """Assess the overall significance of all changes.

        Args:
            changes: List of categorized changes.

        Returns:
            Significance score between 0.0 and 1.0.
        """
        if not changes:
            return 0.0

        # Weight by individual change significance
        significance_weights = {
            SignificanceLevel.LOW: 0.1,
            SignificanceLevel.MEDIUM: 0.3,
            SignificanceLevel.HIGH: 0.6,
            SignificanceLevel.CRITICAL: 1.0,
        }

        total_weight = sum(
            significance_weights.get(c.significance, 0.3) for c in changes
        )

        # Normalize based on number of changes
        max_expected = 20.0  # Expect up to 20 significant changes
        score = min(1.0, total_weight / max_expected)

        return round(score, 3)

    def _rate_significance(self, content: str) -> SignificanceLevel:
        """Rate the significance of a single change.

        Args:
            content: The change content text.

        Returns:
            SignificanceLevel for this change.
        """
        content_lower = content.lower()

        # Check for high-significance keywords
        keyword_matches = sum(
            1 for kw in self._HIGH_SIGNIFICANCE_KEYWORDS if kw in content_lower
        )

        if keyword_matches >= 2:
            return SignificanceLevel.CRITICAL
        elif keyword_matches == 1:
            return SignificanceLevel.HIGH
        elif len(content) > 100:
            return SignificanceLevel.MEDIUM
        return SignificanceLevel.LOW

    def _generate_summary(
        self,
        changes: List[Change],
        added_clauses: List[str],
        removed_clauses: List[str],
    ) -> str:
        """Generate a human-readable summary of changes.

        Args:
            changes: Categorized changes.
            added_clauses: Added clauses.
            removed_clauses: Removed clauses.

        Returns:
            Summary string.
        """
        additions = sum(1 for c in changes if c.change_type == ChangeType.ADDED)
        removals = sum(1 for c in changes if c.change_type == ChangeType.REMOVED)
        modifications = sum(
            1 for c in changes if c.change_type == ChangeType.MODIFIED
        )

        parts = []
        if additions:
            parts.append(f"{additions} addition(s)")
        if removals:
            parts.append(f"{removals} removal(s)")
        if modifications:
            parts.append(f"{modifications} modification(s)")

        summary = f"Document comparison: {', '.join(parts) if parts else 'no changes detected'}."

        if added_clauses:
            summary += f" {len(added_clauses)} clause(s) added."
        if removed_clauses:
            summary += f" {len(removed_clauses)} clause(s) removed."

        return summary
