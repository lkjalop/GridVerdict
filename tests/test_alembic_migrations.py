"""Migration smoke tests — validate Alembic revision chain integrity.

These tests do NOT require a live Postgres database. They verify:
  1. All migration files exist and are syntactically valid Python
  2. The revision chain forms an unambiguous single linear sequence
  3. upgrade() and downgrade() are defined in every migration
  4. The head revision matches the expected latest version

For a full upgrade/downgrade/upgrade cycle against a real Postgres database,
run the companion script:
    python scripts/smoke_alembic.py

That script requires DATABASE_URL to be set (see .env.example).
"""
from __future__ import annotations

import ast
import importlib.util
import os
import sys
from pathlib import Path

import pytest

_MIGRATIONS_DIR = (
    Path(__file__).parent.parent / "app" / "db" / "migrations" / "versions"
)
_EXPECTED_HEAD = "0006"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_migration_modules() -> list[tuple[str, ast.Module]]:
    """Load all *.py migration files as AST modules (no import, no DB needed)."""
    files = sorted(_MIGRATIONS_DIR.glob("*.py"))
    assert files, f"No migration files found in {_MIGRATIONS_DIR}"
    results = []
    for f in files:
        src = f.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(f))
        results.append((f.name, tree))
    return results


def _extract_revision_meta(tree: ast.Module) -> dict[str, str | None]:
    """Extract revision, down_revision, upgrade, downgrade from an AST.

    Handles both plain assignments (``revision = "0001"``) and annotated
    assignments (``revision: str = "0001"``), which Alembic auto-generates.
    """
    meta: dict[str, str | None] = {
        "revision": None,
        "down_revision": None,
        "has_upgrade": False,
        "has_downgrade": False,
    }
    for node in ast.walk(tree):
        # Plain assignment: revision = "..."
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == "revision" and isinstance(node.value, ast.Constant):
                        meta["revision"] = node.value.value
                    elif target.id == "down_revision" and isinstance(node.value, ast.Constant):
                        meta["down_revision"] = node.value.value  # None when value is None
        # Annotated assignment: revision: str = "..."
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.value is not None:
                name = node.target.id
                if name == "revision" and isinstance(node.value, ast.Constant):
                    meta["revision"] = node.value.value
                elif name == "down_revision" and isinstance(node.value, ast.Constant):
                    meta["down_revision"] = node.value.value
        if isinstance(node, ast.FunctionDef):
            if node.name == "upgrade":
                meta["has_upgrade"] = True
            elif node.name == "downgrade":
                meta["has_downgrade"] = True
    return meta


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestMigrationFiles:

    def test_migration_directory_exists(self):
        assert _MIGRATIONS_DIR.exists(), f"Migration directory not found: {_MIGRATIONS_DIR}"

    def test_at_least_three_migrations_present(self):
        files = list(_MIGRATIONS_DIR.glob("[0-9]*.py"))
        assert len(files) >= 3, f"Expected at least 3 migration files, found {len(files)}"

    def test_all_migration_files_are_valid_python(self):
        modules = _load_migration_modules()
        for filename, tree in modules:
            # If ast.parse succeeded (used in _load), the file is valid Python
            assert tree is not None, f"{filename} failed to parse"

    def test_all_migrations_have_upgrade_and_downgrade(self):
        modules = _load_migration_modules()
        for filename, tree in modules:
            meta = _extract_revision_meta(tree)
            assert meta["has_upgrade"], f"{filename} missing upgrade() function"
            assert meta["has_downgrade"], f"{filename} missing downgrade() function"

    def test_all_migrations_have_revision_id(self):
        modules = _load_migration_modules()
        for filename, tree in modules:
            meta = _extract_revision_meta(tree)
            assert meta["revision"] is not None, f"{filename} missing revision string"

    def test_revision_ids_are_unique(self):
        modules = _load_migration_modules()
        revisions = []
        for filename, tree in modules:
            meta = _extract_revision_meta(tree)
            revisions.append(meta["revision"])
        assert len(revisions) == len(set(revisions)), (
            f"Duplicate revision IDs found: {[r for r in revisions if revisions.count(r) > 1]}"
        )


class TestMigrationChain:

    def _build_chain(self) -> dict[str, dict]:
        """Map revision → {down_revision, filename, has_upgrade, has_downgrade}."""
        modules = _load_migration_modules()
        chain = {}
        for filename, tree in modules:
            meta = _extract_revision_meta(tree)
            if meta["revision"]:
                chain[meta["revision"]] = {
                    "down_revision": meta["down_revision"],
                    "filename": filename,
                    **meta,
                }
        return chain

    def test_chain_has_exactly_one_root(self):
        """Exactly one migration should have down_revision=None (the initial migration)."""
        chain = self._build_chain()
        roots = [rev for rev, m in chain.items() if m["down_revision"] is None]
        assert len(roots) == 1, f"Expected 1 root migration, found: {roots}"

    def test_chain_has_exactly_one_head(self):
        """Exactly one migration should not be pointed to by any down_revision (the head)."""
        chain = self._build_chain()
        all_down = {m["down_revision"] for m in chain.values() if m["down_revision"]}
        heads = [rev for rev in chain if rev not in all_down]
        assert len(heads) == 1, f"Expected 1 head migration, found: {heads}"

    def test_chain_is_linear_no_branches(self):
        """Every revision should appear at most once as a down_revision."""
        chain = self._build_chain()
        down_revs = [m["down_revision"] for m in chain.values() if m["down_revision"]]
        assert len(down_revs) == len(set(down_revs)), (
            "Non-linear chain: a revision appears as down_revision more than once"
        )

    def test_chain_has_no_dangling_references(self):
        """Every down_revision must point to an existing revision."""
        chain = self._build_chain()
        for rev, meta in chain.items():
            dr = meta["down_revision"]
            if dr is not None:
                assert dr in chain, (
                    f"{meta['filename']}: down_revision={dr!r} not found in migration set"
                )

    def test_head_matches_expected(self):
        """The head migration should be the expected latest revision."""
        chain = self._build_chain()
        all_down = {m["down_revision"] for m in chain.values() if m["down_revision"]}
        heads = [rev for rev in chain if rev not in all_down]
        assert len(heads) == 1
        assert heads[0] == _EXPECTED_HEAD, (
            f"Expected head revision {_EXPECTED_HEAD!r}, found {heads[0]!r}"
        )

    def test_full_upgrade_path_covers_all_revisions(self):
        """Walking upgrade from root to head should visit every revision exactly once."""
        chain = self._build_chain()
        # Find root
        roots = [rev for rev, m in chain.items() if m["down_revision"] is None]
        assert roots
        # Build forward map: down_revision → revision
        forward: dict[str | None, str] = {}
        for rev, meta in chain.items():
            forward[meta["down_revision"]] = rev

        visited = []
        current: str | None = None  # start from None (root's down_revision)
        while current in forward:
            nxt = forward[current]
            assert nxt not in visited, f"Cycle detected at {nxt}"
            visited.append(nxt)
            current = nxt

        assert len(visited) == len(chain), (
            f"Expected to visit {len(chain)} revisions, visited {len(visited)}: {visited}"
        )
