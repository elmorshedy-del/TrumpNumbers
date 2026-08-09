"""The guard that keeps the unrebuildable records unrebuilt-upon.

`scripts/check_records_append_only.py` is the only thing standing between a
plausible-looking cleanup and data that exists nowhere else. It is worth as much
as its ability to say no, so each way of destroying a record gets a test — and
so does each way of legitimately changing one, because a guard that also blocks
the normal path gets switched off.
"""

from __future__ import annotations

import gzip
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_records_append_only.py"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()


def commit(repo: Path, message: str) -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", message)
    return git(repo, "rev-parse", "HEAD")


def check(repo: Path, base: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--base", base],
        cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    (r / "data" / "record").mkdir(parents=True)
    git_dir_init = ["git", "-C", str(r), "init", "-q"]
    subprocess.run(git_dir_init, check=True)
    git(r, "config", "user.email", "t@example.com")
    git(r, "config", "user.name", "t")

    (r / "data" / "record" / "calendar.jsonl").write_text('{"a":1}\n{"a":2}\n')
    (r / "data" / "live_forecasts.jsonl").write_text('{"cut":0}\n{"cut":1}\n')
    # Derived output, deliberately NOT protected.
    (r / "data" / "exports").mkdir()
    (r / "data" / "exports" / "forecast.json").write_text("{}")
    commit(r, "seed")
    return r


def test_appending_is_allowed(repo):
    base = git(repo, "rev-parse", "HEAD")
    with (repo / "data" / "record" / "calendar.jsonl").open("a") as fh:
        fh.write('{"a":3}\n')
    commit(repo, "append")
    assert check(repo, base).returncode == 0


def test_deleting_a_record_is_refused(repo):
    base = git(repo, "rev-parse", "HEAD")
    (repo / "data" / "record" / "calendar.jsonl").unlink()
    commit(repo, "delete")
    out = check(repo, base)
    assert out.returncode == 1
    assert "calendar.jsonl: deleted" in out.stdout


def test_editing_an_existing_line_is_refused(repo):
    """The register's claim is that nothing was amended after the fact."""
    base = git(repo, "rev-parse", "HEAD")
    (repo / "data" / "live_forecasts.jsonl").write_text('{"cut":0}\n{"cut":99}\n')
    commit(repo, "amend a forecast")
    out = check(repo, base)
    assert out.returncode == 1
    assert "live_forecasts.jsonl: rewritten" in out.stdout


def test_truncating_the_front_is_refused(repo):
    """Dropping the oldest lines is the shape "compact the file" takes."""
    base = git(repo, "rev-parse", "HEAD")
    (repo / "data" / "record" / "calendar.jsonl").write_text('{"a":2}\n')
    commit(repo, "compact")
    assert check(repo, base).returncode == 1


def test_reformatting_without_losing_content_is_still_refused(repo):
    """Byte-prefix, not set-of-lines: a rewrite nobody can diff is still a rewrite."""
    base = git(repo, "rev-parse", "HEAD")
    (repo / "data" / "record" / "calendar.jsonl").write_text('{"a": 1}\n{"a": 2}\n')
    commit(repo, "pretty-print")
    assert check(repo, base).returncode == 1


def test_rotation_to_gzip_is_allowed(repo):
    """The one legal deletion: record.py gzips a period that has closed."""
    base = git(repo, "rev-parse", "HEAD")
    src = repo / "data" / "record" / "calendar.jsonl"
    with gzip.open(f"{src}.gz", "wb") as fh:
        fh.write(src.read_bytes())
    src.unlink()
    commit(repo, "rotate")
    assert check(repo, base).returncode == 0


def test_rotation_that_loses_content_is_refused(repo):
    """A .gz that does not carry the old bytes is a deletion wearing a costume."""
    base = git(repo, "rev-parse", "HEAD")
    src = repo / "data" / "record" / "calendar.jsonl"
    with gzip.open(f"{src}.gz", "wb") as fh:
        fh.write(b'{"a":2}\n')          # the first line is gone
    src.unlink()
    commit(repo, "rotate badly")
    assert check(repo, base).returncode == 1


def test_a_rotated_archive_can_never_be_removed(repo):
    base_gz = repo / "data" / "record" / "old.jsonl.gz"
    with gzip.open(base_gz, "wb") as fh:
        fh.write(b'{"old":1}\n')
    base = commit(repo, "an archive")
    base_gz.unlink()
    commit(repo, "drop the archive")
    out = check(repo, base)
    assert out.returncode == 1
    assert "never removed" in out.stdout


def test_derived_exports_may_be_rewritten_freely(repo):
    """The distinction the guard exists to draw: derived data is not protected."""
    base = git(repo, "rev-parse", "HEAD")
    (repo / "data" / "exports" / "forecast.json").write_text('{"totally":"different"}')
    commit(repo, "rebuild the exports")
    assert check(repo, base).returncode == 0


def test_a_first_push_has_nothing_to_protect(repo):
    """An all-zero base is what GitHub reports for a new branch, not a violation."""
    assert check(repo, "0" * 40).returncode == 0
