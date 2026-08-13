"""
Tests for sandboxed web acquisition.

The risk being guarded against is not a failed download; it is a download that
writes somewhere it should not, or that arrives as code and gets run. These
assert the containment rather than the fetching.
"""

import os

import pytest

from agents import sandbox, web_acquire


class TestContainment:
    def test_path_inside_root_is_contained(self, tmp_path):
        inside = tmp_path / "a" / "b.txt"
        inside.parent.mkdir(parents=True)
        inside.write_text("x")
        assert sandbox.is_contained(str(inside), str(tmp_path))

    def test_parent_traversal_is_not_contained(self, tmp_path):
        assert not sandbox.is_contained(str(tmp_path / ".." / "elsewhere"), str(tmp_path))

    def test_symlink_pointing_out_is_not_contained(self, tmp_path):
        """A file that resolves outside the root is an escape even if its path looks fine."""
        link = tmp_path / "innocent"
        link.symlink_to("/etc")
        assert not sandbox.is_contained(str(link), str(tmp_path))

    def test_audit_reports_escaping_entries(self, tmp_path):
        (tmp_path / "fine.txt").write_text("x")
        (tmp_path / "sneaky").symlink_to("/etc")
        escaped = web_acquire._audit_containment(str(tmp_path))
        assert len(escaped) == 1
        assert escaped[0].endswith("sneaky")

    def test_audit_is_clean_for_an_ordinary_tree(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "a.bin").write_text("x")
        assert web_acquire._audit_containment(str(tmp_path)) == []


@pytest.mark.skipif(not sandbox.available(), reason="sandbox-exec is macOS only")
class TestSandboxEnforcement:
    def test_write_inside_the_root_succeeds(self, tmp_path):
        target = tmp_path / "ok.txt"
        code, _out, _err, confined = sandbox.run_sandboxed(
            ["/bin/sh", "-c", f"echo hi > {target}"], write_root=str(tmp_path), timeout=30)
        assert confined
        assert code == 0
        assert target.read_text().strip() == "hi"

    def test_write_outside_the_root_is_denied(self, tmp_path):
        """The whole point: a download cannot write into the repo."""
        outside = tmp_path.parent / "escaped.txt"
        _code, _out, err, confined = sandbox.run_sandboxed(
            ["/bin/sh", "-c", f"echo pwned > {outside}"], write_root=str(tmp_path), timeout=30)
        assert confined
        assert not outside.exists()
        assert "not permitted" in err.lower()


class TestNoCodeExecution:
    def test_safe_patterns_exclude_pickles_and_scripts(self):
        """
        .bin/.pt/.ckpt are pickles that execute on load, so they are not fetched
        by default; safetensors carry no such risk.
        """
        patterns = web_acquire.SAFE_MODEL_PATTERNS
        assert "*.safetensors" in patterns
        for risky in ("*.bin", "*.pt", "*.pth", "*.ckpt", "*.py", "*.pkl"):
            assert risky not in patterns

    def test_repo_id_must_be_owner_slash_name(self):
        assert web_acquire.download_model("not-a-repo")["success"] is False

    def test_oversized_repo_is_refused_before_downloading(self, monkeypatch):
        monkeypatch.setattr(web_acquire, "inspect_model", lambda r: {
            "success": True, "total_size_mb": 50_000, "licence": "apache-2.0", "has_pickled_weights": False})
        called = []
        monkeypatch.setattr(web_acquire, "_run_worker", lambda *a, **k: called.append(1))

        result = web_acquire.download_model("big/model", max_gb=6.0)
        assert result["success"] is False
        assert "above the" in result["error"]
        assert called == [], "must refuse before starting the download"


class TestInputValidation:
    def test_subject_must_not_contain_a_path_separator(self):
        result = web_acquire.download_images(f"..{os.sep}escape", "anything")
        assert result["success"] is False

    def test_subject_is_required(self):
        assert web_acquire.download_images("", "q")["success"] is False

    def test_query_is_required(self):
        assert web_acquire.download_images("subject", "")["success"] is False


class TestPromote:
    def test_refuses_a_path_outside_quarantine(self):
        assert web_acquire.promote("../../etc/passwd")["success"] is False

    def test_refuses_something_that_is_not_there(self):
        assert web_acquire.promote("models/definitely_not_here")["success"] is False

    def test_moves_a_vetted_download_out(self, tmp_path, monkeypatch):
        quarantine = tmp_path / "quarantine"
        (quarantine / "models" / "thing").mkdir(parents=True)
        (quarantine / "models" / "thing" / "w.safetensors").write_text("x")
        monkeypatch.setattr(web_acquire, "QUARANTINE_ROOT", str(quarantine))

        destination = tmp_path / "checkpoints"
        result = web_acquire.promote("models/thing", str(destination))

        assert result["success"] is True
        assert (destination / "thing" / "w.safetensors").exists()
        assert not (quarantine / "models" / "thing").exists()

    def test_refuses_to_overwrite_an_existing_destination(self, tmp_path, monkeypatch):
        quarantine = tmp_path / "quarantine"
        (quarantine / "models" / "thing").mkdir(parents=True)
        monkeypatch.setattr(web_acquire, "QUARANTINE_ROOT", str(quarantine))
        destination = tmp_path / "checkpoints"
        (destination / "thing").mkdir(parents=True)

        assert web_acquire.promote("models/thing", str(destination))["success"] is False
