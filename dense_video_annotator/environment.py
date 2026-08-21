"""Lazy setup and discovery for heavyweight external recipe environments."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


COSMOS_CURATOR_REPOSITORY = "https://github.com/NVIDIA/cosmos-curator.git"
COSMOS_CURATOR_REVISION = "975b68910f23067fb2391e68368d5f7cd8cf64ce"
COSMOS_CURATOR_ARCHIVE = (
    "https://github.com/NVIDIA/cosmos-curator/archive/"
    f"{COSMOS_CURATOR_REVISION}.tar.gz"
)
COSMOS_CURATOR_ENV_VAR = "DVA_COSMOS_CURATOR_ROOT"
PIXI_VERSION = "0.77.0"
_AUDITED_COSMOS_SYMLINKS = {
    "AGENTS.md": "CLAUDE.md",
    "GEMINI.md": "CLAUDE.md",
}


@dataclass(frozen=True)
class EnvironmentReport:
    """Read-only status for one external recipe environment."""

    name: str
    root: Path
    pixi: str | None
    repository: Path
    manifest: Path
    revision: str | None
    ready: bool
    missing: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "root": str(self.root),
            "pixi": self.pixi,
            "repository": str(self.repository),
            "manifest": str(self.manifest),
            "revision": self.revision,
            "ready": self.ready,
            "missing": list(self.missing),
        }


def default_cosmos_root() -> Path:
    configured = os.environ.get(COSMOS_CURATOR_ENV_VAR)
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".cache" / "dense-video-annotator" / "cosmos-curator").resolve()


def cosmos_repository(root: str | Path) -> Path:
    return Path(root).expanduser().resolve() / "source"


def _local_pixi(root: Path) -> Path:
    return root / "tools" / ("pixi.exe" if sys.platform == "win32" else "pixi")


def _pixi_asset() -> str:
    system = sys.platform
    machine = os.uname().machine.lower() if hasattr(os, "uname") else ""
    arch = "aarch64" if machine in {"arm64", "aarch64"} else "x86_64"
    if system == "darwin":
        return f"pixi-{arch}-apple-darwin.tar.gz"
    if system == "linux":
        return f"pixi-{arch}-unknown-linux-musl.tar.gz"
    if system == "win32":
        return f"pixi-{arch}-pc-windows-msvc.zip"
    raise RuntimeError(f"automatic Pixi setup is unsupported on platform {system!r}")


def _pixi_path(root: Path) -> str | None:
    local = _local_pixi(root)
    if local.is_file() and os.access(local, os.X_OK):
        return str(local)
    return shutil.which("pixi")


def _download_pixi(root: Path) -> Path:
    destination = _local_pixi(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    asset = _pixi_asset()
    url = f"https://github.com/prefix-dev/pixi/releases/download/v{PIXI_VERSION}/{asset}"
    with tempfile.TemporaryDirectory(prefix="dva-pixi-") as temporary:
        archive = Path(temporary) / asset
        try:
            urllib.request.urlretrieve(url, archive)
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeError(f"failed to download Pixi from {url}") from exc
        if asset.endswith(".zip"):
            import zipfile

            with zipfile.ZipFile(archive) as handle:
                member = "pixi.exe"
                destination.write_bytes(handle.read(member))
        else:
            with tarfile.open(archive, "r:gz") as handle:
                member = next((item for item in handle.getmembers() if item.name.endswith("/pixi") or item.name == "pixi"), None)
                if member is None:
                    raise RuntimeError("downloaded Pixi archive did not contain a pixi executable")
                payload = handle.extractfile(member)
                if payload is None:
                    raise RuntimeError("could not read Pixi executable from downloaded archive")
                destination.write_bytes(payload.read())
    destination.chmod(0o755)
    return destination


def _download_cosmos_source(repository: Path) -> None:
    """Download the pinned source archive without requiring a system git."""

    repository.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dva-cosmos-") as temporary:
        archive = Path(temporary) / "cosmos-curator.tar.gz"
        try:
            urllib.request.urlretrieve(COSMOS_CURATOR_ARCHIVE, archive)
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeError(
                f"failed to download Cosmos Curator revision {COSMOS_CURATOR_REVISION} "
                f"from {COSMOS_CURATOR_ARCHIVE}"
            ) from exc

        with tarfile.open(archive, "r:gz") as handle:
            members = handle.getmembers()
            if not members:
                raise RuntimeError("downloaded Cosmos Curator archive is empty")
            # GitHub archives contain one top-level directory. Reject absolute and
            # parent-traversal paths before extracting into the cache directory.
            for member in members:
                candidate = Path(member.name)
                if candidate.is_absolute() or ".." in candidate.parts:
                    raise RuntimeError("downloaded Cosmos Curator archive contains an unsafe path")
            top_levels = {Path(member.name).parts[0] for member in members if Path(member.name).parts}
            if len(top_levels) != 1:
                raise RuntimeError("downloaded Cosmos Curator archive has an unexpected layout")
            top_level = next(iter(top_levels))
            audited_links: list[tuple[Path, str]] = []
            regular_members = []
            for member in members:
                if member.islnk():
                    raise RuntimeError("downloaded Cosmos Curator archive contains a hard link")
                if member.issym():
                    relative = Path(member.name).relative_to(top_level)
                    if (
                        len(relative.parts) != 1
                        or _AUDITED_COSMOS_SYMLINKS.get(relative.as_posix()) != member.linkname
                    ):
                        raise RuntimeError(
                            "downloaded Cosmos Curator archive contains an unaudited symbolic link"
                        )
                    audited_links.append((relative, member.linkname))
                    continue
                regular_members.append(member)

            extracted = Path(temporary) / top_level
            handle.extractall(temporary, members=regular_members)
            for relative, target in audited_links:
                target_path = extracted / target
                if not target_path.is_file():
                    raise RuntimeError(
                        "downloaded Cosmos Curator archive has a broken audited symbolic link"
                    )
                (extracted / relative).symlink_to(target)
            if not (extracted / "pixi.toml").is_file():
                raise RuntimeError("downloaded Cosmos Curator archive has no pixi.toml")
            if repository.exists():
                raise RuntimeError(
                    f"refusing to replace an existing directory: {repository}; "
                    "choose an empty --path"
                )
            shutil.move(str(extracted), str(repository))

    # Keep provenance available even when the source was installed without git.
    (repository / ".dva-source.json").write_text(
        '{"repository": "' + COSMOS_CURATOR_REPOSITORY + '", '
        '"revision": "' + COSMOS_CURATOR_REVISION + '", '
        '"archive": "' + COSMOS_CURATOR_ARCHIVE + '"}\n',
        encoding="utf-8",
    )


def _clone_cosmos_source(repository: Path) -> None:
    """Clone the pinned revision when Git is available so setuptools-scm sees metadata."""

    repository.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                COSMOS_CURATOR_REPOSITORY,
                str(repository),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repository), "checkout", "--detach", COSMOS_CURATOR_REVISION],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        if repository.exists():
            shutil.rmtree(repository)
        detail = getattr(exc, "stderr", "") or ""
        raise RuntimeError("failed to clone the pinned Cosmos Curator revision" + (f": {detail.strip()[-800:]}" if detail else "")) from exc


def _ensure_pixi(root: Path, *, allow_download: bool) -> str | None:
    existing = _pixi_path(root)
    if existing or not allow_download:
        return existing
    return str(_download_pixi(root))


def _git_revision(repository: Path) -> str | None:
    if not (repository / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _source_revision(repository: Path) -> str | None:
    revision = _git_revision(repository)
    if revision:
        return revision
    provenance = repository / ".dva-source.json"
    if not provenance.is_file():
        return None
    try:
        import json

        data = json.loads(provenance.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = data.get("revision") if isinstance(data, dict) else None
    return value if isinstance(value, str) else None


def inspect_cosmos_environment(root: str | Path | None = None) -> EnvironmentReport:
    resolved_root = (
        Path(root).expanduser().resolve() if root is not None else default_cosmos_root()
    )
    repository = cosmos_repository(resolved_root)
    manifest = repository / "pixi.toml"
    pixi = _pixi_path(resolved_root)
    revision = _source_revision(repository)
    missing: list[str] = []
    if pixi is None:
        missing.append("Pixi executable (run `dva env install cosmos-curator`)")
    if not repository.is_dir():
        missing.append(f"Cosmos Curator checkout at {repository}")
    if not manifest.is_file():
        missing.append(f"Cosmos Curator manifest at {manifest}")
    pixi_environment = repository / ".pixi" / "envs" / "default"
    if not pixi_environment.is_dir():
        missing.append(f"Pixi environment at {pixi_environment}")
    if revision != COSMOS_CURATOR_REVISION:
        missing.append(f"pinned revision {COSMOS_CURATOR_REVISION}")
    return EnvironmentReport(
        name="cosmos-curator",
        root=resolved_root,
        pixi=pixi,
        repository=repository,
        manifest=manifest,
        revision=revision,
        ready=not missing,
        missing=tuple(missing),
    )


def cosmos_pixi_command(
    config_path: str | Path,
    *,
    root: str | Path | None = None,
) -> list[str]:
    report = inspect_cosmos_environment(root)
    if not report.ready:
        details = "; ".join(report.missing)
        raise RuntimeError(
            "Cosmos Curator environment is not ready: "
            f"{details}. Run `dva env install cosmos-curator` or use import mode."
        )
    return [
        report.pixi or "pixi",
        "run",
        "--manifest-path",
        str(report.manifest),
        "--as-is",
        "video-pipeline",
        str(Path(config_path).expanduser().resolve()),
    ]


def install_cosmos_environment(
    root: str | Path | None = None,
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> EnvironmentReport:
    """Download the pinned upstream checkout and let Pixi resolve its environment."""

    resolved_root = (
        Path(root).expanduser().resolve() if root is not None else default_cosmos_root()
    )
    resolved_root.mkdir(parents=True, exist_ok=True)
    pixi = _ensure_pixi(resolved_root, allow_download=True)
    if pixi is None:
        raise RuntimeError("could not prepare the local Pixi executable")
    repository = cosmos_repository(resolved_root)
    def run_checked(command: list[str]) -> None:
        try:
            command_runner(
                command,
                capture_output=True,
                text=True,
                check=True,
                env={
                    **os.environ,
                    # GitHub archive fallback has no .git metadata for setuptools-scm.
                    "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_COSMOS_CURATOR": "0.0.0",
                },
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"required setup executable is unavailable: {command[0]}") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()[-1200:]
            if "missing virtual packages" in detail and (
                "__cuda" in detail or "__glibc" in detail
            ):
                raise RuntimeError(
                    "Cosmos Curator's pinned Pixi environment is incompatible with this host: "
                    "the official lock requires CUDA/glibc virtual packages that are not present. "
                    "Use an NVIDIA-compatible host or the upstream container/cluster environment; "
                    "do not use CONDA_OVERRIDE_* to bypass this check."
                ) from exc
            raise RuntimeError(
                f"external setup command failed ({' '.join(command[:3])})"
                + (f": {detail}" if detail else "")
            ) from exc

    if repository.exists() and _source_revision(repository) != COSMOS_CURATOR_REVISION:
        raise RuntimeError(
            f"Cosmos Curator source at {repository} is not the pinned revision "
            f"{COSMOS_CURATOR_REVISION}; choose an empty --path"
        )
    if not repository.exists():
        if shutil.which("git"):
            _clone_cosmos_source(repository)
        else:
            _download_cosmos_source(repository)
    run_checked(
        [pixi, "install", "--manifest-path", str(repository / "pixi.toml")],
    )
    report = inspect_cosmos_environment(resolved_root)
    if not report.ready:
        raise RuntimeError(
            "Cosmos Curator installation finished but the environment is incomplete: "
            + "; ".join(report.missing)
        )
    return report


def resolve_cosmos_command(
    config_path: str | Path,
    command: Sequence[str] | None,
    *,
    environment_path: str | Path | None = None,
) -> list[str] | None:
    """Resolve an explicit command or a configured lazy Pixi environment."""

    if command is not None:
        return list(command)
    if environment_path is not None:
        return cosmos_pixi_command(config_path, root=environment_path)
    if os.environ.get(COSMOS_CURATOR_ENV_VAR):
        return cosmos_pixi_command(config_path)
    return None
