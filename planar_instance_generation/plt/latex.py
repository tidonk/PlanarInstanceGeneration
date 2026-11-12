"""Compile standalone TikZ documents without leaking LaTeX intermediates."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .tikz import PlotArtifacts


@dataclass(frozen=True, slots=True)
class PlotPdfArtifacts:
    """The compiled PDF and the two logs retained for diagnostics."""

    pdf: Path
    stdout_log: Path
    latex_log: Path
    reused: bool = False


PDFLATEX_EXTRA_MEMORY_WORDS = 12_000_000
PDFLATEX_TIMEOUT_SECONDS = 600.0


class LatexCompilationError(RuntimeError):
    """Raised when ``pdflatex`` cannot produce a standalone plot PDF."""


def compile_plot_pdf(
    plot: PlotArtifacts,
    *,
    output_pdf: str | Path,
    log_directory: str | Path,
    overwrite: bool = False,
    timeout_seconds: float = PDFLATEX_TIMEOUT_SECONDS,
) -> PlotPdfArtifacts:
    """Compile one existing :class:`PlotArtifacts` standalone document.

    Compilation happens in a temporary directory.  Only the PDF, captured
    process output, and native LaTeX log are retained at the requested paths.
    """

    tex_path = Path(plot.standalone_tex)
    tikz_path = Path(plot.tikz)
    _validate_source(tex_path, tikz_path)
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    pdf_path = Path(output_pdf)
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("output_pdf must have a .pdf extension")
    logs = Path(log_directory)
    artifacts = PlotPdfArtifacts(
        pdf=pdf_path,
        stdout_log=logs / f"{pdf_path.stem}.stdout.log",
        latex_log=logs / f"{pdf_path.stem}.latex.log",
    )
    if _reuse_or_raise(artifacts, overwrite):
        return PlotPdfArtifacts(
            pdf=artifacts.pdf,
            stdout_log=artifacts.stdout_log,
            latex_log=artifacts.latex_log,
            reused=True,
        )

    executable = shutil.which("pdflatex")
    if executable is None:
        raise LatexCompilationError("pdflatex was not found on PATH")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pig-pdflatex-") as temporary:
        build_directory = Path(temporary)
        command = [
            executable,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            f"-output-directory={build_directory}",
            tex_path.name,
        ]
        try:
            process = subprocess.run(
                command,
                cwd=tex_path.parent,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout_seconds,
                env={
                    **os.environ,
                    "extra_mem_top": str(PDFLATEX_EXTRA_MEMORY_WORDS),
                },
            )
        except subprocess.TimeoutExpired as error:
            output = _timeout_output(error)
            artifacts.stdout_log.write_text(output, encoding="utf-8")
            _retain_native_log(
                build_directory / f"{tex_path.stem}.log",
                artifacts.latex_log,
            )
            _discard_stale_pdf(artifacts.pdf, overwrite)
            raise LatexCompilationError(
                f"pdflatex timed out after {timeout_seconds:g} seconds; "
                f"see {artifacts.stdout_log}"
            ) from error
        except OSError as error:
            _discard_stale_pdf(artifacts.pdf, overwrite)
            raise LatexCompilationError(f"could not execute pdflatex: {error}") from error

        artifacts.stdout_log.write_text(process.stdout or "", encoding="utf-8")
        compiled_pdf = build_directory / f"{tex_path.stem}.pdf"
        native_log = build_directory / f"{tex_path.stem}.log"
        _retain_native_log(native_log, artifacts.latex_log)

        if process.returncode != 0:
            _discard_stale_pdf(artifacts.pdf, overwrite)
            raise LatexCompilationError(
                _failure_message(process.returncode, process.stdout, artifacts)
            )
        if not compiled_pdf.is_file() or compiled_pdf.stat().st_size == 0:
            _discard_stale_pdf(artifacts.pdf, overwrite)
            raise LatexCompilationError(
                "pdflatex exited successfully but produced no nonempty PDF; "
                f"see {artifacts.stdout_log}"
            )
        if not native_log.is_file():
            _discard_stale_pdf(artifacts.pdf, overwrite)
            raise LatexCompilationError(
                "pdflatex exited successfully but produced no native log; "
                f"see {artifacts.stdout_log}"
            )

        shutil.copyfile(compiled_pdf, artifacts.pdf)
    return artifacts


def _validate_source(tex_path: Path, tikz_path: Path) -> None:
    if tex_path.suffix.lower() != ".tex" or not tex_path.is_file():
        raise ValueError(f"expected an existing standalone .tex file: {tex_path}")
    if tikz_path.suffix.lower() != ".tikz" or not tikz_path.is_file():
        raise ValueError(f"expected an existing TikZ fragment: {tikz_path}")
    if tex_path.stem != tikz_path.stem or tex_path.parent != tikz_path.parent:
        raise ValueError("standalone .tex and .tikz files must share a directory and stem")


def _reuse_or_raise(artifacts: PlotPdfArtifacts, overwrite: bool) -> bool:
    existing = (
        artifacts.pdf.exists(),
        artifacts.stdout_log.exists(),
        artifacts.latex_log.exists(),
    )
    if all(existing) and not overwrite:
        return True
    if any(existing) and not overwrite:
        raise FileExistsError(
            f"incomplete compiled plot artifact set for {artifacts.pdf}; "
            "use overwrite=True"
        )
    return False


def _discard_stale_pdf(path: Path, overwrite: bool) -> None:
    if overwrite:
        path.unlink(missing_ok=True)


def _retain_native_log(source: Path, destination: Path) -> None:
    if source.is_file():
        shutil.copyfile(source, destination)


def _timeout_output(error: subprocess.TimeoutExpired) -> str:
    output = error.stdout or ""
    if isinstance(output, bytes):
        output = output.decode(errors="replace")
    return str(output)


def _failure_message(
    returncode: int,
    stdout: str | None,
    artifacts: PlotPdfArtifacts,
) -> str:
    nonempty_lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
    tail = " | ".join(nonempty_lines[-5:])
    detail = f" Last output: {tail}" if tail else ""
    logs = str(artifacts.stdout_log)
    if artifacts.latex_log.is_file():
        logs += f" and {artifacts.latex_log}"
    return f"pdflatex failed with exit code {returncode}; see {logs}.{detail}"
