"""A4 inspection report for one generated experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Iterable


@dataclass(frozen=True, slots=True)
class ReviewFigure:
    """One structural or skewness comparison PDF."""

    label: str
    pdf: Path


@dataclass(frozen=True, slots=True)
class ReviewBoundRow:
    """One compact paper-bound versus observation row."""

    property: str
    paper_bound: str
    observed: str
    check: str
    slack: str = "n/a"


@dataclass(frozen=True, slots=True)
class ReviewInstance:
    """The stage figures shown together on one report page.

    ``bus_count`` is retained for backward compatibility; ``size_unit``
    controls the domain noun printed in the heading.
    """

    stem: str
    bus_count: int
    stage_pdfs: tuple[Path, ...]
    detail: str = ""
    preservation_rows: tuple[ReviewBoundRow, ...] = ()
    size_unit: str = "buses"

    @property
    def size_count(self) -> int:
        return self.bus_count


@dataclass(frozen=True, slots=True)
class ReviewReportArtifacts:
    tex: Path
    pdf: Path | None
    stdout_log: Path | None


def write_a4_review_report(
    output_base: str | Path,
    comparison_figures: Iterable[ReviewFigure],
    instances: Iterable[ReviewInstance],
    *,
    title: str = "Planar instance generation review",
    compile_pdf: bool = True,
    log_directory: str | Path | None = None,
    overwrite: bool = False,
) -> ReviewReportArtifacts:
    """Write the requested comparisons-first, one-final-instance-per-page report."""

    figures = tuple(comparison_figures)
    reviewed = tuple(instances)
    if not figures:
        raise ValueError("at least one comparison figure is required")
    if not reviewed:
        raise ValueError("at least one final instance is required")
    _validate_inputs(figures, reviewed)

    base = Path(output_base)
    if base.suffix:
        raise ValueError("output_base must not have a file extension")
    tex_path = base.with_suffix(".tex")
    pdf_path = base.with_suffix(".pdf")
    if not overwrite and (tex_path.exists() or pdf_path.exists()):
        raise FileExistsError(f"review report already exists at {base}")
    base.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(_report_tex(title, figures, reviewed), encoding="utf-8")
    if not compile_pdf:
        return ReviewReportArtifacts(tex_path, None, None)

    executable = shutil.which("pdflatex")
    if executable is None:
        raise RuntimeError("pdflatex is required to compile the A4 review report")
    logs = Path(log_directory) if log_directory is not None else base.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [
            executable,
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={logs}",
            tex_path.name,
        ],
        cwd=tex_path.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    stdout_log = logs / f"{base.name}.stdout.log"
    stdout_log.write_text(process.stdout, encoding="utf-8")
    compiled = logs / f"{base.name}.pdf"
    if process.returncode != 0 or not compiled.is_file():
        raise RuntimeError(f"A4 report compilation failed; see {stdout_log}")
    if pdf_path.exists():
        pdf_path.unlink()
    compiled.replace(pdf_path)
    return ReviewReportArtifacts(tex_path, pdf_path, stdout_log)


def _validate_inputs(
    figures: tuple[ReviewFigure, ...], instances: tuple[ReviewInstance, ...]
) -> None:
    for figure in figures:
        if not figure.label:
            raise ValueError("comparison figure labels must not be empty")
        _require_pdf(figure.pdf)
    stems = set()
    for instance in instances:
        if not instance.stem or instance.stem in stems:
            raise ValueError("review instance stems must be nonempty and unique")
        stems.add(instance.stem)
        if instance.size_count < 1 or not instance.stage_pdfs:
            raise ValueError("review instances need a positive size and stage PDFs")
        if not isinstance(instance.size_unit, str) or not instance.size_unit.strip():
            raise ValueError("review instance size units must be nonempty strings")
        if any(
            not row.property
            or not row.paper_bound
            or not row.observed
            or not row.check
            or not row.slack
            for row in instance.preservation_rows
        ):
            raise ValueError("preservation table cells must not be empty")
        for path in instance.stage_pdfs:
            _require_pdf(path)


def _require_pdf(path: Path) -> None:
    if path.suffix.lower() != ".pdf" or not path.is_file():
        raise ValueError(f"expected an existing PDF: {path}")


def _report_tex(
    title: str,
    figures: tuple[ReviewFigure, ...],
    instances: tuple[ReviewInstance, ...],
) -> str:
    lines = [
        "\\documentclass[10pt,a4paper]{article}\n",
        "\\usepackage[T1]{fontenc}\n",
        "\\usepackage{lmodern}\n",
        "\\usepackage[left=14mm,right=14mm,top=22mm,bottom=16mm,headheight=14pt,headsep=5mm]{geometry}\n",
        "\\usepackage{graphicx}\n",
        "\\usepackage{booktabs,tabularx}\n",
        "\\usepackage{xcolor}\n",
        "\\usepackage{microtype}\n",
        "\\usepackage{fancyhdr}\n",
        "\\pagestyle{fancy}\\fancyhf{}\n",
        f"\\lhead{{{_escape(title)}}}\\rhead{{\\thepage}}\n",
        "\\setlength{\\parindent}{0pt}\n",
        "\\begin{document}\n",
        f"\\begin{{center}}\\Large\\bfseries {_escape(title)}\\end{{center}}\n",
        "\\vspace{1mm}\n",
        "\\section*{Structural properties and skewness}\n",
    ]
    for index, figure in enumerate(figures, start=1):
        lines.extend(
            [
                "\\begin{center}\n",
                f"\\includegraphics[width=.94\\textwidth,height=.34\\textheight,keepaspectratio]{{{_pdf_path(figure.pdf)}}}\\par\n",
                f"\\small\\bfseries {_escape(figure.label)}\n",
                "\\end{center}\n",
            ]
        )
        if index % 2 == 0 and index != len(figures):
            lines.append("\\clearpage\n\\section*{Structural properties and skewness (continued)}\n")
        elif index % 2 == 1:
            lines.append("\\vfill\n")

    for instance in instances:
        available_height = 0.54 if instance.preservation_rows else 0.72
        image_height = min(available_height / len(instance.stage_pdfs), 0.32)
        lines.extend(
            [
                "\\clearpage\n",
                f"\\section*{{Instance {_escape(instance.stem)} "
                f"({instance.size_count} {_escape(instance.size_unit)})}}\n",
            ]
        )
        if instance.detail:
            lines.append(f"{{\\small {_escape(instance.detail)}}}\\par\\vspace{{1mm}}\n")
        if instance.preservation_rows:
            lines.extend(
                [
                    "{\\scriptsize\n",
                    "\\setlength{\\tabcolsep}{3pt}\n",
                    "\\begin{tabularx}{\\textwidth}{@{}p{.25\\textwidth}XXXp{.13\\textwidth}@{}}\n",
                    "\\toprule Property & Paper bound & Observed & Slack & Check \\\\\n",
                    "\\midrule\n",
                ]
            )
            for row in instance.preservation_rows:
                lines.append(
                    f"{_escape(row.property)} & {_escape(row.paper_bound)} & "
                    f"{_escape(row.observed)} & {_escape(row.slack)} & "
                    f"{_escape(row.check)} \\\\\n"
                )
            lines.extend(
                ["\\bottomrule\n", "\\end{tabularx}}\n", "\\vspace{1mm}\n"]
            )
        for stage, pdf in enumerate(instance.stage_pdfs, start=1):
            lines.extend(
                [
                    "\\begin{center}\n",
                    f"\\includegraphics[width=\\textwidth,height={image_height:.4f}\\textheight,keepaspectratio]{{{_pdf_path(pdf)}}}\\par\n",
                    f"{{\\footnotesize Pairwise merge stage {stage}}}\n",
                    "\\end{center}\n",
                    "\\vfill\n",
                ]
            )
    lines.append("\\end{document}\n")
    return "".join(lines)


def _pdf_path(path: Path) -> str:
    return "\\detokenize{" + str(path.resolve()) + "}"


def _escape(value: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    return "".join(replacements.get(character, character) for character in value)
