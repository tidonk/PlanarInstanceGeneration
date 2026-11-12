from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from planar_instance_generation.plt import (
    LatexCompilationError,
    PlotArtifacts,
    compile_plot_pdf,
)


def _plot(root: Path) -> PlotArtifacts:
    plot = PlotArtifacts(
        tikz=root / "network.tikz",
        standalone_tex=root / "network.tex",
        coordinates_json=root / "network.json",
    )
    plot.tikz.write_text("\\begin{tikzpicture}\\end{tikzpicture}\n", encoding="utf-8")
    plot.standalone_tex.write_text(
        "\\documentclass{standalone}\\begin{document}"
        "\\input{\\jobname.tikz}\\end{document}\n",
        encoding="utf-8",
    )
    plot.coordinates_json.write_text("{}\n", encoding="utf-8")
    return plot


def _successful_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
    output_argument = next(value for value in command if value.startswith("-output-directory="))
    build_directory = Path(output_argument.split("=", 1)[1])
    (build_directory / "network.pdf").write_bytes(b"%PDF-1.7\nmock\n")
    (build_directory / "network.log").write_text("native log\n", encoding="utf-8")
    return subprocess.CompletedProcess(command, 0, stdout="pdflatex output\n")


class LatexCompileTests(unittest.TestCase):
    @patch("planar_instance_generation.plt.latex.shutil.which", return_value="/usr/bin/pdflatex")
    @patch("planar_instance_generation.plt.latex.subprocess.run", side_effect=_successful_run)
    def test_places_pdf_and_logs_without_intermediate_files(self, run, _which) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            plot = _plot(source)
            output = root / "plots" / "full.pdf"
            logs = root / "logs" / "full"

            result = compile_plot_pdf(plot, output_pdf=output, log_directory=logs)

            self.assertEqual(result.pdf.read_bytes(), b"%PDF-1.7\nmock\n")
            self.assertEqual(result.stdout_log.read_text(), "pdflatex output\n")
            self.assertEqual(result.latex_log.read_text(), "native log\n")
            self.assertFalse(result.reused)
            self.assertFalse((source / "network.aux").exists())
            arguments, keywords = run.call_args
            self.assertIn("-halt-on-error", arguments[0])
            self.assertEqual(keywords["cwd"], source)
            self.assertEqual(keywords["env"]["extra_mem_top"], "12000000")
            self.assertEqual(keywords["timeout"], 600.0)

    @patch("planar_instance_generation.plt.latex.shutil.which", return_value="/usr/bin/pdflatex")
    @patch("planar_instance_generation.plt.latex.subprocess.run", side_effect=_successful_run)
    def test_reuses_only_a_complete_artifact_set(self, run, _which) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plot = _plot(root)
            output = root / "rendered.pdf"
            logs = root / "logs"
            compile_plot_pdf(plot, output_pdf=output, log_directory=logs)
            cached = compile_plot_pdf(plot, output_pdf=output, log_directory=logs)

            self.assertTrue(cached.reused)
            self.assertEqual(run.call_count, 1)

            cached.latex_log.unlink()
            with self.assertRaisesRegex(FileExistsError, "incomplete"):
                compile_plot_pdf(plot, output_pdf=output, log_directory=logs)

            rebuilt = compile_plot_pdf(
                plot,
                output_pdf=output,
                log_directory=logs,
                overwrite=True,
            )
            self.assertFalse(rebuilt.reused)
            self.assertTrue(rebuilt.latex_log.is_file())
            self.assertEqual(run.call_count, 2)

    @patch("planar_instance_generation.plt.latex.shutil.which", return_value="/usr/bin/pdflatex")
    @patch("planar_instance_generation.plt.latex.subprocess.run")
    def test_failure_keeps_diagnostics_and_reports_output_tail(self, run, _which) -> None:
        def fail(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            output_argument = next(
                value for value in command if value.startswith("-output-directory=")
            )
            build_directory = Path(output_argument.split("=", 1)[1])
            (build_directory / "network.log").write_text("TeX detail\n", encoding="utf-8")
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="first line\nUndefined control sequence\n",
            )

        run.side_effect = fail
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plot = _plot(root)
            with self.assertRaisesRegex(
                LatexCompilationError, "Undefined control sequence"
            ):
                compile_plot_pdf(
                    plot,
                    output_pdf=root / "failed.pdf",
                    log_directory=root / "logs",
                )

            self.assertEqual(
                (root / "logs" / "failed.stdout.log").read_text(),
                "first line\nUndefined control sequence\n",
            )
            self.assertEqual(
                (root / "logs" / "failed.latex.log").read_text(), "TeX detail\n"
            )
            self.assertFalse((root / "failed.pdf").exists())

    @patch("planar_instance_generation.plt.latex.shutil.which", return_value=None)
    def test_requires_pdflatex(self, _which) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(LatexCompilationError, "not found"):
                compile_plot_pdf(
                    _plot(root),
                    output_pdf=root / "plot.pdf",
                    log_directory=root / "logs",
                )


if __name__ == "__main__":
    unittest.main()
