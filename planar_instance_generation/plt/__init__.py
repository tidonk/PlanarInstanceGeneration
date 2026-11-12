"""Publication-ready graph output."""

from .boxplot import BoxplotSeries, BoxplotSummary, write_boxplot
from .ecdf import EcdfSeries, write_ecdf
from .latex import (
    LatexCompilationError,
    PlotPdfArtifacts,
    compile_plot_pdf,
)
from .paired_scatter import PairedScatterPoint, write_paired_scatter
from .report import (
    ReviewBoundRow,
    ReviewFigure,
    ReviewInstance,
    ReviewReportArtifacts,
    write_a4_review_report,
)
from .tikz import (
    MergePlotArtifacts,
    PlotArtifacts,
    PlotConfig,
    SourceStyle,
    TIKZ_FORMAT_VERSION,
    kamada_kawai_positions,
    make_source_styles,
    write_graph_plot,
    write_merge_process,
)

__all__ = [
    "BoxplotSeries",
    "BoxplotSummary",
    "EcdfSeries",
    "LatexCompilationError",
    "MergePlotArtifacts",
    "PairedScatterPoint",
    "PlotArtifacts",
    "PlotConfig",
    "PlotPdfArtifacts",
    "ReviewFigure",
    "ReviewBoundRow",
    "ReviewInstance",
    "ReviewReportArtifacts",
    "SourceStyle",
    "TIKZ_FORMAT_VERSION",
    "compile_plot_pdf",
    "kamada_kawai_positions",
    "make_source_styles",
    "write_graph_plot",
    "write_boxplot",
    "write_ecdf",
    "write_paired_scatter",
    "write_a4_review_report",
    "write_merge_process",
]
