# Third-party notices

rfq-bench is MIT-licensed (see [`LICENSE`](./LICENSE)). It redistributes the following
third-party code:

| Component | Version | License | Location |
|---|---|---|---|
| [uPlot](https://github.com/leeoniya/uPlot) by Leon Sorokin | 1.6.31 | MIT | `src/rfq_bench/report/templates/vendor/` (license text: `LICENSE-uPlot`) |

uPlot is inlined into every generated dashboard HTML file; its copyright notice is kept in
the script header.

Python dependencies (NegMAS, pydantic, pydantic-settings, openai, numpy, typer) are
installed from PyPI under their own licenses and are not redistributed here.

The optional Laya agent talks to a separately installed Laya server (Laya by Convai
Innovations, Apache-2.0). No Laya code or weights are included in this repository.
