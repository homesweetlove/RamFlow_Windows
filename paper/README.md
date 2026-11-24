# RamFlow paper / RamFlow 논문

| | PDF | Source |
|---|---|---|
| 한국어 | [RamFlow-paper-ko.pdf](ko/RamFlow-paper-ko.pdf) | [ko/main.tex](ko/main.tex) |
| English | [RamFlow-paper-en.pdf](en/RamFlow-paper-en.pdf) | [en/main.tex](en/main.tex) |

**메모리를 비우는 대신 현재 작업을 지키는 보수적·가역적 Windows 자원 조정: 설계와 단일 PC 실측 연구**
*Protecting the Task in Use Instead of Emptying Memory: Conservative, Reversible Resource Orchestration on Windows and a Single-PC Measurement Study*

## 다시 만들기 / Rebuild

```bash
python3 paper/analysis/analyze.py        # numbers.tex, stats.json, figures/*
cd paper/ko && xelatex -jobname=RamFlow-paper-ko main && xelatex -jobname=RamFlow-paper-ko main
cd ../en && xelatex -jobname=RamFlow-paper-en main && xelatex -jobname=RamFlow-paper-en main
```

본문의 모든 수치와 그림은 `docs/evidence/*/trials.csv`에서 `analysis/analyze.py`가 직접 계산합니다. 수치를 손으로 고치지 마세요.
Every number and figure is computed from `docs/evidence/*/trials.csv` by `analysis/analyze.py`; do not edit `shared/numbers.tex` by hand.

Requirements: Python 3 (numpy, scipy, matplotlib), XeLaTeX, Noto CJK KR fonts, TeX Gyre Pagella.

```
paper/
├── ko/, en/        main.tex + built PDF
├── shared/         preamble, TikZ diagrams, references, generated numbers.tex
├── figures/        generated charts (PDF for print, PNG for preview)
└── analysis/       analyze.py, stats.json
```
