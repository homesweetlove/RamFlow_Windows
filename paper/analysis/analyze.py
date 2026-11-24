#!/usr/bin/env python3
"""Recompute every number and figure in the RamFlow paper from the CSV files
that are committed under docs/evidence/.

Usage (from the repository root):
    python3 paper/analysis/analyze.py

Outputs:
    paper/analysis/stats.json        all derived statistics
    paper/shared/numbers.tex         LaTeX macros used by both papers
    paper/figures/*.pdf              figures (Korean "-ko" and English "-en")

Only the Python standard library, numpy, scipy and matplotlib are used.
Nothing here edits the evidence files.
"""
from __future__ import annotations

import csv
import json
import math
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs" / "evidence"
OUT = ROOT / "paper"
FIG = OUT / "figures"
SHARED = OUT / "shared"
FIG.mkdir(parents=True, exist_ok=True)
SHARED.mkdir(parents=True, exist_ok=True)


def read(name: str) -> list[dict]:
    with open(EVIDENCE / name / "trials.csv", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- statistics
def paired(off: list[float], on: list[float]) -> dict:
    """Paired comparison over matched blocks (n = number of blocks)."""
    off_a, on_a = np.asarray(off, float), np.asarray(on, float)
    n = len(off_a)
    reduction = 100 * (1 - on_a / off_a)              # % reduction per block
    log_ratio = np.log(on_a / off_a)
    t = stats.t.ppf(0.975, n - 1)
    se = log_ratio.std(ddof=1) / math.sqrt(n)
    gm = math.exp(log_ratio.mean())
    improved = int((on_a < off_a).sum())
    # exact one-sided sign test: P(X >= improved | p = 0.5)
    sign_p = float(stats.binomtest(improved, n, 0.5, alternative="greater").pvalue)
    # exact Wilcoxon signed-rank (two-sided); with n = 5 the minimum is 0.0625
    wil_p = float(stats.wilcoxon(on_a, off_a, alternative="two-sided", method="exact").pvalue)
    # order-statistic confidence level of [min, max] for the median, n blocks
    ci_level = 1 - 2 * 0.5 ** n
    return {
        "n": n,
        "off_median": float(np.median(off_a)),
        "on_median": float(np.median(on_a)),
        "reduction_of_medians_pct": float(100 * (1 - np.median(on_a) / np.median(off_a))),
        "block_reduction_pct": [round(float(x), 2) for x in reduction],
        "block_reduction_median_pct": float(np.median(reduction)),
        "block_reduction_min_pct": float(reduction.min()),
        "block_reduction_max_pct": float(reduction.max()),
        "median_ci_level": ci_level,
        "geo_mean_ratio": gm,
        "geo_mean_reduction_pct": 100 * (1 - gm),
        "geo_mean_reduction_ci95_pct": [100 * (1 - math.exp(log_ratio.mean() + t * se)),
                                        100 * (1 - math.exp(log_ratio.mean() - t * se))],
        "improved_blocks": improved,
        "sign_test_one_sided_p": sign_p,
        "wilcoxon_exact_two_sided_p": wil_p,
    }


def by(rows, *keys):
    out = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


S: dict = {}

# ------------------------------------------------- E3: real Chrome / VS Code
real = read("2026-11-10-real-apps")
METRICS = [("chrome", "search"), ("chrome", "sort"), ("vscode", "completion"), ("vscode", "editSave")]
present = sorted({(r["app"], r["metric"]) for r in real})
METRICS = [m for m in METRICS if m in present] + [m for m in present if m not in METRICS]
S["real"] = {}
real_series = {}
for app, metric in METRICS:
    rows = [r for r in real if r["app"] == app and r["metric"] == metric]
    blocks = sorted({int(r["block"]) for r in rows})
    off = [float(next(r for r in rows if int(r["block"]) == b and r["condition"] == "off")["p95Ms"]) for b in blocks]
    on = [float(next(r for r in rows if int(r["block"]) == b and r["condition"] == "on")["p95Ms"]) for b in blocks]
    off50 = [float(next(r for r in rows if int(r["block"]) == b and r["condition"] == "off")["p50Ms"]) for b in blocks]
    on50 = [float(next(r for r in rows if int(r["block"]) == b and r["condition"] == "on")["p50Ms"]) for b in blocks]
    res = paired(off, on)
    res["p50"] = paired(off50, on50)
    S["real"][f"{app}-{metric}"] = res
    real_series[(app, metric)] = (blocks, off, on)

S["real_background"] = {}
real_bg = {}
for app in ("chrome", "vscode"):
    rows = [r for r in real if r["app"] == app]
    blocks = sorted({int(r["block"]) for r in rows})
    off = [float(next(r for r in rows if int(r["block"]) == b and r["condition"] == "off")["backgroundCyclesPerSecond"]) for b in blocks]
    on = [float(next(r for r in rows if int(r["block"]) == b and r["condition"] == "on")["backgroundCyclesPerSecond"]) for b in blocks]
    S["real_background"][app] = paired(off, on)
    real_bg[app] = (off, on)
    S["real_background"][app]["writes_on"] = sorted({int(r["writes"]) for r in rows if r["condition"] == "on"})
    S["real_background"][app]["available_before_range_mib"] = [
        min(float(r["availableBeforeMiB"]) for r in rows), max(float(r["availableBeforeMiB"]) for r in rows)]

# Session-level pooled sign test: each fresh app session is one independent unit;
# a session counts as "improved" only if every measured feature improved in it.
sessions = by(real, "app", "block")
improved_sessions = 0
for (app, block), rows in sessions.items():
    ok = True
    for metric in {r["metric"] for r in rows}:
        off = float(next(r for r in rows if r["metric"] == metric and r["condition"] == "off")["p95Ms"])
        on = float(next(r for r in rows if r["metric"] == metric and r["condition"] == "on")["p95Ms"])
        ok &= on < off
    improved_sessions += ok
n_sessions = len(sessions)
S["real_pooled"] = {
    "sessions": n_sessions,
    "sessions_all_features_improved": improved_sessions,
    "sign_test_two_sided_p": float(stats.binomtest(improved_sessions, n_sessions, 0.5).pvalue),
    "conditions": len({(r["app"], r["block"], r["condition"]) for r in real}),
    "tasks": int(sum(int(r["samples"]) for r in real)),
    "safety_released": sum(r["safetyReleased"] == "True" for r in real),
    "denied": sum(int(r["denied"]) for r in real),
    "restored_all": all(r["resourcesRestored"] == "True" for r in real),
}
# Chrome in-page event -> second rAF p95 (separates automation round-trip)
S["real_event_to_frame"] = {}
for metric in ("search", "sort"):
    rows = [r for r in real if r["app"] == "chrome" and r["metric"] == metric]
    blocks = sorted({int(r["block"]) for r in rows})
    off = [float(next(r for r in rows if int(r["block"]) == b and r["condition"] == "off")["eventToFrameP95Ms"]) for b in blocks]
    on = [float(next(r for r in rows if int(r["block"]) == b and r["condition"] == "on")["eventToFrameP95Ms"]) for b in blocks]
    S["real_event_to_frame"][metric] = paired(off, on)

# ------------------------------------------- E2: explicit background control
bgc = read("2026-11-09-background-control")
S["bgc"] = {}
bgc_series = {}
for regime, base, new in (("memory-mixed", "absent", "new"), ("cpu", "old", "new")):
    rows = [r for r in bgc if r["regime"] == regime]
    blocks = sorted({int(r["block"]) for r in rows})
    get = lambda b, c, k: float(next(r for r in rows if int(r["block"]) == b and r["condition"] == c)[k])
    res = paired([get(b, base, "p95Ms") for b in blocks], [get(b, new, "p95Ms") for b in blocks])
    res["p50"] = paired([get(b, base, "p50Ms") for b in blocks], [get(b, new, "p50Ms") for b in blocks])
    res["background"] = paired([get(b, base, "backgroundCyclesPerSecond") for b in blocks],
                               [get(b, new, "backgroundCyclesPerSecond") for b in blocks])
    res["off_p95_range"] = [min(get(b, base, "p95Ms") for b in blocks), max(get(b, base, "p95Ms") for b in blocks)]
    res["on_p95_range"] = [min(get(b, new, "p95Ms") for b in blocks), max(get(b, new, "p95Ms") for b in blocks)]
    res["writes_new"] = sorted({int(r["writes"]) for r in rows if r["condition"] == new})
    res["available_before_range_mib"] = [min(float(r["availableBeforeMiB"]) for r in rows),
                                         max(float(r["availableBeforeMiB"]) for r in rows)]
    res["digests_identical"] = len({r["digest"] for r in rows}) == 1
    res["requests"] = sum(int(r["requests"]) for r in rows)
    res["engine_private_mib_median"] = float(np.median([float(r["enginePrivateMiB"]) for r in rows if r["condition"] == new]))
    res["tray_private_mib_median"] = float(np.median([float(r["trayPrivateMiB"]) for r in rows if r["condition"] == new]))
    S["bgc"][regime] = res
    bgc_series[regime] = (blocks, [get(b, base, "p95Ms") for b in blocks], [get(b, new, "p95Ms") for b in blocks],
                          [get(b, base, "backgroundCyclesPerSecond") for b in blocks],
                          [get(b, new, "backgroundCyclesPerSecond") for b in blocks])

# ------------------------------------------------ E1: working-set scarcity
sc = read("2026-11-09-scarcity")
S["scarcity"] = {}
sc_series = {}
for cond in ("normal", "absent", "dry", "live"):
    vals = [float(r["requestMs"]) for r in sorted(sc, key=lambda r: int(r["block"])) if r["condition"] == cond]
    S["scarcity"][cond] = {"runs": vals, "median": float(np.median(vals))}
    sc_series[cond] = vals
S["scarcity"]["live_vs_absent_ms"] = S["scarcity"]["live"]["median"] - S["scarcity"]["absent"]["median"]
S["scarcity"]["live_vs_absent_pct"] = 100 * S["scarcity"]["live_vs_absent_ms"] / S["scarcity"]["absent"]["median"]
S["scarcity"]["trims"] = {f"block{r['block']}": int(r["trimsAfter"]) for r in sc if r["condition"] == "live"}
S["scarcity"]["before_mib"] = {f"{r['condition']}-{r['block']}": float(r["decisionMiB"]) for r in sc}

(OUT / "analysis" / "stats.json").write_text(json.dumps(S, indent=2, ensure_ascii=False), encoding="utf-8")

# ------------------------------------------------------------- LaTeX macros
macros = []
def m(name, value, fmt="{:.1f}"):
    text = fmt.format(value)
    if text.startswith("-"):
        text = "\u2212" + text[1:]          # typographic minus, never breaks like a hyphen
    macros.append(f"\\newcommand{{\\{name}}}{{{text}}}")

for key, tag in (("chrome-search", "CS"), ("chrome-sort", "CO"), ("vscode-completion", "VC"), ("vscode-editSave", "VS")):
    r = S["real"][key]
    m(f"num{tag}Off", r["off_median"], "{:.2f}"); m(f"num{tag}On", r["on_median"], "{:.2f}")
    m(f"num{tag}Red", r["reduction_of_medians_pct"]); m(f"num{tag}BlkMed", r["block_reduction_median_pct"])
    m(f"num{tag}BlkMin", r["block_reduction_min_pct"]); m(f"num{tag}BlkMax", r["block_reduction_max_pct"])
    m(f"num{tag}GmRed", r["geo_mean_reduction_pct"])
    m(f"num{tag}GmLo", r["geo_mean_reduction_ci95_pct"][0]); m(f"num{tag}GmHi", r["geo_mean_reduction_ci95_pct"][1])
    m(f"num{tag}OffFifty", r["p50"]["off_median"], "{:.2f}"); m(f"num{tag}OnFifty", r["p50"]["on_median"], "{:.2f}")
for app, tag in (("chrome", "BgC"), ("vscode", "BgV")):
    r = S["real_background"][app]
    m(f"num{tag}Off", r["off_median"], "{:,.0f}"); m(f"num{tag}On", r["on_median"], "{:,.0f}")
    m(f"num{tag}Red", r["reduction_of_medians_pct"])
for metric, tag in (("search", "EfS"), ("sort", "EfO")):
    r = S["real_event_to_frame"][metric]
    m(f"num{tag}Off", r["off_median"], "{:.2f}"); m(f"num{tag}On", r["on_median"], "{:.2f}")
p = S["real_pooled"]
m("numSessions", p["sessions"], "{:d}"); m("numSessionsOk", p["sessions_all_features_improved"], "{:d}")
m("numPooledP", p["sign_test_two_sided_p"], "{:.4f}"); m("numTasks", p["tasks"], "{:,d}")
m("numConditions", p["conditions"], "{:d}")
for regime, tag in (("memory-mixed", "Mx"), ("cpu", "Cp")):
    r = S["bgc"][regime]
    m(f"num{tag}Off", r["off_median"], "{:.2f}"); m(f"num{tag}On", r["on_median"], "{:.2f}")
    m(f"num{tag}Red", r["reduction_of_medians_pct"]); m(f"num{tag}BlkMin", r["block_reduction_min_pct"])
    m(f"num{tag}BlkMax", r["block_reduction_max_pct"])
    m(f"num{tag}GmLo", r["geo_mean_reduction_ci95_pct"][0]); m(f"num{tag}GmHi", r["geo_mean_reduction_ci95_pct"][1])
    m(f"num{tag}OffFifty", r["p50"]["off_median"], "{:.2f}"); m(f"num{tag}OnFifty", r["p50"]["on_median"], "{:.2f}")
    m(f"num{tag}FiftyRed", r["p50"]["reduction_of_medians_pct"])
    m(f"num{tag}BgRed", r["background"]["reduction_of_medians_pct"])
    m(f"num{tag}AvLo", r["available_before_range_mib"][0]); m(f"num{tag}AvHi", r["available_before_range_mib"][1])
for cond, tag in (("normal", "Amb"), ("absent", "Abs"), ("dry", "Dry"), ("live", "Live")):
    m(f"numSc{tag}", S["scarcity"][cond]["median"], "{:,.2f}")
m("numScDiff", S["scarcity"]["live_vs_absent_ms"], "{:.2f}"); m("numScDiffPct", S["scarcity"]["live_vs_absent_pct"], "{:.2f}")
m("numCiLevel", 100 * S["real"]["chrome-search"]["median_ci_level"], "{:.1f}")
m("numSignP", S["real"]["chrome-search"]["sign_test_one_sided_p"], "{:.4f}")
m("numWilP", S["real"]["chrome-search"]["wilcoxon_exact_two_sided_p"], "{:.4f}")
(SHARED / "numbers.tex").write_text("% Generated by paper/analysis/analyze.py -- do not edit by hand.\n" + "\n".join(macros) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OFF, ON = "#eb6834", "#2a78d6"          # validated pair (dataviz reference palette)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f or "NotoSansKR" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": ["Noto Sans CJK KR", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "axes.edgecolor": GRID, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.fonttype": "none", "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

L = {
    "ko": {"off": "미사용", "on": "사용", "p95": "블록별 p95 지연 (ms)",
           "chrome-search": "Chrome 검색 → 렌더링 경계", "chrome-sort": "Chrome 정렬 → 렌더링 경계",
           "vscode-completion": "VS Code 편집 → 코드 완성", "vscode-editSave": "VS Code 편집 → 저장",
           "bg": "배경 처리량 (천 조각/초)", "fg": "전경 p95 지연 (ms)",
           "mx": "메모리+CPU 경쟁\n(미실행 vs 수정 엔진)", "cp": "CPU 경쟁만\n(이전 엔진 vs 수정 엔진)",
           "t_mx": "통제 부하: 메모리+CPU", "t_c": "Chrome 검색", "t_v": "VS Code 코드 완성",
           "old": "이전 엔진", "new": "수정 엔진", "absent": "미실행",
           "normal": "여유 RAM\n(참고)", "absent2": "저메모리\n미실행", "dry": "저메모리\n관찰만", "live": "저메모리\n실제 관리",
           "ms": "완료 시간 (ms)", "blk": "블록", "med": "중앙값", "trim": "Working Set 축소 1회"},
    "en": {"off": "Without", "on": "With RamFlow", "p95": "Per-block p95 latency (ms)",
           "chrome-search": "Chrome search → render boundary", "chrome-sort": "Chrome sort → render boundary",
           "vscode-completion": "VS Code edit → completion", "vscode-editSave": "VS Code edit → save",
           "bg": "Background throughput (k chunks/s)", "fg": "Foreground p95 latency (ms)",
           "mx": "Memory + CPU contention\n(absent vs revised engine)", "cp": "CPU contention only\n(previous vs revised engine)",
           "t_mx": "Controlled load: memory + CPU", "t_c": "Chrome search", "t_v": "VS Code completion",
           "old": "Previous", "new": "Revised", "absent": "Absent",
           "normal": "Ample\n(ref.)", "absent2": "Low mem.\nabsent", "dry": "Low mem.\nobserve", "live": "Low mem.\nlive",
           "ms": "Completion time (ms)", "blk": "Block", "med": "median", "trim": "one working-set trim"},
}


def slope(ax, off, on, a_lbl, b_lbl, title, ylabel=None):
    xs = (0, 1)
    for o, n in zip(off, on):
        ax.plot(xs, (o, n), color="#b9b8b2", lw=0.9, zorder=1)
    ax.scatter([0] * len(off), off, s=22, color=OFF, edgecolor="#fcfcfb", linewidth=1.2, zorder=3)
    ax.scatter([1] * len(on), on, s=22, color=ON, edgecolor="#fcfcfb", linewidth=1.2, zorder=3)
    mo, mn = float(np.median(off)), float(np.median(on))
    ax.plot(xs, (mo, mn), color=INK, lw=2, zorder=2, solid_capstyle="round")
    ax.text(-0.09, mo, f"{mo:.1f}", ha="right", va="center", color=INK, fontsize=7.5)
    ax.text(1.09, mn, f"{mn:.1f}", ha="left", va="center", color=INK, fontsize=7.5)
    ax.set_xticks(xs, [a_lbl, b_lbl]); ax.set_xlim(-0.55, 1.55)
    ax.set_title(title, color=INK, loc="left", pad=6)
    ax.set_ylim(0, max(off) * 1.12)
    if ylabel: ax.set_ylabel(ylabel)


for lang, T in L.items():
    # Fig: real apps (4 panels)
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.3), sharey=False)
    axes = axes.ravel()
    for ax, (app, metric) in zip(axes, METRICS):
        _, off, on = real_series[(app, metric)]
        slope(ax, off, on, T["off"], T["on"], T[f"{app}-{metric}"], T["p95"] if ax in (axes[0], axes[2]) else None)
    fig.text(0.995, -0.01, ("회색 선: 같은 앱 세션(블록) 5개의 짝 · 검은 선: 중앙값" if lang == "ko" else
                            "Grey lines: five paired app sessions (blocks) · black line: median"),
             ha="right", color=INK2, fontsize=7)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(FIG / f"real-apps-{lang}.pdf"); fig.savefig(FIG / f"real-apps-{lang}.png", dpi=220); plt.close(fig)

    # Fig: trade-off (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.4))
    panels = [(T["t_mx"], bgc_series["memory-mixed"][3], bgc_series["memory-mixed"][4],
               bgc_series["memory-mixed"][1], bgc_series["memory-mixed"][2], T["absent"], T["new"]),
              (T["t_c"], real_bg["chrome"][0], real_bg["chrome"][1],
               real_series[("chrome", "search")][1], real_series[("chrome", "search")][2], T["off"], T["on"]),
              (T["t_v"], real_bg["vscode"][0], real_bg["vscode"][1],
               real_series[("vscode", "completion")][1], real_series[("vscode", "completion")][2], T["off"], T["on"])]
    for ax, (title, bo, bn, fo, fn, lo, ln) in zip(axes, panels):
        bo, bn = np.asarray(bo) / 1000, np.asarray(bn) / 1000
        for x0, y0, x1, y1 in zip(bo, fo, bn, fn):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color="#b9b8b2", lw=0.8, shrinkA=4, shrinkB=4, mutation_scale=7))
        ax.scatter(bo, fo, s=22, color=OFF, edgecolor="#fcfcfb", linewidth=1.2, zorder=3, label=lo)
        ax.scatter(bn, fn, s=22, color=ON, edgecolor="#fcfcfb", linewidth=1.2, zorder=3, label=ln)
        ax.set_title(title, color=INK, loc="left", pad=6)
        ax.set_xlabel(T["bg"]); ax.grid(True, axis="both")
        ax.set_xlim(0, max(bo) * 1.15); ax.set_ylim(0, max(fo) * 1.18)
        if ax is axes[0]: ax.set_ylabel(T["fg"])
        ax.legend(frameon=False, fontsize=7, loc="upper left", handletextpad=0.2, borderaxespad=0.2)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(FIG / f"tradeoff-{lang}.pdf"); fig.savefig(FIG / f"tradeoff-{lang}.png", dpi=220); plt.close(fig)

    # Fig: background-control study (2 panels)
    fig, axes = plt.subplots(1, 2, figsize=(4.16, 2.35))
    slope(axes[0], bgc_series["memory-mixed"][1], bgc_series["memory-mixed"][2], T["absent"], T["new"], T["mx"], T["p95"])
    slope(axes[1], bgc_series["cpu"][1], bgc_series["cpu"][2], T["old"], T["new"], T["cp"])
    for ax in axes: ax.set_ylim(0, 260)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIG / f"background-control-{lang}.pdf"); fig.savefig(FIG / f"background-control-{lang}.png", dpi=220); plt.close(fig)

    # Fig: scarcity study (dot plot)
    fig, ax = plt.subplots(figsize=(2.95, 2.25))
    order = ["normal", "absent", "dry", "live"]; labels = [T["normal"], T["absent2"], T["dry"], T["live"]]
    colors = {"normal": "#9b9a94", "absent": OFF, "dry": "#9b9a94", "live": ON}
    for i, c in enumerate(order):
        vals = sc_series[c]
        ax.scatter([i - 0.08, i + 0.08], vals, s=22, color=colors[c], edgecolor="#fcfcfb", linewidth=1.2, zorder=3)
        med = float(np.median(vals))
        ax.plot([i - 0.25, i + 0.25], [med, med], color=INK, lw=1.6, solid_capstyle="round")
        ax.text(i, med + 32, f"{med:,.0f}", ha="center", va="bottom", fontsize=7, color=INK)
    trimmed = [r for r in sc if r["condition"] == "live" and int(r["trimsAfter"]) > 0]
    for r in trimmed:
        y = float(r["requestMs"]); x = 3 - 0.08 if int(r["block"]) == 1 else 3 + 0.08
        ax.annotate(T["trim"], xy=(x, y), xytext=(2.15, 1700), fontsize=6.5, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.set_xticks(range(4), labels, fontsize=6.6); ax.set_ylim(1600, 2000); ax.set_ylabel(T["ms"])
    ax.set_xlim(-0.5, 3.5)
    fig.tight_layout()
    fig.savefig(FIG / f"scarcity-{lang}.pdf"); fig.savefig(FIG / f"scarcity-{lang}.png", dpi=220); plt.close(fig)

print(json.dumps({k: S[k] for k in ("real_pooled",)}, indent=1))
for k, v in S["real"].items():
    print(k, round(v["off_median"], 2), round(v["on_median"], 2), round(v["reduction_of_medians_pct"], 1),
          "blocks", v["block_reduction_pct"], "gm-CI", [round(x, 1) for x in v["geo_mean_reduction_ci95_pct"]],
          "sign", round(v["sign_test_one_sided_p"], 4), "wil", v["wilcoxon_exact_two_sided_p"])
for k, v in S["bgc"].items():
    print(k, round(v["off_median"], 2), round(v["on_median"], 2), round(v["reduction_of_medians_pct"], 1),
          v["block_reduction_pct"], "p50", round(v["p50"]["off_median"], 2), round(v["p50"]["on_median"], 2),
          "bg", round(v["background"]["reduction_of_medians_pct"], 1), "eng", v["engine_private_mib_median"], v["tray_private_mib_median"])
for k, v in S["real_background"].items():
    print(k, round(v["off_median"]), round(v["on_median"]), round(v["reduction_of_medians_pct"], 1))
print("scarcity", {k: S["scarcity"][k]["median"] for k in ("normal", "absent", "dry", "live")}, S["scarcity"]["live_vs_absent_pct"])
print("e2f", {k: (v["off_median"], v["on_median"]) for k, v in S["real_event_to_frame"].items()})
