"""Black-and-white patent-style drawings for the JalMarg invention disclosure.

Run: python disclosure_figures/make_figures.py   -> writes fig1.png ... fig5.png next to this file.
Rules followed: line drawings only, no dimensions/numbers marked, text only as labels
in flow diagrams, symbols and reference numerals elsewhere.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "hatch.linewidth": 0.6})
K = "black"
DPI = 300


def box(ax, x, y, w, h, text, num=None, fs=10.5, rounded=True):
    style = "round,pad=0.02,rounding_size=0.08" if rounded else "square,pad=0.02"
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style, fc="white", ec=K, lw=1.3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, wrap=True)
    if num is not None:
        ax.text(x + w / 2 - 0.04, y + h / 2 + 0.03, str(num), ha="right", va="bottom", fontsize=9)


def arrow(ax, p, q, text=None, rad=0.0, off=(0, 0), fs=9.5, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=1.2, color=K,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=2, shrinkB=2))
    if text:
        mx, my = (p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1]
        ax.text(mx, my, text, ha="center", va="center", fontsize=fs,
                bbox=dict(fc="white", ec="none", pad=1.0))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------- Fig. 1 pipeline
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 9.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 13.6); ax.axis("off")
    W, H = 6.2, 0.95
    cx = 5.0
    steps = [
        (12.7, "Image / video input\n(fixed camera, vehicle camera, citizen photo)", 100),
        (11.2, "Camera source adapter\n(frame + road segment + timestamp)", 102),
        (9.7, "Segmentation of water surface\nand reference objects", 104),
        (8.2, "Depth per reference object and\nuncertainty-weighted fusion  (h ± σ)", 106),
        (6.7, "Per-segment state estimator\n(depth, rate of change, segment state)", 108),
        (5.2, "Physics-constrained neural\nrecession forecaster (depth quantiles)", 110),
        (3.7, "Vehicle-class passability evaluator\n(time-to-passable per vehicle class)", 112),
        (2.2, "Time-dependent router\n(wait-then-go vs. detour)", 114),
        (0.7, "Route and advisory output", 116),
    ]
    for y, t, n in steps:
        box(ax, cx, y, W, H, t, n)
    for (y1, _, _), (y2, _, _) in zip(steps, steps[1:]):
        arrow(ax, (cx, y1 - H / 2), (cx, y2 + H / 2))
    # side inputs
    box(ax, 9.05, 5.2, 1.7, 0.95, "Rainfall\ndata", 118, fs=9.5)
    arrow(ax, (8.2, 5.2), (cx + W / 2, 5.2))
    box(ax, 0.95, 3.7, 1.7, 0.95, "Vehicle\nprofile", 120, fs=9.5)
    arrow(ax, (1.8, 3.7), (cx - W / 2, 3.7))
    box(ax, 0.95, 2.2, 1.7, 0.95, "Road\ngraph", 122, fs=9.5)
    arrow(ax, (1.8, 2.2), (cx - W / 2, 2.2))
    box(ax, 9.05, 6.7, 1.7, 0.95, "Operator\noverride", 124, fs=9.5)
    arrow(ax, (8.2, 6.7), (cx + W / 2, 6.7))
    save(fig, "fig1.png")


# --------------------------------------------------------------- Fig. 2 wheel geometry
def fig2():
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.set_xlim(-4.2, 4.6); ax.set_ylim(-0.6, 5.6); ax.set_aspect("equal"); ax.axis("off")
    road_y = 0.0
    cx, cy, D = 0.0, 2.2, 4.0  # wheel centre and diameter (drawing units only)
    water = 1.45  # waterline height above road
    # vehicle body outline (partial)
    ax.plot([-4.0, -2.7, -2.3, 2.3, 2.7, 4.0], [3.0, 3.0, 4.3, 4.3, 3.0, 3.0], color=K, lw=1.2)
    # wheel: tyre + rim fitted ellipse
    ax.add_patch(Ellipse((cx, cy), D * 0.92, D, fc="white", ec=K, lw=2.0))
    ax.add_patch(Ellipse((cx, cy), D * 0.5, D * 0.55, fc="none", ec=K, lw=1.0))
    ax.add_patch(Ellipse((cx, cy), D * 0.98, D * 1.06, fc="none", ec=K, lw=0.9, ls=(0, (4, 3))))
    # water body (hatched) clipped below waterline
    ax.add_patch(Rectangle((-4.0, road_y), 8.0, water, fc="none", ec=K, lw=0, hatch="---", zorder=3))
    ax.plot([-4.0, 4.0], [water, water], color=K, lw=1.6, zorder=4)
    # road
    ax.plot([-4.0, 4.0], [road_y, road_y], color=K, lw=2.2)
    ax.add_patch(Rectangle((-4.0, road_y - 0.35), 8.0, 0.35, fc="none", ec=K, lw=0, hatch="////"))
    # dimension-style symbol markers (no values)
    x1 = 3.0
    ax.annotate("", xy=(x1, water), xytext=(x1, road_y), arrowprops=dict(arrowstyle="<->", lw=1.1))
    ax.text(x1 + 0.18, water / 2, "h", fontsize=14, style="italic", va="center")
    x2 = -2.75
    ax.annotate("", xy=(x2, cy + D / 2), xytext=(x2, cy - D / 2), arrowprops=dict(arrowstyle="<->", lw=1.1))
    ax.plot([x2 - 0.1, cx], [cy + D / 2, cy + D / 2], color=K, lw=0.6, ls=":")
    ax.text(x2 - 0.25, cy + 1.2, "D", fontsize=14, style="italic", ha="right", va="center")
    # reference numerals with leader lines
    def num(n, xy, txt_xy):
        ax.annotate(str(n), xy=xy, xytext=txt_xy, fontsize=10,
                    arrowprops=dict(arrowstyle="-", lw=0.7, color=K))
    num(200, (cx + 1.7, cy + 1.0), (3.4, 4.0))     # wheel
    num(202, (cx + 1.9, cy + 1.55), (3.4, 5.0))    # fitted ellipse
    num(204, (-1.2, water), (-3.95, 2.0))           # waterline
    num(206, (-2.0, 0.7), (-3.95, 0.45))               # water
    num(208, (-1.5, -0.2), (-1.3, -0.55))          # road
    num(210, (-3.4, 3.0), (-3.7, 3.6))             # vehicle body
    save(fig, "fig2.png")


# --------------------------------------------------------------- Fig. 3 forecast band
def fig3():
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    t = np.linspace(0, 10, 400)
    now = 2.0
    hist = t <= now
    h50 = np.where(t < 1.4, 2.2 + 3.6 * t / 1.4, 0.6 + (5.8 - 0.6) * np.exp(-0.42 * (t - 1.4)))
    spread = np.where(hist, 0.0, 0.25 + 0.08 * (t - now))
    q90, q10 = h50 + spread, np.maximum(h50 - spread, 0)
    rng = np.random.default_rng(3)
    to = np.linspace(0.2, now, 8)
    ho = np.interp(to, t, h50) + rng.normal(0, 0.18, to.size)
    ax.plot(to, ho, "o", ms=5, mfc="white", mec=K)
    ax.plot(t[hist], h50[hist], color=K, lw=1.4)
    f = ~hist
    ax.plot(t[f], h50[f], color=K, lw=1.6)
    ax.plot(t[f], q90[f], color=K, lw=1.0, ls="--")
    ax.plot(t[f], q10[f], color=K, lw=1.0, ls="--")
    ax.fill_between(t[f], q10[f], q90[f], facecolor="none", edgecolor=K, hatch="....", lw=0)
    ax.axvline(now, color=K, lw=0.9, ls=":")
    ax.text(now, 7.15, "now", ha="center", fontsize=10)
    for d, lab in ((4.3, "d₃"), (3.0, "d₂"), (2.0, "d₁")):
        ax.axhline(d, color=K, lw=1.0, ls="-.")
        ax.text(10.15, d, lab, va="center", fontsize=12, style="italic")
        idx = np.where(f & (q90 < d))[0]
        if idx.size:
            ts = t[idx[0]]
            ax.plot([ts, ts], [0, d], color=K, lw=0.8, ls=":")
            ax.plot(ts, d, "s", ms=5, color=K)
            ax.text(ts, -0.45, {"d₃": "t*₃", "d₂": "t*₂", "d₁": "t*₁"}[lab],
                    ha="center", fontsize=11, style="italic")
    for n, xy, txy in ((300, (1.6, ho[5] if False else np.interp(1.6, to, ho)), (0.6, 6.9)),
                       (302, (7.5, np.interp(7.5, t, h50)), (7.8, 3.6)),
                       (304, (4.5, np.interp(4.5, t, q90)), (4.8, 5.6)),
                       (306, (8.8, 0.9), (9.2, 1.6))):
        ax.annotate(str(n), xy=xy, xytext=txy, fontsize=10, arrowprops=dict(arrowstyle="-", lw=0.7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("t", fontsize=13, style="italic", labelpad=14)
    ax.set_ylabel("h", fontsize=13, style="italic", rotation=0, labelpad=12)
    save(fig, "fig3.png")


# --------------------------------------------------------------- Fig. 4 state machine
def fig4():
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 9.2); ax.axis("off")
    P = {"Dry": (1.6, 7.4), "Wet": (5.0, 7.4), "Rising": (8.4, 7.4), "Receding": (8.4, 4.0), "Unknown": (2.5, 1.1)}
    nums = {"Dry": 400, "Wet": 402, "Rising": 404, "Receding": 406, "Unknown": 408}
    w, h = 2.1, 0.95
    for k, (x, y) in P.items():
        box(ax, x, y, w, h, k, nums[k], fs=11.5)

    def e(a, side):
        x, y = P[a]
        return {"r": (x + w / 2, y), "l": (x - w / 2, y), "t": (x, y + h / 2), "b": (x, y - h / 2)}[side]

    arrow(ax, (e("Dry", "r")[0], 7.6), (e("Wet", "l")[0], 7.6), "rain > r\u2080", off=(0, 0.55))
    arrow(ax, (e("Wet", "l")[0], 7.2), (e("Dry", "r")[0], 7.2), "dry for\nset time", off=(0, -0.75))
    arrow(ax, e("Wet", "r"), e("Rising", "l"), "h > threshold", off=(0, -0.55))
    arrow(ax, (8.1, e("Rising", "b")[1]), (8.1, e("Receding", "t")[1]), "dh/dt < 0\nsustained", off=(-1.1, 0))
    arrow(ax, (8.7, e("Receding", "t")[1]), (8.7, e("Rising", "b")[1]), "rain\nresumes", off=(0.6, 0))
    arrow(ax, e("Receding", "l"), (5.0, e("Wet", "b")[1]), "h < threshold", rad=-0.25, off=(-0.9, -0.5))
    # dashed frame 410: any monitored state
    ax.add_patch(FancyBboxPatch((0.2, 3.15), 9.75, 5.55, boxstyle="round,pad=0.05", fc="none", ec=K, lw=0.9,
                                ls=(0, (5, 4))))
    ax.text(9.9, 8.85, "410", ha="right", fontsize=9)
    arrow(ax, (1.8, 3.1), (1.8, e("Unknown", "t")[1]), "no observation\nfor set time", off=(-0.05, 0.15))
    arrow(ax, (3.2, e("Unknown", "t")[1]), (3.2, 3.1), "new\nobservation", off=(1.0, 0.15))
    save(fig, "fig4.png")


# --------------------------------------------------------------- Fig. 5 routing example
def fig5():
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(1.2, 7.6); ax.set_aspect("equal"); ax.axis("off")
    N = {"O": (1.0, 3.6), "A": (3.4, 3.6), "B": (6.6, 3.6), "D": (9.0, 3.6),
         "C": (3.4, 6.6), "E": (6.6, 6.6), "W": (3.4, 3.6)}
    # all roads (thin)
    for a, b in (("O", "A"), ("A", "B"), ("B", "D"), ("A", "C"), ("C", "E"), ("E", "B")):
        ax.plot(*zip(N[a], N[b]), color=K, lw=0.8)
    # flooded segment A-B (underpass): hatched band
    ax.add_patch(Polygon([(3.7, 3.25), (6.3, 3.25), (6.3, 3.95), (3.7, 3.95)], fc="none", ec=K, hatch="xxxx", lw=0.8))
    # wait-then-go route: bold solid along O-A-B-D, offset below
    y = 3.0
    ax.add_patch(FancyArrowPatch((1.0, y), (9.0, y), arrowstyle="-|>", mutation_scale=14, lw=2.2, color=K))
    ax.text(5.0, 2.45, "wait until t*, then go", ha="center", fontsize=10.5)
    # detour route: dashed via C-E
    pts = [(1.0, 4.15), (2.85, 4.15), (2.85, 7.15), (7.15, 7.15), (7.15, 4.15), (9.0, 4.15)]
    for p, q in zip(pts, pts[1:-1]):
        ax.plot(*zip(p, q), color=K, lw=1.8, ls=(0, (5, 3)))
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=14, lw=1.8, color=K, linestyle=(0, (5, 3))))
    ax.text(5.0, 7.45, "detour, go now", ha="center", fontsize=10.5)
    # nodes
    for k in ("O", "A", "B", "D", "C", "E"):
        x, yv = N[k]
        ax.add_patch(plt.Circle((x, yv), 0.22, fc="white", ec=K, lw=1.4, zorder=5))
        if k in ("O", "D"):
            ax.text(x, yv, k, ha="center", va="center", fontsize=10.5, zorder=6, weight="bold")
    # waiting point marker at A
    ax.add_patch(Rectangle((3.15, 3.15), 0.5, 0.0, fc="none"))
    for n, xy, txy in ((500, (5.0, 3.6), (5.0, 5.2)), (502, (8.2, 3.0), (8.4, 1.6)),
                       (504, (7.15, 5.6), (8.2, 5.8)), (506, (3.4, 3.6), (3.0, 1.6))):
        ax.annotate(str(n), xy=xy, xytext=txy, fontsize=10, arrowprops=dict(arrowstyle="-", lw=0.7))
    save(fig, "fig5.png")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5):
        f()
    print("written to", OUT)
