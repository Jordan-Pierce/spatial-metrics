"""
visualize_mining_scenarios
==========================

Drop-in addition to SpatialMetricsVisualizer.

Adds one public method:

    visualize_mining_scenarios(vehicle_width_m, buffer_m, output_name)

The method auto-generates all four mining paths from the semantic mask
and saves FIVE figures:

    1. {name}_baseline.png       – Dumb Centre-Cut
    2. {name}_profit.png         – Profit Maximizer (Greedy)
    3. {name}_terrain.png        – Terrain-Aware (Passability)
    4. {name}_eco.png            – Eco-Optimised (Compromise)
    5. {name}_summary.png        – 2×2 comparison + grouped bar chart

The four individual figures follow the project's "clean / overlay"
aesthetic: orthomosaic background (if available), semantic polygon
outlines, the mining rectangle coloured by scenario, casualties
highlighted, and a per-scenario metrics strip below the map.

The summary figure tiles all four scenarios in a 2×2 grid and adds
a grouped-bar comparison across three KPIs.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as mticker
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ── Semantic mask class IDs (must match SpatialMetricsAnalyzer convention) ──
_CLS_BG          = 0
_CLS_SUBSTRATE   = 1
_CLS_NODULE      = 2
_CLS_ORGANISM    = 3
_CLS_OBSTRUCTION = 4

# ── Visual identity for each scenario ────────────────────────────────────────
_SCENARIO_STYLE = {
    'baseline': dict(
        label='Dumb Baseline',
        subtitle='Centre-Cut',
        face='#94a3b8',
        edge='#cbd5e1',
        desc=(
            'Straight pass down the map centre.  Ignores topology, biology '
            'and nodule density — acts as the control variable.'
        ),
    ),
    'profit': dict(
        label='Profit Maximizer',
        subtitle='Greedy Sweep',
        face='#f59e0b',
        edge='#fcd34d',
        desc=(
            'Scans every possible parallel lane; selects the strip with the '
            'highest raw nodule-pixel count.  No avoidance of organisms or '
            'obstructions.'
        ),
    ),
    'terrain': dict(
        label='Terrain-Aware',
        subtitle='Operational Reality',
        face='#38bdf8',
        edge='#7dd3fc',
        desc=(
            'Routes through the corridor with the fewest Class-3/4 pixels.  '
            'Organisms and Obstructions are treated as hard barriers; nodule '
            'yield is a secondary concern.'
        ),
    ),
    'eco': dict(
        label='Eco-Optimised',
        subtitle='The Compromise',
        face='#34d399',
        edge='#6ee7b7',
        desc=(
            'Maximises nodule yield weighted by a heavy biological penalty '
            '(3 × hazard cells subtracted from nodule count).  Finds the '
            '"Green Zone" — high resource density, low ecological conflict.'
        ),
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _meters_to_rows(meters: float, mpp: float) -> int:
    """Convert a physical width in metres to an integer row count."""
    return max(1, int(round(meters / mpp)))


def _lane_stats(mask: np.ndarray,
                row_start: int,
                row_end: int) -> Dict[str, int]:
    """Count semantic-class pixels inside a horizontal lane."""
    region = mask[row_start:row_end, :]
    return {
        'nodules':       int(np.sum(region == _CLS_NODULE)),
        'organisms':     int(np.sum(region == _CLS_ORGANISM)),
        'obstructions':  int(np.sum(region == _CLS_OBSTRUCTION)),
        'hazards':       int(np.sum((region == _CLS_ORGANISM) |
                                    (region == _CLS_OBSTRUCTION))),
    }


def _compute_lanes(mask: np.ndarray,
                   vehicle_rows: int) -> Dict[str, Dict[str, int]]:
    """
    Return the four lane definitions as {scenario: {row_start, row_end}}.

    Each lane is a horizontal strip `vehicle_rows` tall that spans the
    full image width.
    """
    n_rows, n_cols = mask.shape
    margin = vehicle_rows  # keep the strip fully within the image

    # Pre-compute per-lane scores for all candidate positions
    best = {
        'profit':  {'score': -np.inf, 'row': n_rows // 2 - vehicle_rows // 2},
        'terrain': {'score':  np.inf,  'row': n_rows // 2 - vehicle_rows // 2},
        'eco':     {'score': -np.inf,  'row': n_rows // 2 - vehicle_rows // 2},
    }

    for r in range(margin, n_rows - vehicle_rows - margin):
        s = _lane_stats(mask, r, r + vehicle_rows)
        # Profit: maximise nodules, ignore hazards
        if s['nodules'] > best['profit']['score']:
            best['profit']['score'] = s['nodules']
            best['profit']['row']   = r
        # Terrain: minimise hazards (ties broken by row order → first found)
        if s['hazards'] < best['terrain']['score']:
            best['terrain']['score'] = s['hazards']
            best['terrain']['row']   = r
        # Eco: maximise (nodules - 3×hazards)
        eco_score = s['nodules'] - 3 * s['hazards']
        if eco_score > best['eco']['score']:
            best['eco']['score'] = eco_score
            best['eco']['row']   = r

    centre_row = n_rows // 2 - vehicle_rows // 2

    lanes = {
        'baseline': {'row_start': centre_row,
                     'row_end':   centre_row + vehicle_rows},
        'profit':   {'row_start': best['profit']['row'],
                     'row_end':   best['profit']['row'] + vehicle_rows},
        'terrain':  {'row_start': best['terrain']['row'],
                     'row_end':   best['terrain']['row'] + vehicle_rows},
        'eco':      {'row_start': best['eco']['row'],
                     'row_end':   best['eco']['row'] + vehicle_rows},
    }
    return lanes


def _calc_metrics(mask: np.ndarray,
                  lane: Dict[str, int],
                  buffer_rows: int) -> Dict[str, Any]:
    """
    Compute KPIs for one scenario lane.

    Returns
    -------
    dict with keys:
        nodules_captured, organisms_direct, organisms_buffer,
        obstructions_hit, bio_loss_total, nodule_pct,
        efficiency_pct, lane_area_cells
    """
    rs, re = lane['row_start'], lane['row_end']
    direct = _lane_stats(mask, rs, re)

    # Buffer zone: rows immediately above and below the lane
    buf_top    = _lane_stats(mask, max(0, rs - buffer_rows), rs)
    buf_bottom = _lane_stats(mask, re, min(mask.shape[0], re + buffer_rows))
    buffer_org = buf_top['organisms'] + buf_bottom['organisms']

    total_nodules = int(np.sum(mask == _CLS_NODULE))
    nodule_pct    = (direct['nodules'] / total_nodules * 100
                     if total_nodules > 0 else 0.0)

    lane_area = (re - rs) * mask.shape[1]
    efficiency = (direct['nodules'] / lane_area * 100
                  if lane_area > 0 else 0.0)

    return {
        'nodules_captured':   direct['nodules'],
        'organisms_direct':   direct['organisms'],
        'organisms_buffer':   int(buffer_org * 0.30),   # 30 % indirect mortality
        'obstructions_hit':   direct['obstructions'],
        'bio_loss_total':     direct['organisms'] + int(buffer_org * 0.30),
        'nodule_pct':         nodule_pct,
        'efficiency_pct':     efficiency,
        'lane_area_cells':    lane_area,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-scenario figure renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_scenario_figure(
    scenario_id:       str,
    mask:              np.ndarray,
    lane:              Dict[str, int],
    metrics:           Dict[str, Any],
    meters_per_pixel:  float,
    ortho:             Optional[np.ndarray],
    output_path:       Path,
    figure_dpi:        int = 150,
) -> None:
    """
    Render a single-scenario figure:

    Top row  ── [Clean map]   [Overlay map]
    Bottom   ── Metrics strip (4 KPI boxes + description)
    """
    st = _SCENARIO_STYLE[scenario_id]
    face_c = st['face']
    edge_c = st['edge']

    n_rows, n_cols = mask.shape
    max_x = n_cols * meters_per_pixel
    max_y = n_rows * meters_per_pixel

    rs, re = lane['row_start'], lane['row_end']
    lane_y0 = rs * meters_per_pixel
    lane_y1 = re * meters_per_pixel
    lane_h  = lane_y1 - lane_y0

    fig = plt.figure(figsize=(16, 10), dpi=figure_dpi)
    fig.patch.set_facecolor('#07101f')

    # ── Layout: 2 maps (top), metrics strip (bottom) ──────────────────────
    outer = gridspec.GridSpec(2, 1, figure=fig,
                              height_ratios=[4, 1],
                              hspace=0.06,
                              left=0.04, right=0.96,
                              top=0.92, bottom=0.04)
    map_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0], wspace=0.05)
    ax_clean   = fig.add_subplot(map_gs[0])
    ax_overlay = fig.add_subplot(map_gs[1])

    def _setup_map_ax(ax, title_suffix):
        ax.set_facecolor('#0a1628')
        ax.set_xlim(0, max_x)
        ax.set_ylim(max_y, 0)
        ax.set_aspect('equal')
        ax.tick_params(colors='#334155', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e293b')
        ax.set_xlabel('X (m)', fontsize=8, color='#475569')
        ax.set_ylabel('Y (m)', fontsize=8, color='#475569')
        ax.set_title(
            f'{st["label"]}  ·  {title_suffix}',
            fontsize=10, fontweight='bold',
            color=edge_c, pad=6,
            fontfamily='monospace',
        )

    _setup_map_ax(ax_clean,   'Clean')
    _setup_map_ax(ax_overlay, 'Overlay')

    # ── Semantic mask as RGBA raster (shared between both axes) ───────────
    #   Build a colour image from the mask classes
    rgba = np.zeros((n_rows, n_cols, 4), dtype=np.float32)
    rgba[mask == _CLS_SUBSTRATE]  = [0.08, 0.12, 0.10, 1.0]
    rgba[mask == _CLS_NODULE]     = [0.45, 0.40, 0.22, 1.0]
    rgba[mask == _CLS_ORGANISM]   = [0.12, 0.38, 0.18, 1.0]
    rgba[mask == _CLS_OBSTRUCTION]= [0.38, 0.22, 0.14, 1.0]
    rgba[mask == _CLS_BG]         = [0.04, 0.06, 0.08, 1.0]

    extent = [0, max_x, max_y, 0]   # origin='upper' → [left, right, bottom, top]

    ax_clean.imshow(rgba, origin='upper', extent=extent,
                    interpolation='nearest', aspect='auto', zorder=1)

    if ortho is not None:
        # Downsample to 512 px max edge to avoid memory issues
        h, w = ortho.shape[:2]
        stride = max(1, max(h, w) // 512)
        img_ds = ortho[::stride, ::stride]
        ax_overlay.imshow(img_ds, origin='upper', extent=extent,
                          interpolation='bilinear', aspect='auto', zorder=1)
    else:
        ax_overlay.imshow(rgba, origin='upper', extent=extent,
                          interpolation='nearest', aspect='auto', zorder=1)

    # ── Mining lane rectangle ─────────────────────────────────────────────
    def _draw_lane(ax, alpha_face=0.28):
        rect = mpatches.FancyBboxPatch(
            (0, lane_y0), max_x, lane_h,
            boxstyle='square,pad=0',
            linewidth=1.4,
            edgecolor=edge_c,
            facecolor=face_c,
            alpha=alpha_face,
            zorder=3,
        )
        ax.add_patch(rect)
        # Dashed border lines (top + bottom edge)
        for y_edge in (lane_y0, lane_y1):
            ax.axhline(y_edge, color=edge_c, linewidth=1.2,
                       linestyle='--', alpha=0.7, zorder=4)
        # Direction arrow
        mid_y = (lane_y0 + lane_y1) / 2
        ax.annotate(
            '', xy=(max_x * 0.92, mid_y),
            xytext=(max_x * 0.08, mid_y),
            arrowprops=dict(
                arrowstyle='->', color=edge_c,
                lw=1.6, mutation_scale=14),
            zorder=5,
        )

    _draw_lane(ax_clean,   alpha_face=0.22)
    _draw_lane(ax_overlay, alpha_face=0.35)

    # ── Highlight casualties inside lane (overlay axis only) ─────────────
    lane_mask_region = mask[rs:re, :]
    # Build per-pixel highlight overlay
    hl = np.zeros((re - rs, n_cols, 4), dtype=np.float32)
    hl[lane_mask_region == _CLS_NODULE]      = [1.00, 0.90, 0.20, 0.65]
    hl[lane_mask_region == _CLS_ORGANISM]    = [1.00, 0.18, 0.18, 0.75]
    hl[lane_mask_region == _CLS_OBSTRUCTION] = [1.00, 0.45, 0.10, 0.70]

    hl_extent = [0, max_x, lane_y1, lane_y0]
    ax_overlay.imshow(hl, origin='upper', extent=hl_extent,
                      interpolation='nearest', aspect='auto', zorder=5)

    # ── Legend patches (clean axis) ───────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor='#73664e', edgecolor='#a09060', label='Nodule (Cl.2)'),
        mpatches.Patch(facecolor='#1f603b', edgecolor='#30a060', label='Organism (Cl.3)'),
        mpatches.Patch(facecolor='#5e3820', edgecolor='#a05030', label='Obstruction (Cl.4)'),
        mpatches.Patch(facecolor=face_c,    edgecolor=edge_c,    label='Mining lane', alpha=0.55),
    ]
    ax_clean.legend(
        handles=legend_items,
        loc='lower right',
        fontsize=7,
        framealpha=0.7,
        facecolor='#0a1628',
        edgecolor='#1e293b',
        labelcolor='#94a3b8',
    )

    # ── Casualty highlight legend (overlay axis) ──────────────────────────
    ov_legend = [
        mpatches.Patch(facecolor='#ffdd00', edgecolor='none', alpha=0.75, label='Nodule captured'),
        mpatches.Patch(facecolor='#ff2222', edgecolor='none', alpha=0.75, label='Organism killed'),
        mpatches.Patch(facecolor='#ff7020', edgecolor='none', alpha=0.75, label='Obstruction struck'),
    ]
    ax_overlay.legend(
        handles=ov_legend,
        loc='lower right',
        fontsize=7,
        framealpha=0.7,
        facecolor='#0a1628',
        edgecolor='#1e293b',
        labelcolor='#94a3b8',
    )

    # ── Metrics strip ─────────────────────────────────────────────────────
    ax_metrics = fig.add_subplot(outer[1])
    ax_metrics.set_facecolor('#07101f')
    ax_metrics.axis('off')

    # 4 KPI boxes
    kpis = [
        ('Nodules captured',   metrics['nodules_captured'],
         f"{metrics['nodule_pct']:.1f}% of field", face_c),
        ('Bio casualties',     metrics['bio_loss_total'],
         f"direct {metrics['organisms_direct']}  +  buffer {metrics['organisms_buffer']}", '#f87171'),
        ('Obstructions struck', metrics['obstructions_hit'],
         'Hard obstacles in path', '#fb923c'),
        ('Lane efficiency',    f"{metrics['efficiency_pct']:.1f}%",
         'Nodule cells / total lane cells', '#818cf8'),
    ]

    box_w = 0.20
    x_positions = [0.02, 0.26, 0.50, 0.74]
    for (label, value, sub, color), xp in zip(kpis, x_positions):
        # Box background
        ax_metrics.add_patch(mpatches.FancyBboxPatch(
            (xp, 0.08), box_w, 0.82,
            boxstyle='round,pad=0.01',
            transform=ax_metrics.transAxes,
            facecolor='#0d1b2e',
            edgecolor=color + '55',
            linewidth=1,
            clip_on=False,
        ))
        # Value
        ax_metrics.text(
            xp + box_w / 2, 0.62, str(value),
            transform=ax_metrics.transAxes,
            ha='center', va='center',
            fontsize=20, fontweight='bold',
            color=color, fontfamily='monospace',
        )
        # Label
        ax_metrics.text(
            xp + box_w / 2, 0.35, label,
            transform=ax_metrics.transAxes,
            ha='center', va='center',
            fontsize=8, color='#94a3b8',
        )
        # Sub-label
        ax_metrics.text(
            xp + box_w / 2, 0.16, sub,
            transform=ax_metrics.transAxes,
            ha='center', va='center',
            fontsize=7, color='#475569',
        )

    # Description text (right of KPI boxes)
    ax_metrics.text(
        0.98, 0.50, st['desc'],
        transform=ax_metrics.transAxes,
        ha='right', va='center',
        fontsize=8.5, color='#64748b',
        style='italic',
        wrap=True,
        multialignment='right',
    )

    # ── Title bar ─────────────────────────────────────────────────────────
    fig.text(
        0.50, 0.955,
        f'{st["label"].upper()}  ─  {st["subtitle"]}',
        ha='center', va='top',
        fontsize=14, fontweight='bold',
        color=edge_c, fontfamily='monospace',
    )

    plt.savefig(output_path, dpi=figure_dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: {output_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary figure
# ─────────────────────────────────────────────────────────────────────────────

def _render_summary_figure(
    scenarios:         list,          # ordered list of scenario IDs
    all_lanes:         Dict,
    all_metrics:       Dict,
    mask:              np.ndarray,
    meters_per_pixel:  float,
    ortho:             Optional[np.ndarray],
    output_path:       Path,
    figure_dpi:        int = 150,
) -> None:
    """
    2×2 grid of mini-maps + grouped bar chart comparing all four scenarios.
    """
    n_rows, n_cols = mask.shape
    max_x = n_cols * meters_per_pixel
    max_y = n_rows * meters_per_pixel

    # Downsampled semantic RGBA (shared background)
    rgba = np.zeros((n_rows, n_cols, 4), dtype=np.float32)
    rgba[mask == _CLS_SUBSTRATE]   = [0.08, 0.12, 0.10, 1.0]
    rgba[mask == _CLS_NODULE]      = [0.45, 0.40, 0.22, 1.0]
    rgba[mask == _CLS_ORGANISM]    = [0.12, 0.38, 0.18, 1.0]
    rgba[mask == _CLS_OBSTRUCTION] = [0.38, 0.22, 0.14, 1.0]
    rgba[mask == _CLS_BG]          = [0.04, 0.06, 0.08, 1.0]
    extent = [0, max_x, max_y, 0]

    fig = plt.figure(figsize=(20, 14), dpi=figure_dpi)
    fig.patch.set_facecolor('#07101f')

    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[3, 1.6],
        hspace=0.10,
        left=0.04, right=0.96,
        top=0.93, bottom=0.04,
    )

    # ── 2×2 map grid ──────────────────────────────────────────────────────
    map_gs = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer[0],
        wspace=0.05, hspace=0.12,
    )

    for idx, sid in enumerate(scenarios):
        row_i, col_i = divmod(idx, 2)
        ax = fig.add_subplot(map_gs[row_i, col_i])
        st = _SCENARIO_STYLE[sid]
        lane = all_lanes[sid]
        rs, re = lane['row_start'], lane['row_end']
        lane_y0 = rs * meters_per_pixel
        lane_y1 = re * meters_per_pixel
        lane_h  = lane_y1 - lane_y0
        m = all_metrics[sid]

        ax.set_facecolor('#0a1628')
        ax.set_xlim(0, max_x);  ax.set_ylim(max_y, 0)
        ax.set_aspect('equal')
        ax.tick_params(colors='#334155', labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e293b')
        ax.set_xlabel('X (m)', fontsize=7, color='#475569')
        ax.set_ylabel('Y (m)', fontsize=7, color='#475569')

        # Background
        if ortho is not None:
            h, w = ortho.shape[:2]
            stride = max(1, max(h, w) // 512)
            img_ds = ortho[::stride, ::stride]
            ax.imshow(img_ds, origin='upper', extent=extent,
                      interpolation='bilinear', aspect='auto',
                      alpha=0.6, zorder=1)
        ax.imshow(rgba, origin='upper', extent=extent,
                  interpolation='nearest', aspect='auto',
                  alpha=0.5 if ortho is not None else 1.0, zorder=2)

        # Lane rectangle
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, lane_y0), max_x, lane_h,
            boxstyle='square,pad=0',
            linewidth=1.5, edgecolor=st['edge'],
            facecolor=st['face'], alpha=0.30, zorder=3,
        ))
        for ye in (lane_y0, lane_y1):
            ax.axhline(ye, color=st['edge'], linewidth=1.0,
                       linestyle='--', alpha=0.65, zorder=4)

        # Casualties highlight
        hl = np.zeros((re - rs, n_cols, 4), dtype=np.float32)
        lmr = mask[rs:re, :]
        hl[lmr == _CLS_NODULE]      = [1.00, 0.90, 0.20, 0.55]
        hl[lmr == _CLS_ORGANISM]    = [1.00, 0.18, 0.18, 0.65]
        hl[lmr == _CLS_OBSTRUCTION] = [1.00, 0.45, 0.10, 0.60]
        ax.imshow(hl, origin='upper',
                  extent=[0, max_x, lane_y1, lane_y0],
                  interpolation='nearest', aspect='auto', zorder=5)

        # Mini title + KPI badge
        ax.set_title(
            f'{st["label"]}  ·  {st["subtitle"]}',
            fontsize=9, fontweight='bold',
            color=st['edge'], pad=4, fontfamily='monospace',
        )
        ax.text(
            0.99, 0.02,
            f"Yield {m['nodule_pct']:.1f}%  |  Bio loss {m['bio_loss_total']}  |  Obstr {m['obstructions_hit']}",
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=7, color='#64748b',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', fc='#07101f', ec='#1e293b', alpha=0.85),
        )

    # ── Grouped bar chart ─────────────────────────────────────────────────
    chart_gs = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[1],
        wspace=0.12, hspace=0,
    )

    kpi_defs = [
        {
            'ax_idx': 0,
            'title': 'Nodule Yield (cells captured)',
            'key': 'nodules_captured',
            'color_fn': lambda i, sid: _SCENARIO_STYLE[sid]['face'],
            'higher_is_better': True,
        },
        {
            'ax_idx': 1,
            'title': 'Biological Casualties (direct + buffer)',
            'key': 'bio_loss_total',
            'color_fn': lambda i, sid: '#f87171',
            'higher_is_better': False,
        },
        {
            'ax_idx': 2,
            'title': 'Obstructions Struck',
            'key': 'obstructions_hit',
            'color_fn': lambda i, sid: '#fb923c',
            'higher_is_better': False,
        },
    ]

    x = np.arange(len(scenarios))
    bar_w = 0.55

    for kd in kpi_defs:
        ax = fig.add_subplot(chart_gs[kd['ax_idx']])
        ax.set_facecolor('#0a1628')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e293b')
        ax.tick_params(colors='#475569', labelsize=7)
        ax.yaxis.label.set_color('#475569')

        values = [all_metrics[sid][kd['key']] for sid in scenarios]
        best_val = max(values) if kd['higher_is_better'] else min(values)
        worst_val = min(values) if kd['higher_is_better'] else max(values)

        bar_colors = []
        for i, (val, sid) in enumerate(zip(values, scenarios)):
            if val == best_val:
                bar_colors.append('#34d399')
            elif val == worst_val:
                bar_colors.append('#f87171')
            else:
                bar_colors.append(kd['color_fn'](i, sid) + 'bb')

        bars = ax.bar(x, values, bar_w,
                      color=bar_colors,
                      edgecolor='#1e293b',
                      linewidth=0.5,
                      zorder=3)

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                str(val) if isinstance(val, int) else f'{val:.1f}',
                ha='center', va='bottom',
                fontsize=8, color='#94a3b8',
                fontfamily='monospace',
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [_SCENARIO_STYLE[sid]['label'] for sid in scenarios],
            rotation=12, ha='right', fontsize=7.5,
        )
        ax.set_title(kd['title'], fontsize=9, color='#64748b', pad=6)
        ax.set_facecolor('#0a1628')
        ax.grid(axis='y', color='#1e293b', linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)

        # "best / worst" annotation
        note = '↑ higher = better' if kd['higher_is_better'] else '↓ lower = better'
        ax.text(
            0.99, 0.97, note,
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=7, color='#334155', style='italic',
        )

    # ── Main title ────────────────────────────────────────────────────────
    fig.text(
        0.50, 0.965,
        'EXTRACTION PATH ANALYSIS  ─  MISSION COMPARISON',
        ha='center', va='top',
        fontsize=15, fontweight='bold',
        color='#e2e8f0', fontfamily='monospace',
    )
    fig.text(
        0.50, 0.944,
        'All four scenarios generated automatically from semantic mask · '
        'GREEN = best outcome · RED = worst outcome',
        ha='center', va='top',
        fontsize=8, color='#475569',
    )

    plt.savefig(output_path, dpi=figure_dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: {output_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Public method (attach to SpatialMetricsVisualizer)
# ─────────────────────────────────────────────────────────────────────────────

def visualize_mining_scenarios(
    self,
    vehicle_width_m: float = 5.0,
    buffer_m: float = 2.0,
    output_name: str = 'mining_scenarios',
) -> Dict[str, str]:
    """
    Auto-generate all four mining-path scenarios and save five figures.

    No ``mining_polygon`` input is required — all paths are derived
    algorithmically from the semantic segmentation mask.

    Parameters
    ----------
    vehicle_width_m : float
        Collector-head width in metres (default 5 m).
    buffer_m : float
        Buffer zone around the lane used for indirect biological loss
        (default 2 m).
    output_name : str
        Base filename prefix for output PNGs (default 'mining_scenarios').

    Returns
    -------
    dict
        Mapping scenario_id → file path (plus 'summary' key).

    Saved files
    -----------
    ``{output_name}_baseline.png``  – Dumb Centre-Cut
    ``{output_name}_profit.png``    – Profit Maximizer
    ``{output_name}_terrain.png``   – Terrain-Aware
    ``{output_name}_eco.png``       – Eco-Optimised
    ``{output_name}_summary.png``   – 2×2 comparison + bar chart
    """
    print("\n" + "=" * 60)
    print("VISUALIZING MINING SCENARIOS (4-PATH ANALYSIS)")
    print("=" * 60)

    # ── Load mask ─────────────────────────────────────────────────────────
    mask = self.analyzer._load_mask()
    mpp  = self.analyzer.meters_per_pixel

    vehicle_rows = _meters_to_rows(vehicle_width_m, mpp)
    buffer_rows  = _meters_to_rows(buffer_m, mpp)

    print(f"  Vehicle width : {vehicle_width_m} m  →  {vehicle_rows} rows")
    print(f"  Buffer zone   : {buffer_m} m  →  {buffer_rows} rows")
    print(f"  Mask shape    : {mask.shape}")

    # ── Compute lanes ─────────────────────────────────────────────────────
    lanes   = _compute_lanes(mask, vehicle_rows)
    metrics = {sid: _calc_metrics(mask, lanes[sid], buffer_rows)
               for sid in lanes}

    print("\n  Lane positions (row_start → row_end):")
    for sid, lane in lanes.items():
        m = metrics[sid]
        print(f"    {sid:12s}  rows {lane['row_start']:4d}→{lane['row_end']:4d}"
              f"  nodules={m['nodules_captured']:5d}"
              f"  bio_loss={m['bio_loss_total']:4d}"
              f"  obstr={m['obstructions_hit']:3d}")

    # ── Orthomosaic (optional) ────────────────────────────────────────────
    ortho = self.orthomosaic_array  # may be None

    # ── Render individual figures ─────────────────────────────────────────
    output_paths: Dict[str, str] = {}
    ordered = ['baseline', 'profit', 'terrain', 'eco']

    print()
    for sid in ordered:
        out_path = self.output_dir / f"{output_name}_{sid}.png"
        _render_scenario_figure(
            scenario_id=sid,
            mask=mask,
            lane=lanes[sid],
            metrics=metrics[sid],
            meters_per_pixel=mpp,
            ortho=ortho,
            output_path=out_path,
            figure_dpi=self.figure_dpi,
        )
        output_paths[sid] = str(out_path)

    # ── Render summary figure ─────────────────────────────────────────────
    summary_path = self.output_dir / f"{output_name}_summary.png"
    _render_summary_figure(
        scenarios=ordered,
        all_lanes=lanes,
        all_metrics=metrics,
        mask=mask,
        meters_per_pixel=mpp,
        ortho=ortho,
        output_path=summary_path,
        figure_dpi=self.figure_dpi,
    )
    output_paths['summary'] = str(summary_path)

    print("\n" + "=" * 60)
    print("MINING SCENARIO VISUALIZATION COMPLETE")
    print("=" * 60)

    return output_paths


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test with a synthetic mask
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import tempfile, os

    print("Running standalone test with synthetic mask...")

    # ── Build a fake mask (500×700) ───────────────────────────────────────
    rng  = np.random.default_rng(42)
    rows, cols = 500, 700
    mask = np.ones((rows, cols), dtype=np.uint8)  # all substrate

    # Nodule patches: random Gaussian blobs
    for _ in range(18):
        cy, cx = rng.integers(40, rows-40), rng.integers(40, cols-40)
        sy, sx = rng.integers(15, 45), rng.integers(15, 45)
        for r in range(max(0, cy-sy), min(rows, cy+sy)):
            for c in range(max(0, cx-sx), min(cols, cx+sx)):
                d = ((r-cy)/sy)**2 + ((c-cx)/sx)**2
                if d < 1.0 and rng.random() < 0.75:
                    mask[r, c] = _CLS_NODULE

    # Organism clusters
    for _ in range(8):
        cy, cx = rng.integers(30, rows-30), rng.integers(30, cols-30)
        r_range = range(max(0, cy-12), min(rows, cy+12))
        c_range = range(max(0, cx-12), min(cols, cx+12))
        for r in r_range:
            for c in c_range:
                if ((r-cy)**2 + (c-cx)**2) < 100 and rng.random() < 0.55:
                    mask[r, c] = _CLS_ORGANISM

    # Obstruction blobs
    for _ in range(5):
        cy, cx = rng.integers(30, rows-30), rng.integers(30, cols-30)
        for r in range(max(0, cy-8), min(rows, cy+8)):
            for c in range(max(0, cx-8), min(cols, cx+8)):
                if ((r-cy)**2 + (c-cx)**2) < 40 and rng.random() < 0.8:
                    mask[r, c] = _CLS_OBSTRUCTION

    print(f"  Mask classes: "
          f"substrate={np.sum(mask==1)}, nodule={np.sum(mask==2)}, "
          f"organism={np.sum(mask==3)}, obstruction={np.sum(mask==4)}")

    # ── Minimal stub of analyser/visualiser ───────────────────────────────
    class _FakeAnalyzer:
        meters_per_pixel = 0.01
        def _load_mask(self): return mask

    class _FakeVisualizer:
        analyzer         = _FakeAnalyzer()
        orthomosaic_array = None
        output_dir       = Path('/home/claude/test_output')
        figure_dpi       = 120

        # bind the module-level function as a method
        def visualize_mining_scenarios(self, **kw):
            return visualize_mining_scenarios(self, **kw)

    Path('/home/claude/test_output').mkdir(exist_ok=True)
    viz = _FakeVisualizer()
    paths = viz.visualize_mining_scenarios(
        vehicle_width_m=1.5,
        buffer_m=0.5,
        output_name='test_mining',
    )

    print("\nOutput files:")
    for k, v in paths.items():
        size = os.path.getsize(v) // 1024
        print(f"  {k:12s} → {v}  ({size} KB)")