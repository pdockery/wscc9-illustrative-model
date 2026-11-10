# The combined figure: defaults and approved variants

The 9-bus combined figure carries the series' pedagogy. It has an exemplar cell, a default presentation and a short list of approved variants, each named by the notebook and section that draws it (2026-11-07). A new figure starts as the exemplar's default and is changed from there, additively (a highlighted path, an inset, an annotation, a purpose-built configuration of the same panels), with each change stated in the cell that draws it. Every clearing that carries a finding is shown as the standard letter first, and the chart of the finding follows it. The defaults were set on 2026-09-12 and live as module constants; a stored figure from before that date carries the earlier presentation until its notebook is re-run.

## The 9-bus letter

**Exemplar.** `workshop3_transactions_across_boundaries.ipynb`, Example 3, the cell that draws Market A's award of slide 21: `seam_views.own_view` → `wscc9_figures.footprint_figure` → `nodal_plot.plot_combined_letter`, with the bid drawn through `wscc9_figures.bid_bars`, and without the titles and the legend a deck notebook drops. The unified composite of Example 1 of `101_sandbox.ipynb` is the same letter with nothing greyed.

**Left panel, the network (`plot_network_topology`).**
- Buses in the canonical palette (`BUS_COLORS`), numbered in chips beside the bubble; a bus outside the focus footprint is greyed.
- Every node labelled with its signed net injection (`+59 MW`, `−90 MW`) from `res.injection` (`node_net_mw=True`); a transit bus carries no label. The LMP is printed on the network only when the ring does not carry it.
- Line width proportional to susceptance $b=1/x$ (`susceptance_widths`); line colour the managing market or balancing authority (`fp.line_colors`), grey where no single operator manages it; each line labelled `flow/rating` with a flow arrow.
- A line drawn red only when its dual is non-zero (`constrained_lines` from `res.line_dual`), never because flow reaches the rating.

**Right panel, the nodal dispatch chord (`plot_nodal_circlize`).**
- One sector per bus in the ring order and rotation of `wscc9_model`; an outer band per market or balancing authority (`bus_groups`, `group_colors`), labelled.
- Each unit a bar at its offer price, solid where dispatched and faded where idle, labelled with its MW inside and its `$offer × capacity` outside; a rising curve (`curves=`) as a sloped wedge.
- A dashed line at the bus LMP across the dispatched stack, so the gap above a bar is the inframarginal rent; the sector labelled with the bus and its LMP.
- Demand at LMP height, labelled in MW; shed as a faint tail (`shed_segments`).
- PTDF-traced generator-to-load chords on the served load (`compute_ptdf_flows`, `served_demand`).
- A self-schedule, an import or export from outside the market area, or a price-sensitive bid drawn in gray (`self_schedule_bars`, `bid_bars`, `exo_sched=`): a price-taking block carries no price label and takes the node's price for its height; a bid keeps its price label, its envelope is the bid quantity and its solid fill the award; a balanced schedule gets its own gray chord lane; the legend note names what the gray is.
- One legend along the bottom for both panels.

**Approved variants.**
1. The unified composite, nothing greyed (`composite_figure`): `101_sandbox.ipynb`, Example 1.
2. The footprint view, the other area greyed (`footprint_figure` with `highlight=` or `dim_buses=`): `101_sandbox.ipynb`, Example 2, Market B's clearing with the awards as a fixed injection; each market's own clearing in the workshop notebooks reaches it through `own_view` (variant 4).
3. The transfer view, the Transfers inset in the lower left of the network panel (`transfer_figure`, `transfer_inset`): each area a bubble labelled with its net surplus, the schedule as `flow/limit` with width the released capacity, red with $\mu_T$ when it binds. Available; `101_sandbox.ipynb` draws it after Example 1's composite when an interface binds, and no stored run binds one.
4. The seam frames (`seam_views`): `input_view`, `bid_view`, `own_view` (with `proxy=` the neighbour greyed as that market's forecast), `forecast_view`, `combined_views` for the superposition. `own_view`: `workshop2_congestion_management.ipynb` slides 6, 9, 14 and 15 (with `proxy=` on slides 14 and 15), `workshop3_transactions_across_boundaries.ipynb` Examples 1 to 4, `workshop4_loading_relief.ipynb` slides 57, 60, 82 and 83. `forecast_view`: `workshop2_congestion_management.ipynb` slides 13 and 15. `input_view`: `workshop3_transactions_across_boundaries.ipynb` Example 2. `bid_view`: `workshop3_transactions_across_boundaries.ipynb` Examples 3 and 4. `combined_views`: `workshop2_congestion_management.ipynb` slides 10, 11 and 16, `workshop4_loading_relief.ipynb` slides 61 to 63, 71, 77 and 84, and `101_sandbox.ipynb` Example 2.
5. The rights views (`rights_figure`, `rights_payoff_figure`): POR→POD arcs over the network with the ATC table, then each right's payoff. Available; no notebook here draws it.
6. The configuration view, no dispatch in it: every unit at nameplate drawn solid, demand at the dearest offer's height, no chords, line ratings labelled. It opens or closes a notebook and is never a clearing. `workshop2_congestion_management.ipynb` slide 4, `workshop3_transactions_across_boundaries.ipynb` "The case", `workshop4_loading_relief.ipynb` slide 55 and `101_sandbox.ipynb` "The case" draw it.
7. The intertie 3-set (`intertie.tie_figure`, `neigh_figure`, `combined_figure`): the transacting market's model, the other footprint's, and the combined actual. Available; no notebook here draws it.

## In a deck notebook

A figure in a deck notebook carries no suptitle, no panel title and no legend, because the slide it reproduces carries them (user decision 2026-11-07). It is drawn inline at the default inline dpi. The sandbox follows the same rule, so that a figure drawn there can be set beside a slide.

## Where the defaults live

`nodal_plot.TRANSIT_COLOR` and `wscc9_figures._SELFSCHED`, with the palette in `wscc9_model`. A change to a default is a change to the library, so every notebook draws it. It is proposed in `dev_notes.md`'s open decisions and arrives as a fresh copy of `lib/` with a regenerated `lib/MANIFEST.md`.
