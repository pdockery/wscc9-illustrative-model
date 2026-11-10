# Exemplars: the notebook voice

The passages below are verbatim from the nodal market fundamentals notebook, published in an earlier version of this repository (commit ce66306). The working tree no longer carries it.

The teaching notebooks' markdown, as written in that notebook, the tested and edited standard for the series (user decision 2026-10-12). The voice is the technical description's (strictly definitional declaratives, full names, numbered classifications, `Note that` for a mechanical consequence) applied to the parts a notebook has. Every passage below is verbatim from its markdown cells, except that a sibling notebook's number is replaced by [a later notebook]; inline math is kept as written. Pick three to five by the part being drafted, never by topic.

| id | document part | cell |
|---|---|---|
| NB-01 | title cell: what the notebook is, where it sits, what it hands to a sibling | 0 |
| NB-02 | formulation opener: the engine named, its objects introduced, the nomenclature convention | 11 |
| NB-03 | the nomenclature table's shape (header and first rows) | 11 |
| NB-04 | the program stated whole, with its `*In words.*` blurb | 11 |
| NB-05 | a constraint restated, with the `Note that` consequence | 11 |
| NB-06 | a derived result read in prose (stationarity) | 11 |
| NB-07 | a defined quantity with its equation and blurb (congestion rent) | 12 |
| NB-08 | the `# -- EDIT --` lead-in | 13 |
| NB-09 | the prose under an output: the number read, then stop | 20 |
| NB-10 | a section opener that tightens a construct already stated | 21 |
| NB-11 | a mechanism explained in plain prose (the self-schedule) | 23 |
| NB-12 | the sandbox introduction | 25 |
| NB-13 | the configuration view described, a figure that is not a clearing | 27 |

Its cells 8, 16 and 18 are not exemplars: they carry an agency verb on the network ("where power prefers to flow"), em-dashes, and mid-sentence bold, which the register profile cuts on sight.

<examples>

<example id="NB-01" part="title cell">
# Nodal Markets on the WSCC 9-Bus Network — Fundamentals

> **Work in progress** — an illustrative teaching notebook under active development, not an authoritative reference.

This is the first notebook in the illustrative series. It builds the WSCC / IEEE
9-bus test case and clears a single nodal energy market on it. The settlement ledgers, the
congestion revenue by region, and the autarky Pareto check are covered in [a later notebook].

Everything runs on the shared teaching libraries (`wscc9_model`,
`footprints`, `revenue_allocation`, `wscc9_figures`) built on the `seams_engine`
PTDF DC-OPF and the `nodal_plot` figures.
</example>

<example id="NB-02" part="formulation opener">
The engine solvers (`solve_engine_dispatch` / `solve_engine_transfers`) clear DC-OPF on the shared network. The clearing is a bid-based
security-constrained economic dispatch (SCED) reduced to a single period with
energy-cost bids only. Unit commitment is not included in the formulation, for
simplicity. An engine $M$ brings its resources $\mathcal R^M$ (cost $c_i$, cap
$\bar g_i$ at bus $\mathrm{bus}(i)$), its served load $d^M_n$, an activated line
set $\mathcal M^M_{\mathrm{act}}$, and any exogenous cross-footprint schedule
$\mathrm{exo}^M_n$.

Western markets schedule transfers between balancing-authority areas, $\mathcal N_a$. Each area carries
its own power balance and energy is scheduled between areas as a matched export/import pair on a declared interface.
Transfers are held within a transport capacity constraint.

**Nomenclature.** Sets are script capitals; an element is the matching
lower-case letter (upper-case for markets).
</example>

<example id="NB-03" part="nomenclature table">
| Symbol | Code | Definition |
|---|---|---|
| $\mathcal N$ | `pt.buses` | the set of buses in the network; $n\in\mathcal N$ |
| $\mathcal A$ | `REGIONS` keys (Section 3) | the set of balancing-authority areas **in the market**; $a\in\mathcal A$, and the $\mathcal N_a$ partition $\mathcal N$ |
| $\mathcal M^M_{\mathrm{act}}\subseteq\mathcal M$ | `eng.activated_lines` | the lines whose limits market $M$ activates (enforces) in its clearing |
| $g_i$ | `res.dispatch` | dispatch of resource $i$ — the engine's decision variable, $0\le g_i\le\bar g_i$ |
| $\hat\lambda^M_a,\ \mu^M_{T,ab}$ | `res.area_prices` / `res.transfer_duals` | area $a$'s power-balance dual — its energy price — and the $a\to b$ interface's shadow price, $\hat\lambda^M_b-\hat\lambda^M_a$ when its released capacity binds |

Note: lower-case $m$ indexes a **line**, upper-case $M$
names a **market engine** drawn from $\mathcal E$, and script $\mathcal M$ is the network line set. The balancing-authority areas
$a\in\mathcal A$ enter an engine only as the bus sets $\mathcal N_a$. The
formulation is written for a general $M\in\mathcal E$. The unified clearing of
this section is the special case where $\mathcal E$ has a single member spanning
every bus.
</example>

<example id="NB-04" part="the program with its In words blurb">
**The program minimizes the engine's own dispatch cost, subject to its constraints.**

$$\begin{aligned}
\min_{0\le g_i\le \bar g_i,\ t^M\ge 0}\ \ C^M \;&=\; \sum_{i\in\mathcal R^M} c_i\,g_i \\[2pt]
\text{s.t.}\quad \sum_{i\in\mathcal R^M:\,\mathrm{bus}(i)\in\mathcal N_a} g_i \;&=\; \sum_{n\in\mathcal N_a} \big(d^M_n-\mathrm{exo}^M_n\big)+\sum_{b}\big(t^M_{ab}-t^M_{ba}\big),\quad a\in\mathcal A && [\hat\lambda^M_a] \\
-\bar F_m \;\le\; F^M_m \;&\le\; \bar F_m,\quad m\in\mathcal M^M_{\mathrm{act}} && [\mu^M_m] \\
t^M_{ab} \;&\le\; \bar T_{ab},\quad (a,b)\in\mathcal I^M && [\mu^M_{T,ab}]
\end{aligned}\qquad\qquad(1)$$

*In words.* Engine $M$ minimizes its own dispatch cost
$C^M=\sum_i c_i\,g_i$ over its resources $\mathcal R^M$, choosing each resource's
dispatch $g_i$ within its capacity, $0\le g_i\le\bar g_i$, and non-negative
scheduled transfers $t^M$, subject to three constraint families: one power
balance per balancing-authority area $a\in\mathcal A$, whose generation inside
the area equals the load it serves there net of exogenous injection plus its net
scheduled transfer $\sum_b\big(t^M_{ab}-t^M_{ba}\big)$ (dual $\hat\lambda^M_a$,
area $a$'s energy price); its own flow $F^M_m$ held within the rating $\bar F_m$
on each activated line $m\in\mathcal M^M_{\mathrm{act}}$ (dual $\mu^M_m$, that
line's shadow price); and each scheduled transfer $t^M_{ab}$ over the declared
interface set $\mathcal I^M$ held within its released capacity $\bar T_{ab}$
(dual $\mu^M_{T,ab}$, that interface's shadow price).
</example>

<example id="NB-05" part="a constraint restated, with Note that">
**Flow and activated limits, $\mu^M_m$.**

$$F^M_m=\sum_{n\in\mathcal N} SF_{n,m}\,p^{\mathrm{inj},M}_n\ \ (m\in\mathcal M),\qquad
  -\bar F_m\le F^M_m\le \bar F_m\ \ [\mu^M_m]\ \ (m\in\mathcal M^M_{\mathrm{act}}),\qquad
  p^{\mathrm{inj},M}_n=\!\!\!\sum_{i\in\mathcal R^M:\,\mathrm{bus}(i)=n}\!\!\!g_i-d^M_n+\mathrm{exo}^M_n\ \ (n\in\mathcal N). \qquad(3)$$

*In words.* Engine $M$'s flow on line $m$, $F^M_m$, is each bus's shift factor
$SF_{n,m}$ on that line times the engine's net injection there,
$p^{\mathrm{inj},M}_n$, summed over every bus $n\in\mathcal N$. That flow is held
within the rating $\bar F_m$ only on the lines the engine activates,
$m\in\mathcal M^M_{\mathrm{act}}$ (dual $\mu^M_m$, that line's signed shadow
price). The net injection $p^{\mathrm{inj},M}_n$ is the engine's generation at
bus $n$, minus the load it serves there, plus its exogenous injection
$\mathrm{exo}^M_n$. Note that a line outside $\mathcal M^M_{\mathrm{act}}$ has
its flow computed and its limit relaxed, so that line carries no shadow price
(`activated_lines`).

The transfers are **flow-inert**. Physical flows come from the actual nodal
injections. The schedule is the accounting quantity the interfaces cap and the
settlement rides on.
</example>

<example id="NB-06" part="a derived result read in prose">
For a **marginal** unit the two bound multipliers vanish by complementary slackness, so the
$\overline\mu^M_i-\underline\mu^M_i$ term drops, leaving
$c_i=\hat\lambda^M_a+\sum_{m}SF_{n,m}\,\mu^M_m$. Stationarity in the **transfer variable**
$t^M_{ab}$ ties the areas' balance duals together:

$$\frac{\partial \mathcal L^M}{\partial t^M_{ab}}
  = \hat\lambda^M_a-\hat\lambda^M_b+\mu^M_{T,ab}-\underline\mu^M_{T,ab}\;=\;0
  \quad\Longrightarrow\quad
  \hat\lambda^M_b-\hat\lambda^M_a=\mu^M_{T,ab}\ \text{(binding)},\qquad \hat\lambda^M_a=\hat\lambda^M_b\ \text{(interior)}.$$

An interface's shadow price is exactly the separation between the two areas' energy prices. Areas joined by slack interfaces share one price.
</example>

<example id="NB-07" part="a defined quantity with its equation and blurb">
### Congestion rent

The unified clearing is the single engine over all buses, with $\mathrm{exo}=0$,
every line activated, and every interface slack, so all $\hat\lambda_a$ collapse
to one $\lambda$. The engine superscript is therefore dropped:
$\lambda_n\equiv\lambda^M_n$, $\mu_m\equiv\mu^M_m$, $d_n\equiv d^M_n$, and
$\mathcal M_{\mathrm{act}}$ is the activated set. The dispatch at bus $n$ is
$g_n=\sum_{i:\,\mathrm{bus}(i)=n} g_i$. The congestion rent values each binding
line's limit at its shadow price. Equivalently, it nets the bus-level payments:

$$R=\sum_{m\in\mathcal M_{\mathrm{act}}}|\mu_m|\,|F_m|
   =\underbrace{\sum_{n\in\mathcal N}\lambda_n d_n}_{\text{paid by load}}
   -\underbrace{\sum_{n\in\mathcal N}\lambda_n g_n}_{\text{paid to generation}}. \qquad (7)$$

*In words.* The congestion rent $R$, the merchandising surplus, is the absolute
shadow price $|\mu_m|$ times the absolute flow $|F_m|$, summed over the activated
lines. It equals what all load pays, $\sum_n\lambda_n d_n$, minus what all
generation is paid, $\sum_n\lambda_n g_n$, because the energy components net to
zero when total generation equals total load. With no binding line every $\mu_m$
is zero, no prices separate, and $R=0$.
</example>

<example id="NB-08" part="the EDIT lead-in">
**Edit below:** generator costs, load levels, and line ratings. Limiting a line rating may
create congestion and price separation on the constrained
path.
</example>

<example id="NB-09" part="the prose under an output">
With `line_4` (6–7) limited to 40 MW, the binding internal limit
splits the LMPs. The bus-7 load behind the constraint prices up to ~\$37 while the cheap bus-3 pocket (\$20) stays low. The congestion term is exactly $\sum_m SF_{n,m}\,\mu_m$. Setting `LINE_RATINGS = {}` and re-running collapses every bus
back to a single energy price.
</example>

<example id="NB-10" part="a section opener that tightens a construct">
## 3. Transport constraints and transfer rent

The canonical formulation in Section 2 already carries the scheduled transfers,
eq. (4). However, in section 2 the transfer constraint was not binding. Here it is tightened until it binds. The buses are grouped into two
regions and the scheduled transfer, $t_{ab}$,
between them is capped at a released capacity $\bar T_{ab}$:

$$0 \;\le\; t_{ab} \;\le\; \bar T_{ab} \qquad [\,\mu_T^{ab}\,].$$

The declared interface set $\mathcal I$ holds this single interface, in both
orientations.

When the limit binds it carries a shadow price $\mu_T^{ab}=\hat\lambda_b-\hat\lambda_a$. By eq. (6) every bus prices off its own region's balance, so the two regions' energy
prices separate by exactly $\mu_T^{ab}$. The **transfer rent** is

$$R_T \;=\; \mu_T^{ab}\, t_{ab} ,$$

the transport analogue of line congestion rent.
</example>

<example id="NB-11" part="a mechanism in plain prose">
## 4. Self-schedules

A **self-schedule** is a fixed, price-taking injection $\mathrm{exo}_n$ the clear
must absorb, a quantity with no bid price. It shifts a region's residual supply
curve and moves the LMPs without setting one. Below, a 150 MW self-schedule
delivered into bus 9 displaces local supply one-for-one. That is enough to back
the marginal \$35 unit out of merit entirely, so the cheap \$20 unit becomes
marginal and the system LMP falls from \$35 to \$20. A smaller schedule that
leaves the marginal unit unchanged, 80 MW for instance, moves quantities only and
no price. (`solve_engine_dispatch(exo=…)` carries it; `+` = injection into the
footprint.)
</example>

<example id="NB-12" part="the sandbox introduction">
## 5. Sandbox

Configuring a scenario with the
line ratings (congestion), an optional released transfer capacity on the region
interface (transport), and a self-schedule. Running the solver finds the optimal
dispatch, the congestion rent $R$, and the transfer rent $R_T$. [A later notebook] settles the
same sandbox through the full ledger chain.
</example>

<example id="NB-13" part="the configuration view, a figure that is not a clearing">
## Configuration

An orienting graphic uses the same two panels every figure in this notebook uses,
carrying only what is fixed about the case. Nothing here is a clearing.

On the network, buses take their palette color and lines carry their rating. There are no net injections, no flow arrows, and no prices. On the ring, each bus's supply sector is its **full nameplate capacity** at its offer price, drawn solid, with no distinction between dispatched and idle. Its demand sector is its **load**, drawn to the height of the most expensive unit in the fleet so the supply and demand tracks share one price axis. There are no chords, because nothing has been scheduled yet.

The price and volume arrows sit at the top of the ring: radius reads as price, arc length as megawatts. That is the convention the rest of the notebook builds on.
</example>

</examples>
