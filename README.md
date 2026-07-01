# WSCC 9-bus illustrative models

A **numbered series of self-contained teaching notebooks** that build nodal pricing,
congestion rent, and market-seams concepts from the ground up on the classic **WSCC
9-bus** network. Each notebook builds the 9-bus case from `pandapower`'s built-in
network, clears a DC-OPF with PTDF shift factors, and renders matching network +
nodal-dispatch (chord) figures. No external data files are needed.

The notebooks share a set of small teaching libraries, so the modelling stays consistent
across the series and a student can read a notebook top-to-bottom or open the library to
see how a piece works. The numbering is a learning pathway — the **hundreds digit is the
difficulty tier** (100 fundamentals, 200 core issue, 300 advanced) and the **last digit
is the track**: x01 congestion-revenue allocation, x02 market seams (and the bilateral
scheduling that precedes a centralized market, 102), x03 transmission service, x11
available transfer capability, x22 intertie scheduling.

## Run it in your browser (no install)

Click a badge, then **Runtime → Run all**. The first cell installs the few extra
packages and pulls in the helper modules automatically.

### Fundamentals (start here)

- **101 · Nodal market fundamentals** — nodal LMPs and congestion rent, the transport
  (net-interchange) constraint and transfer rent, self-schedules, the three settlement
  ledgers, and a fully configurable sandbox.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pdockery/wscc9-illustrative-model/blob/main/101_nodal_market_fundamentals.ipynb)

- **102 · Bilateral trading as a double auction** — a numerical model of the bilateral
  world, which has **no central clearing**: it produces the day-ahead pre-schedule (the
  bilateral analogue of unit commitment) as a **repeated discriminatory double auction**
  (Nicolaisen–Petrov–Tesfatsion). Over rounds of back-and-forth, sellers and buyers draw
  offers from their perceived price distributions and concede until the must-serve load
  procures its **full** volume; each pair settles at the bid-ask **midpoint**. Three levers,
  one section each: the **number of parties** (liquidity — more parties tighten the spread,
  Rustichini–Satterthwaite–Williams / Gode–Sunder); **risk aversion and the price-risk
  distribution** (the two-sided forward premium, Bessembinder–Lemmon — each generator a
  single fixed marginal cost); and **real-time balancing** — the rights-feasible schedule
  overloads the grid, and a causer-pays **redispatch** restores feasibility. Mirrors 101's
  formal exposition. Builds on `atc.py` + `bilateral.py`.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pdockery/wscc9-illustrative-model/blob/main/102_bilateral_spot_iteration.ipynb)

- **111 · ATC and the simultaneous-feasibility test** — the OATT / contract-path world
  that precedes nodal pricing: each generator first **plans its portfolio** of rights
  to profitable load centers, then point-to-point service, Available Transfer Capability
  (`ATC = TTC − TRM − ETC − CBM`), and the simultaneous-feasibility test a *set* of
  awards must pass. Shows how ATC posted path-by-path, and ATC summed across two
  balancing authorities, **oversubscribes** the feasible set through loop flow.
  Builds on `atc.py` + `bilateral.py`.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pdockery/wscc9-illustrative-model/blob/main/111_atc_simultaneous_feasibility.ipynb)

- **112 · Three balancing authorities** — a third BA and single-node BAs (co-located
  generation and load); three-BA dispatch, per-BA settlement, and autarky-vs-unified.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pdockery/wscc9-illustrative-model/blob/main/112_three_ba_fundamentals.ipynb)

### Core issues

- **201 · Congestion-revenue allocation** — two balancing authorities; how congestion
  (and transfer) revenue is allocated, Method 1 vs Method 2.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pdockery/wscc9-illustrative-model/blob/main/201_congestion_revenue_allocation.ipynb)

- **202 · Market seams** — two market footprints on the shared grid; the three seam
  issues (dispatch interference, inefficient accommodation, participant-initiated
  interchange), the risk a trader carries, and the seam ledger.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pdockery/wscc9-illustrative-model/blob/main/202_market_seams.ipynb)

- **212 · Two markets and a transfer** — two markets on three BAs: Market A is two
  non-connected BAs coordinated by a transfer, which Market B wheels; the transaction
  P&L, inefficient accommodation, the risk the transmission customer carries, the
  parallel-timing problem, and a sandbox.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pdockery/wscc9-illustrative-model/blob/main/212_two_market_transfer.ipynb)

### Advanced

- **301 · Two settlement footprints** — the moving day-ahead (EDAM) / real-time (WEIM)
  market boundary; the exogenous→endogenous transition, the cross-settlement position
  ledger, a contingency, and its accommodation.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pdockery/wscc9-illustrative-model/blob/main/301_two_settlement_congestion.ipynb)

- **302 · A contingency in the neighbouring market** — two co-equal markets and a
  contingency on the other market's circuit; blind interference, inefficient
  accommodation, and coordination, each shown as four views (Market A, Market B, the
  Combined wire, and the Unified clearing).
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pdockery/wscc9-illustrative-model/blob/main/302_two_market_contingency.ipynb)

## What's here

| File | Role |
|------|------|
| `101_nodal_market_fundamentals.ipynb` | Fundamentals: nodal LMPs, congestion + transfer rent, self-schedules, ledgers, sandbox |
| `102_bilateral_spot_iteration.ipynb` | Fundamentals: bilateral trading as a repeated double auction (full procurement), levers for liquidity / risk premium, causer-pays real-time balancing |
| `111_atc_simultaneous_feasibility.ipynb` | Fundamentals: portfolio planning, ATC, the simultaneous-feasibility test, combined-ATC oversubscription |
| `112_three_ba_fundamentals.ipynb` | Fundamentals: a third BA and single-node BAs |
| `201_congestion_revenue_allocation.ipynb` | Two-BA congestion-revenue allocation (Method 1/2) |
| `202_market_seams.ipynb` | Two markets; three seam issues + trader risk + seam ledger |
| `212_two_market_transfer.ipynb` | Two markets on three BAs; transfer, wheeling, accommodation, timing risk, sandbox |
| `301_two_settlement_congestion.ipynb` | Moving day-ahead/real-time boundary; contingency + accommodation |
| `302_two_market_contingency.ipynb` | Two markets; a contingency in the other market; four-view comparison |
| `wscc9_model.py` | Network, fleet/loads, the market-engine factory, layout constants |
| `footprints.py` | Footprint partitions (balancing authorities / markets) and line assignment |
| `revenue_allocation.py` | Settlement, congestion/transfer-rent allocation, position ledgers |
| `atc.py` | Available Transfer Capability, path TTC, and the simultaneous-feasibility test (used by 102/111) |
| `bilateral.py` | The repeated bilateral double auction (`double_auction_clear`, forward risk, causer-pays redispatch) and the Cournot must-serve model + portfolio planning (used by 102/111) |
| `wscc9_figures.py` | The shared network + nodal-dispatch composite figures |
| `ieee9_network.py` | Builds the IEEE/WSCC 9-bus PyPSA network |
| `seams_engine.py` | PTDF DC-OPF clearing engine (`scipy.linprog`) + LMP decomposition |
| `nodal_plot.py` | Network-topology and nodal-dispatch (chord) figures |
| `requirements.txt` | Extra packages (Colab ships the rest) |

## Run it locally

```bash
pip install -r requirements.txt
jupyter lab        # open any numbered notebook and Run All
```

Python 3.10+; the notebooks use `numpy`, `pandas`, `scipy`, `matplotlib`, `pypsa`,
`pandapower`, and `pycirclize`.
