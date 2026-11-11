# WSCC 9-bus seams examples

This repository holds the 9-bus examples Paul Dockery presented at the CAISO-SPP joint market seams workshops of 2026, one notebook per deck, a fundamentals notebook and a sandbox notebook. The network is the WSCC/IEEE 9-bus test case (`pandapower.networks.case9`), cleared as a lossless DC dispatch with PTDF shift factors, with two markets on one network. Each deck notebook reproduces its deck's examples in slide order. Each notebook re-executes from its own cells, and its knobs sit in `# -- EDIT --` cells.

## The workshops

| date | workshop | deck excerpt | notebook | Colab |
|---|---|---|---|---|
| 2026-08-18 | No. 2, Salt Lake City: *Congestion Management Examples* | [`2026-08-18_congestion_management_examples.pdf`](sources/workshops/2026-08-18_congestion_management_examples.pdf), pages 91 to 109 of the [posted deck](https://www.westernenergymarkets.com/documents/presentation-market-seams-workshop-aug-18-2026.pdf) | [`workshop2_congestion_management.ipynb`](workshop2_congestion_management.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pdockery/wscc9-illustrative-model/blob/main/workshop2_congestion_management.ipynb) |
| 2026-09-16 | No. 3, Lakewood: *Transactions across market boundaries* | link-only, pages 1 to 34 of the deck in the [posted slide-deck zip](https://spp.org/documents/77788/joint%20market%20seams%20workshop%20slide%20decks%2020270916.zip) from the [workshop page](https://spp.org/calendar-list/joint-market-seams-workshops-20260916/), marked "ISO Internal"; SPP posts the zip under the date 20270916 | [`workshop3_transactions_across_boundaries.ipynb`](workshop3_transactions_across_boundaries.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pdockery/wscc9-illustrative-model/blob/main/workshop3_transactions_across_boundaries.ipynb) |
| 2026-10-22 | No. 4, Salt Lake City: *Western Loading Relief process examples* | [`2026-10-22_loading_relief_examples.pdf`](sources/workshops/2026-10-22_loading_relief_examples.pdf), pages 52 to 86 of the [posted deck](https://www.westernenergymarkets.com/documents/presentation-joint-market-seams-workshop-oct-22-2026.pdf) | [`workshop4_loading_relief.ipynb`](workshop4_loading_relief.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pdockery/wscc9-illustrative-model/blob/main/workshop4_loading_relief.ipynb) |
| any | the fundamentals | none | [`101_fundamentals.ipynb`](101_fundamentals.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pdockery/wscc9-illustrative-model/blob/main/101_fundamentals.ipynb) |
| any | the sandbox | none | [`sandbox.ipynb`](sandbox.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pdockery/wscc9-illustrative-model/blob/main/sandbox.ipynb) |

`sources/workshops/manifest.json` records each deck's posted source, its page range, its access date and the SHA-256 of the source and the excerpt. A deck notebook's figures carry no titles or legends, because the slide beside them carries those.

## The fundamentals

`101_fundamentals.ipynb` states the clearing of a nodal energy market at its simplest, in five equations, and clears it three times. Example 1 is one market with no binding line, Example 2 rates line 6-7 at 40 MW, and Example 3 clears the same market as two balancing authority areas. Each example draws the network and nodal dispatch figure and prints the nodal prices, each binding line's shadow price, and each area's production cost and what its load pays. Its closing sandbox clears two markets independently on the same network, each enforcing only its own lines. A reader new to the case starts there.

## The sandbox

`sandbox.ipynb` exposes the engine's configurable features in one `# -- EDIT --` cell, and its closing section lists the features it does not carry. The EDIT cell sets the ratings, the fleet and the loads, the offer form, one market or two independent markets on the same network, the monitored and activated line sets, the interfaces and transfers, self-schedules, export bids and import offers, a scheduling point with an interchange scheduling limit at a transition bus, the real-time relief, the tags and the loading relief, and the congestion and transfer allocation methods. A new example belongs there, so that each deck notebook stays on its deck.

## Running the notebooks

**On Colab.** The badge opens the notebook. Its first cell installs `pypsa`, `pandapower` and `pycirclize` and fetches the 14 library modules from this repository's `lib/` on `main`.

**Locally.** Python 3.10 or later.

```
pip install -r requirements.txt
jupyter lab
```

**For development.** The tested environment, the commit hooks and the readiness report, once after cloning:

```
pip install -r requirements-dev.txt -c constraints.txt
git config core.hooksPath .githooks
python tools/doctor.py
```

`constraints.txt` pins every package of the tested environment, and `requirements.txt` stays unpinned for Colab. The commit hooks of `.githooks/` refuse an AI authorship trailer, a file over 50 MB and a source the repository keeps link-only. `tools/doctor.py` reports whether the interpreter, its packages, the library copy and the hooks path are ready, and changes nothing. `LAB_PY`, where it is set, names the interpreter it examines.

**Headless.** A notebook re-executes in place and records the library state it ran under in its metadata.

```
python tools/nb_execute.py workshop2_congestion_management.ipynb
python tools/nb_staleness.py
```

`tools/nb_staleness.py` lists the notebooks whose stored results predate a library change. `tools/README.md` describes each tool. `MPLBACKEND=Agg` is not set for a run, because it strips the stored figures.

**Tests.** `python tools/check.py` runs the lint of the changed Python files, the harness tests and the model tests, from the repository root. `python tools/check.py --full` adds the tests marked `slides`, which execute each deck notebook and check that its clears give every number its slides print. The suites also run alone: `python -m pytest tests -q`, and `python -m pytest tests -q -m slides` for the slide checks.

## The library

`lib/` holds the 14 modules of a shared 9-bus library, copied in byte for byte. `lib/MANIFEST.md` records each module's SHA-256 and the date of the copy, and a change arrives as a fresh copy. `seams_engine.py` is the engine, and its module docstring maps the code to the notation the notebooks use. `NOTEBOOK_RECORD.md` holds the record of the notebooks and an index of the library's public functions, module by module.

## The sources

`sources/references.bib` holds every source the notebooks cite, each entry with its `url` and a `note` on its bearing. Public documents of the ISOs, RTOs, FERC, NERC, NAESB and the workshop hosts are tracked under `sources/`. Academic papers and books are link-only, so their entries carry the `url` and no file is committed. `sources/excerpts/<key>.md` holds the passages the notebooks rest on, verbatim, with each page and the notebook section that uses it.

## Licence

The code is under the MIT licence in `LICENSE`, whose scope paragraph names what it does not cover. The documents under `sources/` stay under their publishers' terms.

## Working with an agent

`AGENTS.md` orients any agent tool in the repository, and `CLAUDE.md` imports it and adds the Claude Code layer of skills, hooks and permissions. `prompts/01-orientation.md` is the page for a person joining, with the prompt that starts a session in any agent tool and `/orientation` in Claude Code. The starters beside it invoke the skills in `.claude/skills/`, one per kind of work: explaining an example, re-running a notebook, changing a notebook or the sandbox, and staging named paths. In Claude Code a state brief opens each session, and an advisory clock-out check runs when a session stops or compacts.
