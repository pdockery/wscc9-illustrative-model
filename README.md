# WSCC 9-bus illustrative models

Small, self-contained teaching notebooks that illustrate nodal pricing, congestion
rent, and market-seams concepts on the classic **WSCC / IEEE 9-bus** network. They
accompany the working paper *Market Seams in the Western Interconnection* (Dockery, 2026).

Each notebook builds the 9-bus case from `pandapower`'s built-in network, clears a
DC-OPF with PTDF shift factors, and renders matching network + circlize/chord figures.
There are **no external data files** — everything is generated in-notebook.

## Run it in your browser (no install)

Click a badge, then **Runtime → Run all**. The first cell installs the few extra
packages and pulls in the helper modules automatically.

- **Congestion-revenue allocation on the WSCC 9-bus network**  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pdockery/wscc9-illustrative-model/blob/main/congestion_revenue_allocation_example.ipynb)  
  `congestion_revenue_allocation_example.ipynb`

- **Market-seams illustrations on the WSCC 9-bus network**  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pdockery/wscc9-illustrative-model/blob/main/seams_examples.ipynb)  
  `seams_examples.ipynb`


## What's here

| File | Role |
|------|------|
| `congestion_revenue_allocation_example.ipynb` | Two-BA congestion-revenue allocation methodology + scenarios |
| `seams_examples.ipynb` | Nodal pricing / seams illustrations |
| `ieee9_network.py` | Builds the IEEE 9-bus PyPSA network |
| `seams_engine.py` | PTDF DC-OPF clearing engine (`scipy.linprog`) + LMP decomposition |
| `nodal_plot.py` | Network-topology and circlize/chord figures |
| `requirements.txt` | Extra packages (Colab ships the rest) |

## Run it locally

```bash
pip install -r requirements.txt
jupyter lab        # open either .ipynb and Run All
```

Python 3.10+; the notebooks use `numpy`, `pandas`, `scipy`, `matplotlib`,
`pypsa`, `pandapower`, and `pycirclize`.
