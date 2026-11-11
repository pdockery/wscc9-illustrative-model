---
key: caiso_bpm_eim_v37
file: sources/ferc-filings-and-iso-policy/caiso_bpm_eim_v37.pdf
url: https://bpmcm.caiso.com/
redistribution: tracked
read: 2026-11-07
---

## p. 118

> The congestion revenue distribution factors for these constraints allocate the congestion revenue 100% to the BAA for which the constraint is formulated

Used by: no notebook; kept for the explanation of the area that receives a constraint's congestion revenue in the WEIM. Supports: A binding constraint pays its rent to the balancing authority area for which it is formulated.

## p. 120

> The congestion distribution factors for the ETSR transmission cost [...] are set to 50% for each of the two BAAs of the associated ETSR.
>
> The congestion distribution factors for the ETSR scheduling limits [...] are set to 50% for each of the two BAAs of the associated ETSR if they have transmission rights through the intertie (typical case for ETSRs between WEIM BAAs), or 100% to the BAA that has transmission rights to, but not through, the intertie (typical case for ETSRs between a WEIM BAA and the CISO).
>
> The congestion distribution factors for the intertie scheduling limit (ISL or ITC) [...] are set to 100% for the BAA that manages the limit.

Used by: sandbox.ipynb. Supports: The split of rent on an ETSR is 50/50 between the two areas, and the intertie scheduling limit pays the area that manages it.
