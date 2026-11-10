---
key: caiso_edam_techdesc
file: sources/ferc-filings-and-iso-policy/caiso_edam_techdesc.pdf
url: https://stakeholdercenter.caiso.com/StakeholderInitiatives/Extended-day-ahead-market
redistribution: tracked
read: 2026-11-07
---

## p. 25 (PDF p. 27), section 10

> The marginal prices for the various commodities in EDAM are derived from the shadow prices of binding constraints.

Used by: 101_sandbox.ipynb, workshop3_transactions_across_boundaries.ipynb. Supports: Prices come from the shadow prices of binding constraints, which is how a node outside a footprint is priced in Example 1.

## p. 26 (PDF p. 28), section 10

> Therefore, the marginal prices are derived as follows:

Used by: 101_sandbox.ipynb, workshop2_congestion_management.ipynb, workshop3_transactions_across_boundaries.ipynb, workshop4_loading_relief.ipynb. Supports: The price formula that follows this sentence divides the energy price by the loss penalty factor and subtracts the shift-factor-weighted shadow prices of the binding network constraints. The formula is typeset as equations on PDF p. 28 and is read from the rendered page.

## p. 10 (PDF p. 12), notation

> Shift factor for resource i on network constraint m in time period t.

Used by: no notebook; kept for the explanation of the shift factor of a resource on a network constraint. Supports: Defines the shift factor on a network constraint, the quantity that prices a transaction on a constraint.

## p. 7 (PDF p. 9), notation and section 3.3

> Generalized network constraint index, including intertie scheduling limits.
>
> The scheduling limit on a transfer indicates the amount of transfer capacity that is released in the EDAM for optimal scheduling.

Used by: 101_sandbox.ipynb. Supports: Intertie scheduling limits sit in the same constraint index as network constraints, and a transfer limit is released capacity.

## p. 5 (PDF p. 7), section 3.1

> The exception are transfers associated with transmission contracts that can be self-scheduled to exercise the respective scheduling right.

Used by: no notebook; kept for the explanation of the self-scheduling of transfers that carry transmission contracts. Supports: The only place the description allows a self-scheduled transfer. It states no price treatment for the self-schedule.

## p. 15 (PDF p. 17), section 5.6, Net Export Transfer Constraints (the heading is on PDF p. 16)

> For imports from non-EDAM BAAs, the available supply capacity is further constrained by the applicable intertie scheduling limit, considering high priority (price-taker) exports to non-EDAM BAAs.

Used by: no notebook; kept for the explanation of the treatment of exports to areas outside EDAM. Supports: Names high-priority exports to a non-EDAM area as price-takers.
