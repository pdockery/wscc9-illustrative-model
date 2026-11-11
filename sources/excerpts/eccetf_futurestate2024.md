---
key: eccetf_futurestate2024
file: sources/ferc-filings-and-iso-policy/eccetf_futurestate2024.pdf
url: https://naesb.org/pdf4/R24005_attachment.pdf
redistribution: tracked
read: 2026-11-07
---

## p. 7

> Generation-to-Load Distribution Factor is the difference between a GSF and an LSF and determines the total impact of a generator serving its native BA load on an identified transmission facility or monitored element.
>
> For example, an e-Tag with a calculated TDF of 8.3% on an element indicates 8.3% of the MW amount on that e-Tag will flow on element X.

Used by: sandbox.ipynb, workshop4_loading_relief.ipynb. Supports: Defines the generation-to-load distribution factor and shows how a transfer distribution factor sizes an e-Tag’s flow on an element.

## p. 12

> When e-Tags or Dynamic Transfers are used to assign relief obligations, ECC will respect transmission priorities so that non-firm curtailments are requested prior to firm curtailments using a bucket approach.
>
> e-Tags or Dynamic Transfers included in the relief obligations will be subject to systematic curtailment.
>
> BAs participating in an organized market will be aggregated and their net GTL relief obligation will be issued for the entire market area.
>
> GTL components above the Impact Threshold will default to the highest transmission priority so that they are aligned with the firm e-Tags when determining relief obligation.
>
> BAs or market areas issued GTL relief obligations will need to redispatch their systems to achieve the relief obligation.

Used by: sandbox.ipynb, workshop4_loading_relief.ipynb. Supports: Curtailments walk the transmission priorities from non-firm to firm by bucket. A market area receives one net generation-to-load relief obligation, and the obligation is met by redispatch while tagged transfers are curtailed.

## p. 13

> The constraint represents an SOL on the BES with a limit of 500 MW and the post contingent flow is currently 600 MW. So, for this example 100 MW of relief is needed to return flows to the SOL.

Used by: workshop4_loading_relief.ipynb. Supports: Opens the worked bucket example, which runs from page 13 to page 15.

## p. 14

> Since this approach respects transmission rights, the curtailments would start with the lowest priority e-Tags.
>
> The firm e-Tag curtailments would be done on a pro rata basis calculated by the impact each e-Tag has on the constraint.

Used by: sandbox.ipynb, workshop4_loading_relief.ipynb. Supports: The worked example curtails from the lowest priority up, and the last bucket reached is cut pro rata by each e-Tag's impact on the constraint.

## p. 15

> Curtailments and relief obligations are seen in the charts below.

Used by: workshop4_loading_relief.ipynb. Supports: Opens the result of the worked example that runs from page 13 to page 15. The charts show e-Tag 1 below the 5 percent threshold and not curtailed, e-Tags 2 and 3 curtailed to 0 MW with 10 and 15 MW of relief, and e-Tags 4, 5 and 6 curtailed to 266, 166 and 165 MW with 24, 10 and 17 MW of relief. The generation-to-load relief obligations are 4 MW for BA A, 12 MW for Market 1 and 9 MW for Market 2. A footnote reads that the figures are rounded to whole numbers.
