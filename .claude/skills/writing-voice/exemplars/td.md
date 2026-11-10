# Exemplars: the TD voice

The register of the California ISO's *Extended Day-Ahead Market Draft Technical Description* (Version 1.3, September 11, 2024; `caiso_edam_techdesc` in `sources/references.bib`), the named model for formulation sections, definitions, criteria and principle statements. Every passage is verbatim from the document, with PDF and printed page. The only edits are list items rejoined across PDF line breaks and running headers removed where a page break intruded; the source's own spellings and typos are kept. Pick three to five by the part being drafted, never by topic.

| id | document part | printed page |
|---|---|---|
| TD-01 | a numbered classification in place of a prose list | 1 |
| TD-02 | a term defined in two senses, then used on named parties | 4 |
| TD-03 | areas and facilities named in full where a pronoun or an image would be easier | 5–6 |
| TD-04 | a design choice with its reason given inline | 6 |
| TD-05 | an explicit scope sentence with its paragraph | 6 |
| TD-06 | a procedure stated in the order the market runs it | 7 |
| TD-07 | a constraint stated entirely in prose before any symbol | 14 |
| TD-08 | the scope of an evaluation fixed with the constraint-set verbs (considered, ignored, enforced) | 18 |
| TD-09 | a displayed program read in words, with the pass or fail rule and a closing `Note that` | 19 |
| TD-10 | a `Note that` paragraph on an edge case | 21 |
| TD-11 | a displayed constraint read term by term | 23 |
| TD-12 | a counterfactual defined, its outcome named, its use stated | 24 |

<examples>

<example id="TD-01" part="numbered classification" page="PDF 3, printed 1, §1">
Although an EIM BAA may not join the EDAM, because of the intertwined settlement between the DAM and the RTM, all EDAM BAAs must also be participants in the WEIM. Therefore, in this paper the following classification is used for BAAs:

1) The CAISO BAA (CISO) whose BAA Operator (CAISO) is the Market Operator (MO).
2) An EDAM BAA whose BAA Operator (EDAM Entity) has joined the EDAM and the WEIM.
3) An EIM BAA whose BAA Operator (EIM Entity) has joined the WEIM, but not necessarily the EDAM.
4) An exclusively EIM BAA whose BAA Operator (EIM Entity) has joined the WEIM, but not the EDAM.
5) A non-EIM BAA whose BAA Operator has not joined either the EDAM or the WEIM.

A general reference to an EDAM BAA or an EIM BAA collectively includes the CISO. A general reference to a non-EDAM BAA collectively includes exclusively EIM and non-EIM BAAs. Figure 1 illustrates these membership relationships with a Venn diagram.
</example>

<example id="TD-02" part="a term defined, then used" page="PDF 6, printed 4, §3">
This section describes the various transfers that are supported in the EDAM. There are two different references to transfers:

1) A directional transfer between two BAAs in the market footprint. This transfer is modelled as a matching pair of export and import schedules at a pair of Transfer System Resources (TSRs) at the boundary of the two BAAs. The TSRs are associated with the intertie that is used for tagging the transfer. There may be multiple TSR pairs for multiple transfers between two BAAs at different interties, and even at the same intertie for multiple transfer types. A transfer from BAA A to BAA B is modeled as an export from BAA A to BAA B using an export TSR for BAA A, and a matching import to BAA B from BAA A using an import TSR for BAA B.
2) The algebraic net transfer of a BAA in the market footprint. This is the net of all export/import transfers from/to that BAA to/from other BAAs in the market footprint. The net transfer is algebraic, positive for export and negative for import. The sum of the net transfers of all BAAs in the market footprint nets to zero since they form a closed market system for transfers.

Figure 3 shows how transfers are modeled in EDAM.
</example>

<example id="TD-03" part="names in full, no pronoun, no image" page="PDF 7–8, printed 5–6, §3.2">
The concept of energy transfers is expanded in the EDAM to capacity transfers. As such, a capacity transfer from EDAM BAA A to EDAM BAA B is the transfer of capacity requirement from BAA B to BAA A, and a reservation of transfer capacity to dispatch it as energy in the WEIM from BAA A to BAA B, optimally or when needed. There are different capacity transfers for different capacity services:

1) A regulation up/down transfer from BAA A to BAA B is the export of regulation up/down from BAA A to BAA B. This is achieved by transferring an equal amount of regulation up/down requirement from BAA B to BAA A. Transmission capacity is reserved for this transfer on the associated intertie so that the Automatic Generation Control (AGC) of BAA A can provide a dynamic export/import to/from BAA B in the WEIM to regulate a negative/positive Area Control Error (ACE) in BAA B as if it occurred in BAA A. Note that an export regulation down transfer requires an equal amount of export energy transfer because it is dispatched as a dynamic import.
</example>

<example id="TD-04" part="a design choice with its reason inline" page="PDF 8, printed 6, §3.2">
4) A reliability capacity up/down transfer from BAA A to BAA B is the export of reliability capacity up/down from BAA A to BAA B. Transmission capacity is reserved for this transfer on the associated intertie so that reliability capacity up/down awards in BAA A can be dispatched optimally to provide a dynamic export/import transfer to/from BAA B in the WEIM to assist BAA B in meeting its demand forecast. Because there is a single deployment scenario in RUC where reliability capacity up and down awards are simultaneously dispatched, there is no reason to distinguish between reliability capacity up and down transfers; an export reliability capacity down transfer is equivalent to an import reliability capacity up transfer. Consequently, only reliability capacity up transfers are included in the mathematical formulation in this paper for simplicity, and they are generically referred to as reliability capacity transfers.
</example>

<example id="TD-05" part="an explicit scope sentence" page="PDF 8, printed 6, §3.2">
Energy and capacity transfers may coexist on the same transfer (TSR pair) defined between two BAAs in the market footprint because the associated TSRs may have both energy schedules and capacity awards, like any other resource. This is particularly true for energy and imbalance reserves because they are co-optimized in the IFM. However, the scope of the first implementation of the EDAM does not include the co-optimization of ancillary services, i.e., regulation up/down and spinning/non-spinning reserves, in EDAM BAAs other than the CISO. Therefore, ancillary services in EDAM BAAs other than the CISO, are self-provided and not optimized. The ancillary services transfers are RSE-eligible and they are used in the RSE to support bilateral contracts, on demand obligations, and reserve group agreements.
</example>

<example id="TD-06" part="a procedure in the order the market runs it" page="PDF 9, printed 7, §3.3">
The scheduling limit on a transfer indicates the amount of transfer capacity that is released in the EDAM for optimal scheduling. This is specified as an upper capacity limit for the relevant TSR by the relevant EDAM Entity. TSRs used for ancillary services transfers have a self-provision for the relevant service. Any remaining transfer capacity under the scheduling limit is used in the IFM to co-optimize energy and imbalance reserve transfers. Furthermore, any transfer capacity that remains available after the IFM is used in the RUC to schedule optimally reliability capacity transfers. The optimal transfer schedules are the energy schedule and capacity awards of the associated TSRs.

This design leverages the existing functionality for resource registration, bid submission and validation, outage/derate management, optimal scheduling and reporting, and settlement, to implement transfers in the EDAM. Capacity constraints are enforced for TSRs, like all resources, so that even when the underlying transfer capacity is made available for co-optimization among different market commodities, the optimal schedules and awards are feasible under the relevant scheduling limit. There is no overlap among energy and upward capacity transfers; however, downward capacity transfers overlap with energy transfers.
</example>

<example id="TD-07" part="a constraint stated in prose before any symbol" page="PDF 16, printed 14, §5.6">
Because of the economic relaxation of the imbalance reserve requirements via the imbalance reserve demand price curve, an EDAM BAA may come out of the EDAM being sort in meeting its uncertainty requirements, even if it passes the RSE. To preserve available supply capacity for meeting the balance of uncertainty requirements in the WEIM, instead of overly committing export transfers to other EDAM BAAs, the net export transfer may be limited in the IFM by a volumetric constraint. At a conceptual level, the net export transfer, beyond the committed net RSE-eligible transfer, is limited to the available supply capacity in excess of the sum of the RSE requirements and a configurable reliability margin. Furthermore, the available supply capacity from non-RSE-eligible resources is discounted by a confidence factor that reflects the confidence in these resources being available in the RTM.
</example>

<example id="TD-08" part="scope fixed with considered, ignored, enforced" page="PDF 20, printed 18, §8">
This section describes the resource sufficiency evaluation in the EDAM. Unlike the resource sufficiency evaluation in the WEIM, there is no balancing test, but a combined capacity and flexibility test simultaneously for all hours in the time horizon. The objective of the resource sufficiency evaluation is for each EDAM BAA to demonstrate sufficiency in resource capacity and flexibility before the EDAM to meet the BAA demand forecast, ancillary service requirements, and uncertainty requirements, collectively referred to as the RSE requirements. Only RSE-eligible resources are considered in the RSE. Transfers between EDAM BAAs are ignored with the exception of RSE-eligible transfers that are used to transfer requirements from the sink BAA to the source BAA. Transmission constraints are ignored, but all resource constraints, including inter-temporal constraints, are enforced.
</example>

<example id="TD-09" part="a displayed program read in words, with a closing Note that" page="PDF 21, printed 19, §8">
Where the resource constraints are not shown for simplicity. The objective is to find a feasible solution where all RSE shortfalls for each EDAM BAA are zero, i.e., all RSE requirements (demand forecast, ancillary services, and uncertainty), as adjusted by RSE-eligible transfers, are met. The optimality of the solution is not important, thus bid prices are not used at all. If all shortfalls for an EDAM BAA in an hour are zero, that BAA passes the RSE in that hour. Otherwise, if any upward shortfall (US, RUS, SRS, or NRS) takes value, the BAA fails the RSE in the upward direction, or if any downward shortfall (DS or RDS) takes value, the BAA fails the RSE in the downward direction, in the respective hour. An EDAM BAA may fail the RSE in both directions in a given hour. Note that a regulation down self-provided transfer forces an equal amount of an energy transfer in the same direction, hence it must be included in the RSE adjustment for the demand forecast and imbalance reserve requirements.
</example>

<example id="TD-10" part="a Note that paragraph on an edge case" page="PDF 23, printed 21, §9.2">
Note that a resource in a GHG regulation area may not have a GHG bid for that GHG regulation area, i.e., the corresponding GHG bid capacity is zero. The GHG regulation cost of the GHG regulation area that a resource resides in is included in its energy bid. This assumes that all energy produced by such resource is subject to the GHG regulation of its GHG regulation area, even if it is attributed to another GHG regulation area. Nevertheless, such resource may have a GHG bid and a resulting GHG attribution for another GHG regulation area; in this case, the GHG bid represents additional GHG regulation cost imposed by the GHG regulation area for that GHG attribution, which could be zero.
</example>

<example id="TD-11" part="a displayed constraint read term by term" page="PDF 25, printed 23, §9.4">
Where the first term in the max() function is the total resource capacity in the EDAM BAA that is contractually obligated to serve demand in GHG regulation areas, the second term is the gross RSE-eligible export transfers from the EDAM BAA to other EDAM BAAs that overlap with GHG regulation areas, and the third term is the net transfer of the BAA at the optimal solution of the previous iteration. The constraint is not enforced in the first iteration to initialize this value for the second iteration. Note that the constraint is linear because all the terms in the max() function are constant in any given iteration where the constraint is enforced.
</example>

<example id="TD-12" part="a counterfactual defined, its outcome named, its use stated" page="PDF 26, printed 24, §9.5">
This section describes the GHG reference pass in the EDAM. The GHG reference pass is a counterfactual of the IFM where there are no GHG attributions, and thus no GHG regulation cost. The important outcome of the GHG reference pass is the optimal counterfactual schedules of physical supply resources, which are used to limit the respective GHG attributions in the IFM (and the MPM-IFM). The counterfactual schedules reflect the optimal solution without net import into GHG regulation areas, hence no GHG regulation cost.
</example>

</examples>
