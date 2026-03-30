# Probe-Based Directional Credit Assignment

This repository accompanies the research note:

**Richard Vermillion (2026). _Probe-Based Directional Credit Assignment_.**  
Zenodo. https://doi.org/10.5281/zenodo.19332672

## Overview

This repo is for a framework proposal on **delayed credit assignment without full gradient transport**.

The central idea is to treat learning as an active process of local experimentation:

- each perturbable module maintains a small **probe dictionary** of candidate local directions,
- an **assignment matrix** selects probe directions across modules to instantiate a batch of nearby counterfactual trajectories,
- branch-specific delayed **modulatory signals** weight those sampled directions,
- the resulting **directional traces** are accumulated over time and collapsed into parameter updates,
- a **scheduler** allocates limited perturbation budget across modules, directions, and trajectories.

The intended setting is one in which full end-to-end gradients are unavailable, undesirable, biologically implausible, too expensive, or simply not the right abstraction.

## Status

This is an early-stage research repository accompanying a **proposal / research note**, not a validated empirical result.

What is here now:

- the published research note,
- supporting source files,
- and, over time, prototype code for implementation primitives such as `ProbedLinear`.

What is **not** here yet:

- full experiments,
- ablation studies,
- claims of empirical superiority,
- or a finalized production training system.

The goal is to make the framework concrete enough to inspect, discuss, cite, critique, and extend.

## Core ideas

The note is organized around a small set of central objects:

- **Probe Dictionary**  
  A per-module set of candidate local perturbation directions, including a null probe.

- **Assignment Matrix**  
  A batchwise experimental design specifying which probe each module uses in each sampled counterfactual branch.

- **Directional Trace**  
  A weighted summary of recently sampled local directions, preserving more structure than scalar eligibility alone.

- **Modulatory Signal**  
  A branch-specific scalar evaluative signal that reinforces or suppresses sampled directions.

- **Scheduler**  
  A global perturbation policy that allocates limited exploration budget across modules and trajectories.

Together, these define a framework in which the batch dimension is used not just for unrelated data points, but as a parallel evaluation axis over structured counterfactual branches.

## Research note

The canonical published version is:

- **DOI:** https://doi.org/10.5281/zenodo.19332672

If you only want the paper, start there.

## Repository contents

Expected contents of this repo include:

- manuscript source files for the research note
- bibliography/source files used to build the note
- prototype implementations of core abstractions (forthcoming)
- small diagnostic examples or notebooks as they are developed

The exact code layout may evolve as the implementation becomes clearer.

## Planned code

A likely first implementation target is a reusable `ProbedLinear` primitive:

- shared base weight matrix,
- bank of low-rank probe directions,
- batchwise probe assignment,
- null-probe support,
- and collapse of credited probe directions into an update on the shared weights.

This is intended as a concrete substrate for experimenting with the framework before building a larger end-to-end system.

## Citation

Please cite the published research note:

> Richard Vermillion. 2026. _Probe-Based Directional Credit Assignment_. Zenodo. https://doi.org/10.5281/zenodo.19332672

## License

- code: MIT / Apache-2.0
- paper and text: CC BY 4.0

## Contact / notes

This repository documents an evolving line of research. Expect some movement in naming, organization, and implementation details as the framework is refined.