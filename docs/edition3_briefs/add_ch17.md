# Brief: additions to Chapter 17 "Optimizers and the Training Loop" + lab extension

Read first: BRIEF_conventions.md (all, esp. §7), the existing chapter book/chapters/ch17_optimizers.md (keep all
content), its solutions file and code/labs/lab17_optim/*.


## Gap-check additions (owner request, 2026-11-05). Depth rules: BRIEF_conventions.md §7
- Memory-saving optimisers: 8-bit Adam (block-wise quantised states, dynamic exponent), Adafactor's factored second
  moments, low-rank/projection methods (GaLore) with their memory accounting; schedule-free optimisation (Defazio et
  al.: interpolation between iterate and average, why no decay phase is needed) implemented and compared with WSD.
- Averaging: Polyak/EMA weights during training and their evaluation benefit; what averaging costs in memory.
