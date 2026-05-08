# Search modes

| Mode | Switch | What it does | Cost |
|---|---|---|---|
| **Stepwise** (default) | `FullSearch=false` | Greedy hill-climb on (p,q,P,Q) starting from `Order/Seasonal`. Each step fits 4 neighbours in parallel. | Fast — ~10-40 fits |
| **Full search** | `FullSearch=true` | Cartesian product over (0..MaxP) × (0..MaxQ) × (0..MaxCapP) × (0..MaxCapQ). | Slow but thorough |
| **Approximation** | `Approximation=true` | Search at `Method=MethodCSS` (cheap), then refit picked order at user's actual Method. Mirrors R's `approximation=TRUE`. | ~40% faster than full CSSML |

R defaults to stepwise + approximation on large series (n>150 or
m>12). goarima keeps Approximation off by default — explicit opt-in
matches the documented Method (no surprise speedups that change
estimates).

See [`../policies/approximation.md`](../policies/approximation.md) for
GAP-2 details.
