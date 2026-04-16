# Multi-Band Gibbs Sampler — Design Document

## Overview

The single-band model resolves sources and infers population hyperparameters within one frequency band. This document describes the extension to **multiple frequency bands** covering the full analysis range, using an **even/odd (red-black Gauss-Seidel) parallelisation scheme**.

The key constraint: each source has a top-hat template of width `w` in the frequency domain. Templates near band boundaries bleed into neighbouring bands, coupling adjacent bands. The even/odd scheme breaks this coupling by conditioning on one parity to update the other.

---

## 1. Band Layout

### 1.1 Nominal bands

Partition the analysis frequency range `[f_min, f_max]` into `N_bands` contiguous bands:

```
B_k = [f_k^low, f_k^high],   k = 1, ..., N_bands
```

where `f_k^high = f_{k+1}^low` (bands share boundaries, no gaps). Band width `Δf_band = f_k^high - f_k^low` should be much larger than the template width `w` so that most sources are fully contained within a single band.

### 1.2 Extended bands (with buffer)

Each band's likelihood needs to be evaluated over a region that extends `w/2` beyond the nominal boundaries, to capture sources whose templates straddle the boundary:

```
B_k^ext = [f_k^low - w/2,  f_k^high + w/2]
```

The buffer regions `[f_k^low - w/2, f_k^low]` and `[f_k^high, f_k^high + w/2]` overlap with the neighbouring bands.

### 1.3 Source ownership

A source with center frequency `f_j` **belongs to** band `k` if `f_j ∈ B_k` (the nominal range). Its template may extend into `B_{k-1}` or `B_{k+1}`, but its parameters are updated only when band `k` is updated.

---

## 2. Even/Odd Parallelisation

### 2.1 Why adjacent bands are coupled

If band `B_k` contains a source near `f_k^high`, its template leaks into `B_{k+1}`. When updating `B_{k+1}`, the likelihood depends on this source's contribution in the buffer zone. So `B_k` and `B_{k+1}` cannot be updated independently.

### 2.2 The red-black scheme

Label bands as **even** (`B_2, B_4, B_6, ...`) and **odd** (`B_1, B_3, B_5, ...`).

When updating even bands:
- All odd-band sources are **held fixed**
- Even band `B_k` is separated from the next even band `B_{k+2}` by the entire odd band `B_{k+1}`, which has width `Δf_band >> w`
- Therefore **even bands are conditionally independent** given odd-band sources
- All even bands can be updated **in parallel**

Then swap: update odd bands with even held fixed.

### 2.3 Condition for independence

Even bands `B_k` and `B_{k+2}` are independent if their extended regions don't overlap:

```
f_k^high + w/2  <  f_{k+2}^low - w/2
```

Since `f_{k+2}^low = f_{k+1}^high = f_k^high + Δf_band`, this requires:

```
Δf_band > w
```

This is the only requirement on the band width. In practice `Δf_band >> w` for computational efficiency (many sources per band).

---

## 3. Residual Computation

### 3.1 What the residual is

When updating band `B_k`, we need the frequency-domain data in the extended range `B_k^ext`. Some of that data contains contributions from sources owned by neighbouring bands. Since those sources are held fixed (opposite parity), we subtract them to get a **residual**:

```
r_k(f) = d(f) - Σ_{j ∈ neighbours, templates overlap B_k^ext} h_j(f)
```

where `d(f)` is the full data and `h_j(f)` is the template of fixed source `j`.

### 3.2 Which sources contribute to the buffer

For the **left buffer** of band `B_k` (the region `[f_k^low - w/2, f_k^low]`):
- Sources owned by `B_{k-1}` whose center frequency `f_j > f_k^low - w` (their right template edge reaches into the buffer)

For the **right buffer** of band `B_k` (the region `[f_k^high, f_k^high + w/2]`):
- Sources owned by `B_{k+1}` whose center frequency `f_j < f_k^high + w` (their left template edge reaches into the buffer)

In practice, only sources within distance `w` of the band boundary contribute. This is a small number of sources.

### 3.3 Residual as sufficient statistic

For the Whittle likelihood (periodogram-based), the residual enters only through the power in each frequency bin. So the "residual" we pass between bands is the **residual periodogram** (or equivalently the residual PSD) in the buffer region, not the full time series.

---

## 4. Gibbs Cycle

### 4.1 Full cycle

```
For each Gibbs iteration:

  1. EVEN BAND UPDATE (parallel over all even k):
     a. Receive fixed odd-band source contributions in buffer zones
     b. Compute residual in B_k^ext
     c. Update source parameters in B_k (amplitudes, frequencies, phases)
        using the Whittle likelihood on the residual
     d. Update z_k indicators (spike-and-slab)
     e. Compute S_conf(f_k) from unresolved sources in B_k

  2. MIGRATE EVEN → ODD (sequential, but cheap):
     a. For each even band, check if any source frequencies
        have left the nominal range [f_k^low, f_k^high)
     b. Transfer migrants to the destination band
        with current (A_j, f_j, φ_j, z_j) intact

  3. ODD BAND UPDATE (parallel over all odd k):
     a. Receive fixed even-band source contributions in buffer zones
        (migrants are NOT subtracted — they are now owned by odd bands)
     b–e. Same as step 1

  4. MIGRATE ODD → EVEN (sequential, but cheap):
     Same as step 2, roles swapped

  5. HYPERPARAMETER UPDATE (global):
     a. Collect S_conf(f_k) and N_res(f_k) = Σ z_j from all bands
     b. Update Λ = (N_tot, α, β) using the MDN likelihood
        (Section 5 of the MDN design doc)
```

### 4.2 What gets passed between parities

At each parity swap, two things happen:

**1. Migration** (Section 7.3): any sources whose frequencies have crossed a band boundary are transferred to the destination band. This is O(few sources) total — migration is rare.

**2. Buffer contributions**: for each band boundary, the templates of sources near the boundary that bleed into the neighbouring band. Concretely:

```python
# After updating even bands, prepare for odd update:
for each even band B_k:
    # Sources near the LEFT boundary that affect B_{k-1}
    left_buffer_sources = [j for j in B_k.sources 
                           if f_j < f_k^low + w]
    left_buffer_signal = sum(h_j(f) for j in left_buffer_sources 
                             for f in B_{k-1}.right_buffer_freqs)

    # Sources near the RIGHT boundary that affect B_{k+1}
    right_buffer_sources = [j for j in B_k.sources 
                            if f_j > f_k^high - w]
    right_buffer_signal = sum(h_j(f) for j in right_buffer_sources 
                              for f in B_{k+1}.left_buffer_freqs)
```

This is O(few sources) per boundary — very cheap.

### 4.3 Within-band update

Each band's internal update is identical to the single-band model:

1. **Source parameter update**: for each active source (z_j = 1), update (A_j, f_j, φ_j) using the Whittle likelihood on the residual within B_k^ext
2. **Indicator update**: for each template slot, update z_j ∈ {0, 1} using DiscreteHMCGibbs with modified=True
3. **Local S_conf**: compute the confusion noise PSD from unresolved (z_j = 0) sources in this band

The key difference from single-band: the likelihood is evaluated on the **residual** (data minus fixed neighbour contributions), not the raw data.

---

## 5. Implementation

### 5.1 Data structures

```python
@dataclass
class Band:
    k: int                     # band index
    f_low: float               # nominal lower frequency
    f_high: float              # nominal upper frequency
    w: float                   # template width
    source_freqs: jnp.ndarray  # center frequencies of sources in this band
    source_amps: jnp.ndarray   # amplitudes
    source_phases: jnp.ndarray # phases
    z_indicators: jnp.ndarray  # spike-and-slab indicators
    
    @property
    def f_low_ext(self):
        return self.f_low - self.w / 2
    
    @property
    def f_high_ext(self):
        return self.f_high + self.w / 2
    
    def left_buffer_sources(self):
        """Sources whose templates extend into B_{k-1}."""
        mask = self.source_freqs < self.f_low + self.w
        return mask
    
    def right_buffer_sources(self):
        """Sources whose templates extend into B_{k+1}."""
        mask = self.source_freqs > self.f_high - self.w
        return mask
```

### 5.2 Parallel update with JAX vmap

```python
def update_single_band(band_state, residual_in_extended_range, rng_key):
    """
    Update one band's sources and indicators.
    This function is pure (no side effects) and vmappable.
    
    band_state: parameters of sources in this band
    residual_in_extended_range: data minus fixed-neighbour contributions
    rng_key: PRNG key
    
    Returns: updated band_state, buffer contributions for neighbours
    """
    # 1. Update source parameters via Whittle likelihood
    # 2. Update z indicators
    # 3. Compute S_conf from unresolved sources
    # 4. Compute buffer contributions for neighbours
    ...

# Update all even bands in parallel:
even_keys = jax.random.split(rng_key, n_even_bands)
updated_even, even_buffers = jax.vmap(update_single_band)(
    even_band_states, even_residuals, even_keys
)
```

### 5.3 Residual computation

```python
def compute_residuals(data_psd, bands, fixed_parity_bands):
    """
    Compute residuals for each band to be updated,
    subtracting contributions from fixed-parity neighbours.
    
    IMPORTANT: call this AFTER migration. Migrants now belong to the
    bands being updated, so they are part of the model and NOT subtracted.
    
    data_psd: (N_freq_bins,) full-band periodogram
    bands: list of bands to update (e.g. even bands)
    fixed_parity_bands: list of bands held fixed (e.g. odd bands)
    
    Returns: list of residual PSDs, one per band in bands
    """
    residuals = []
    for band in bands:
        # Start with data in extended range
        freq_mask = (freq_grid >= band.f_low_ext) & (freq_grid <= band.f_high_ext)
        residual = data_psd[freq_mask].copy()
        
        # Subtract left neighbour's contributions in left buffer
        left_neighbour = fixed_parity_bands[band.k - 1]  # if exists
        if left_neighbour is not None:
            buffer_mask = left_neighbour.right_buffer_sources()
            for j in buffer_mask:
                residual -= template_power(left_neighbour, j, band.left_buffer_freqs)
        
        # Subtract right neighbour's contributions in right buffer
        right_neighbour = fixed_parity_bands[band.k + 1]  # if exists
        if right_neighbour is not None:
            buffer_mask = right_neighbour.left_buffer_sources()
            for j in buffer_mask:
                residual -= template_power(right_neighbour, j, band.right_buffer_freqs)
        
        residuals.append(residual)
    
    return residuals
```

---

## 6. Interaction with the MDN

After both even and odd updates complete, we have updated `S_conf(f_k)` and `N_res(f_k) = Σ z_j` for all bands. These feed into the MDN-based hyperparameter update exactly as described in the MDN design doc (Section 5):

```python
# Collect from all bands
S_conf_all = jnp.array([band.S_conf for band in all_bands])
N_res_all = jnp.array([jnp.sum(band.z_indicators) for band in all_bands])
f_centers = jnp.array([(band.f_low + band.f_high) / 2 for band in all_bands])

# MDN + Poisson likelihood for hyperparameter update
# (see MDN design doc Section 5.2)
```

The MDN was trained on band-averaged quantities, so it doesn't care about the even/odd mechanics — it just sees `(S_conf, N_res)` per band.

---

## 7. Edge Cases and Practical Notes

### 7.1 Boundary bands

The first band `B_1` has no left neighbour and the last band `B_N` has no right neighbour. Their outer buffer zones extend beyond the analysis range. Handle by zero-padding or truncating the extended range.

### 7.2 Sources exactly on the boundary

A source with `f_j` exactly equal to a band boundary `f_k^high` should be assigned to one band deterministically (e.g. always the right band: `f_j ∈ [f_k^low, f_k^high)` with left-closed right-open convention).

### 7.3 Source migration

If a source's frequency is updated and it moves outside its owning band's nominal range, it must be **transferred to the new band at the sync point** — not during the parallel update.

#### Why migration must happen at the sync point

Consider source `j` owned by even band `B_k`, sitting near the right boundary. During the even update, its frequency gets pulled across into `B_{k+1}` (an odd band). Two wrong approaches:

- **Subtract it when computing odd residuals**: `B_{k+1}` sees a residual with this source removed, but has no template slot for it. The source vanishes from the model. Bad.
- **Leave it in `B_k` and don't subtract**: `B_{k+1}` sees excess power at `f_j` in its residual. Nobody is modeling the source. It gets absorbed into `S_conf`. Bad.

The correct approach: **transfer ownership at the sync point**, before computing residuals for the opposite parity.

#### Migration protocol

```
After even update, before computing odd residuals:

  1. IDENTIFY MIGRANTS:
     For each even band B_k:
       migrants_right = sources with f_j >= f_k^high
       migrants_left  = sources with f_j < f_k^low

  2. TRANSFER:
     For each migrant:
       - Identify destination band (the band whose nominal range contains f_j)
       - Remove from origin band's source list
       - Add to destination band's source list
         with current (A_j, f_j, φ_j, z_j) intact
       - The migrant keeps its parameters — it was already updated
         by the origin band in this half-iteration

  3. COMPUTE RESIDUALS:
     Now proceed to compute residuals for odd bands.
     Migrants are owned by their destination bands (odd bands),
     so they are NOT subtracted from the residuals — they are part
     of the odd-band model and will be updated normally.

  Same protocol after odd update, before even residuals.
```

#### Template slot allocation

The spike-and-slab model uses a fixed number of template slots `N_templates` per band. Migrants need a free slot. Strategies:

1. **Over-allocate** (recommended): give each band a few extra slots (e.g. `N_templates = N_expected + N_spare` with `N_spare = 5-10`). Migration is rare (the matched filter constrains frequency tightly), so a small buffer suffices. Assert if a band ever fills up — this signals the spare allocation is too small.

2. **Use inactive slots**: migrants take over a slot that currently has `z = 0`. Since the spike-and-slab typically has many inactive slots, this is almost always possible. The migrant arrives with its own `z_j` value (usually 1, since active sources are the ones with enough SNR to have their frequency pulled).

Option 2 is clean and doesn't waste memory. In practice, combine both: use inactive slots first, fall back on spare capacity.

#### Timing and mixing

When a source migrates from even `B_k` to odd `B_{k+1}`:
- It was updated during the current even half-iteration (by `B_k`)
- It will be updated again during the current odd half-iteration (by `B_{k+1}`)
- So there is **no delay** — the source gets updated in both halves of the iteration

When migrating in the same direction (even-to-odd), the source gets two updates in one iteration. When migrating against (odd-to-even, caught at the even-to-odd sync), there's a one-half-iteration delay before the next update. Both are fine for MCMC mixing.

#### Expected migration rate

Migration requires a frequency update large enough to cross a band boundary. For well-resolved sources, the matched filter peak width is `~1/T_obs`, which is much smaller than `Δf_band`. Migration should be extremely rare — on the order of 1 in 10^3 to 10^4 iterations per boundary. If migration is frequent, `Δf_band` is too small.

### 7.4 Band width selection

Choose `Δf_band` to balance:
- **Parallelism**: more bands = more parallel units (good for GPU)
- **Overhead**: more bands = more buffer computations and communication
- **Sources per band**: need enough sources per band for the spike-and-slab to work well
- **Constraint**: `Δf_band > w` (required for even/odd independence)

A reasonable starting point: `Δf_band = 10w` to `100w`.

### 7.5 Synchronisation points

The global synchronisation points in each Gibbs iteration are:
1. After even update: migrate even→odd, then pass buffer contributions to odd bands
2. After odd update: migrate odd→even, then pass buffer contributions to even bands
3. After both parity updates: collect S_conf, N_res for hyperparameter update
4. After hyperparameter update: broadcast Λ for next iteration

Migration (steps 1–2) is sequential but extremely cheap — typically zero or a handful of sources per iteration across all boundaries. Within each parity update, all bands are independent and fully parallel.

---

## 8. Gibbs Iteration Summary

```
┌─────────────────────────────────────────────────────┐
│  EVEN UPDATE  (parallel over k = 2, 4, 6, ...)     │
│                                                     │
│  For each even band (in parallel):                  │
│    1. Subtract odd-neighbour templates from buffers │
│    2. Update source params on residual              │
│    3. Update z indicators                           │
│    4. Compute local S_conf                          │
├────────────── sync ─────────────────────────────────┤
│  MIGRATE EVEN → ODD                                 │
│                                                     │
│  For each even band:                                │
│    - Identify sources that left nominal range       │
│    - Transfer to destination band (odd neighbour)   │
│    - Migrants keep current (A, f, φ, z)             │
├─────────────────────────────────────────────────────┤
│  ODD UPDATE   (parallel over k = 1, 3, 5, ...)     │
│                                                     │
│  For each odd band (in parallel):                   │
│    1. Subtract even-neighbour templates from buffers│
│       (migrants are now owned by odd bands,         │
│        NOT subtracted — they're part of the model)  │
│    2. Update source params on residual              │
│    3. Update z indicators                           │
│    4. Compute local S_conf                          │
├────────────── sync ─────────────────────────────────┤
│  MIGRATE ODD → EVEN                                 │
│                                                     │
│  Same as above, roles swapped                       │
├─────────────────────────────────────────────────────┤
│  HYPERPARAMETER UPDATE  (global)                    │
│                                                     │
│  Collect S_conf(f_k), N_res(f_k) from all bands    │
│  Sample Λ via MDN prior + Poisson likelihood        │
│  Sample {log S_conf,k} as latent variables          │
├────────────── sync ─────────────────────────────────┤
│  → next iteration                                   │
└─────────────────────────────────────────────────────┘
```

---

## 9. Testing Checklist

- [ ] **Unit test: buffer identification** — place a source at `f_k^high - w/4`, verify it appears in the right buffer source list
- [ ] **Unit test: residual correctness** — with known source params, verify residual matches `data - templates` in the buffer zone
- [ ] **Unit test: even/odd independence** — with `Δf_band > w`, verify that modifying a source in `B_2` does not change the likelihood of `B_4`
- [ ] **Unit test: migration detection** — place a source at `f_k^high - ε`, update its frequency to `f_k^high + ε`, verify it is flagged as a migrant
- [ ] **Unit test: migration transfer** — verify migrant arrives in destination band with correct `(A, f, φ, z)` and is removed from origin band
- [ ] **Unit test: migrant not subtracted** — after migration, verify the migrant's template is NOT subtracted from the destination band's residual (it's part of the model)
- [ ] **Unit test: slot allocation** — verify migrants land in an inactive (z=0) slot or a spare slot, and that an assertion fires if the band is full
- [ ] **Regression test: single-band recovery** — set `N_bands = 1`, verify identical results to existing single-band code
- [ ] **Integration test: two-band** — synthetic data with sources straddling a boundary, verify both are recovered
- [ ] **Integration test: forced migration** — place a source near a boundary with a wide frequency prior, verify it migrates correctly and is still recovered after many iterations
- [ ] **Integration test: multi-band hyperparameters** — synthetic data across many bands, verify Λ posterior concentrates on truth
- [ ] **Performance test: scaling** — measure wall-clock time vs `N_bands` on GPU, verify near-linear parallel speedup for within-parity updates
