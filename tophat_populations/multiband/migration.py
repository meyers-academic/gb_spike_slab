"""
Source migration between bands.

After updating one parity of bands, sources whose frequencies have crossed
a band boundary must be transferred to the destination band. This happens
at the sync point before computing residuals for the opposite parity.
"""

import jax.numpy as jnp


def detect_migrants(bands):
    """
    Identify sources that have left their owning band's nominal range.

    Parameters
    ----------
    bands : list of Band
        Bands to check (one parity).

    Returns
    -------
    migrations : list of dict
        Each dict has keys:
        - 'origin_k': int, origin band index
        - 'slot_idx': int, index within origin band's source arrays
        - 'dest_k': int, destination band index
        - 'freq': float, source frequency
        - 'amp': float, source amplitude
        - 'phase': float, source phase
    """
    migrations = []
    for band in bands:
        # Check for sources that moved right
        right_mask = band.migrants_right_mask()
        if jnp.any(right_mask):
            for j in jnp.where(right_mask)[0]:
                migrations.append({
                    'origin_k': band.k,
                    'slot_idx': int(j),
                    'dest_k': band.k + 1,
                    'freq': float(band.source_freqs[j]),
                    'amp': float(band.source_amps[j]),
                    'phase': float(band.source_phases[j]),
                })

        # Check for sources that moved left
        left_mask = band.migrants_left_mask()
        if jnp.any(left_mask):
            for j in jnp.where(left_mask)[0]:
                migrations.append({
                    'origin_k': band.k,
                    'slot_idx': int(j),
                    'dest_k': band.k - 1,
                    'freq': float(band.source_freqs[j]),
                    'amp': float(band.source_amps[j]),
                    'phase': float(band.source_phases[j]),
                })

    return migrations


def execute_migrations(migrations, all_bands):
    """
    Transfer migrant sources from origin to destination bands.

    Migrants are placed into an inactive (z=0) slot in the destination
    band. The origin slot is deactivated (z=0).

    Parameters
    ----------
    migrations : list of dict
        Output of detect_migrants.
    all_bands : list of Band
        All bands (both parities), indexed by band number.

    Returns
    -------
    n_migrated : int
        Number of successfully migrated sources.

    Raises
    ------
    RuntimeError
        If no free slot is available in the destination band.
    """
    band_lookup = {b.k: b for b in all_bands}
    n_migrated = 0

    for mig in migrations:
        origin = band_lookup[mig['origin_k']]
        dest_k = mig['dest_k']

        # Skip if destination band doesn't exist (boundary bands)
        if dest_k not in band_lookup:
            continue

        dest = band_lookup[dest_k]

        # Find a free slot (z=0) in the destination band
        free_slots = jnp.where(dest.z_indicators < 0.5)[0]
        if len(free_slots) == 0:
            raise RuntimeError(
                f"No free slot in band {dest_k} for migrant from band "
                f"{mig['origin_k']} (freq={mig['freq']:.6e}). "
                f"Increase n_templates_per_band."
            )

        target_slot = int(free_slots[0])

        # Transfer source parameters to destination
        dest.source_freqs = dest.source_freqs.at[target_slot].set(mig['freq'])
        dest.source_amps = dest.source_amps.at[target_slot].set(mig['amp'])
        dest.source_phases = dest.source_phases.at[target_slot].set(mig['phase'])
        dest.z_indicators = dest.z_indicators.at[target_slot].set(1.0)

        # Deactivate in origin
        slot_idx = mig['slot_idx']
        origin.z_indicators = origin.z_indicators.at[slot_idx].set(0.0)

        n_migrated += 1

    return n_migrated


def migrate_sources(updated_bands, all_bands):
    """
    Convenience function: detect and execute migrations for one parity.

    Parameters
    ----------
    updated_bands : list of Band
        Bands that were just updated (one parity).
    all_bands : list of Band
        All bands.

    Returns
    -------
    n_migrated : int
    """
    migrations = detect_migrants(updated_bands)
    if migrations:
        return execute_migrations(migrations, all_bands)
    return 0
