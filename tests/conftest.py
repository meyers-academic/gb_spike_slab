"""
Pytest configuration and fixtures for gb_spike_slab tests.
"""

import jax

# Configure JAX for 64-bit precision (applies to all tests)
jax.config.update("jax_enable_x64", True)

