"""CBFpy: Control Barrier Functions in Python and Jax"""

import os
import warnings
from packaging import version

import jax

from cbfpy.cbfs.cbf import CBF
from cbfpy.cbfs.clf_cbf import CLFCBF
from cbfpy.config.cbf_config import CBFConfig
from cbfpy.config.clf_cbf_config import CLFCBFConfig


def check_env_vars():
    """
    If CBFpy code is running on a single CPU, for best performance, there are
    some environment variables that should be set. Here, we'll check for if
    those env vars are set and print an info message if they're not set
    """
    devices = jax.devices()
    jax_version = jax.__version__

    if (len(devices) == 1 and devices[0].platform == "cpu") or (
        os.environ.get("JAX_PLATFORMS", "") == "cpu"
    ):
        x64_enabled = os.environ.get("JAX_ENABLE_X64", "").lower() in ("1", "true")
        xla_flags = os.environ.get("XLA_FLAGS", "").split()
        single_thread_eigen = "--xla_cpu_multi_thread_eigen=false" in xla_flags
        single_thread_blas = os.environ.get("OPENBLAS_NUM_THREADS", "") == "1"
        before_jax_0_4_32 = version.parse(jax_version) < version.parse("0.4.32")

        msg = (
            "[cbfpy] CPU backend detected but some performance settings are not configured.\n"
            + "These are optional, but should lead to better precision and speed. "
            + "See the cbfpy README for more details."
        )
        if not before_jax_0_4_32:
            msg += (
                f"\n- Detected JAX version {jax_version}. "
                + "Consider using a version before JAX 0.4.32 for best CPU performance."
            )
        if not x64_enabled:
            msg += (
                "\n- JAX_ENABLE_X64 not detected. Recommendation: set JAX_ENABLE_X64=1"
            )
        if not single_thread_eigen:
            msg += (
                "\n- Single threaded Eigen configuration not detected. "
                + "Recommendation: set XLA_FLAGS='--xla_cpu_multi_thread_eigen=false'"
            )
        if not single_thread_blas:
            msg += (
                "\n- Single threaded BLAS configuration not detected. "
                + "Recommendation: set OPENBLAS_NUM_THREADS=1"
            )
        should_warn = (
            not before_jax_0_4_32
            or not x64_enabled
            or not single_thread_eigen
            or not single_thread_blas
        )
        if should_warn:
            warnings.warn(msg, stacklevel=2)


check_env_vars()
