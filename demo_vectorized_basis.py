import time
from typing import List

from loguru import logger

from canonical import VectorizedCanonicalBasis
from logging_config import setup_logging
from symbolic import Variable

setup_logging()


def time_block(label: str):
    """Context manager to time a block of code."""

    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            logger.info(f"Starting: {label}...")
            return self

        def __exit__(self, *args):
            self.end = time.perf_counter()
            self.duration = self.end - self.start
            logger.success(f"Finished: {label}. Duration: {self.duration:.4f}s")

    return Timer()


def run_benchmark(variables: List[Variable]):
    name = f"{len(variables)} variables ({', '.join(v.name for v in variables)})"
    logger.info(f"=== Benchmarking: {name} ===")

    # # 1. Standard Generator
    # logger.info("[Standard BasisGenerator]")
    # try:
    #     with time_block("Standard Generation"):
    #         gen_std = BasisGenerator(variables)
    #         basis_std = gen_std.generate_basis()
    #     print(f"Standard Basis Size: {len(basis_std)}")
    # except Exception as e:
    #     logger.error(f"Standard Generation failed: {e}")

    # 2. Vectorized Generator
    logger.info("[VectorizedCanonicalBasis]")
    try:
        with time_block("Vectorized Generation"):
            gen_vec = VectorizedCanonicalBasis(variables)
        logger.info(f"Vectorized Basis Worlds: {gen_vec.n_worlds}")

        # Benchmark Evaluation
        if len(variables) > 1:
            target = variables[-1]
            root = variables[0]
            # Term: LastVar_{FirstVar=0}
            term = target @ {root: 0}
            logger.info(f"Evaluating term: {term}")

            with time_block("Vectorized Evaluation (All Worlds)"):
                res = gen_vec.evaluate(term)
                logger.info(f"Result shape: {res.shape}")

            # Sanity check
            logger.info(f"First 10 results: {res[:10]}")

    except Exception as e:
        logger.error(f"Vectorized Generation failed: {e}")


def main():
    # Case 1: 3 Binary Variables
    X = Variable("X", (0, 1))
    Y = Variable("Y", (0, 1))
    Z = Variable("Z", (0, 1))
    run_benchmark([X, Y, Z])

    # Case 2: 4 Binary Variables
    V3 = Variable("V3", (0, 1))
    run_benchmark([X, Y, Z, V3])

    # Case 3: 5 Binary Variables (Standard might be slow)
    # Basis size: 2 * 4 * 16 * 256 * 65536 = ~2 billion?
    # Wait.
    # V0: 2
    # V1: 2^2 = 4
    # V2: 2^4 = 16
    # V3: 2^8 = 256
    # V4: 2^16 = 65536.
    # Total: 2*4*16*256*65536 = 2,147,483,648.
    # That's too big for memory even for vectorized.
    # 2B bytes = 2GB.
    # But we store func tables.
    # Table for V4: (2B, 16) -> 32GB.
    # Berkley
    W = Variable("W", range(5))
    run_benchmark([X, Y, Z, W])

    # Let's try simpler domains or fewer vars.
    # 3 Ternary Variables?
    # X(3). Y(3^3=27). Z(3^9=19683).
    # Total: 3 * 27 * 19683 = 1,594,323.
    # This fits in memory easily.

    logger.info("--- Scaling Test (Ternary Domains) ---")
    vars_ternary = [
        Variable("T1", (0, 1, 2)),
        Variable("T2", (0, 1, 2)),
        Variable("T3", (0, 1, 2)),
    ]
    run_benchmark(vars_ternary)


if __name__ == "__main__":
    main()
