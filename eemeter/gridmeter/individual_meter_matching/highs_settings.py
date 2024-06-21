"""
Defines settings to be used for qpsolver's HiGHS optimization solver
"""

from __future__ import annotations

import numpy as np
import pydantic

from eemeter.gridmeter._utils.base_settings import BaseSettings

from typing import Optional, Literal


# system maximum float
MIN_FLOAT = np.finfo(np.float64).tiny
MAX_FLOAT = np.finfo(np.float64).max
    

class HiGHS_Settings(BaseSettings):
    """Settings for HiGHS optimization solver"""

    """Presolve option"""
    PRESOLVE: Literal["off", "choose", "on"] = pydantic.Field(
        default="choose",
        validate_default=True,
    )

    """If 'simplex'/'ipm'/'pdlp' is chosen then, for a MIP (QP) the integrality constraint (quadratic term) will be ignored"""
    # SOLVER: Literal["simplex", "choose", "ipm", "pdlp"] = pydantic.Field(
    #     default="choose",
    #     validate_default=True,
    # )

    """Parallel option"""
    PARALLEL: Literal["off", "choose", "on"] = pydantic.Field(
        default="off", # was "choose"
        validate_default=True,
    )

    """Run IPM crossover"""
    RUN_CROSSOVER: Literal["off", "choose", "on"] = pydantic.Field(
        default="on",
        validate_default=True,
    )

    """Time limit (seconds)"""
    TIME_LIMIT: float = pydantic.Field(
        default=float('inf'),
        ge=0,
        le=float('inf'),
        validate_default=True,
    )

    """Compute cost, bound, RHS and basic solution ranging"""
    RANGING: Literal["off", "on"] = pydantic.Field(
        default="off",
        validate_default=True,
    )

    """Limit on |cost coefficient|: values greater than or equal to this will be treated as infinite"""
    INFINITE_COST: float = pydantic.Field(
        default=1e+20,
        ge=1e+15,
        le=float('inf'),
        validate_default=True,
    )

    """Limit on |constraint bound|: values greater than or equal to this will be treated as infinite"""
    INFINITE_BOUND: float = pydantic.Field(
        default=1e+20,
        ge=1e+15,
        le=float('inf'),
        validate_default=True,
    )

    """Lower limit on |matrix entries|: values less than or equal to this will be treated as zero"""
    SMALL_MATRIX_VALUE: float = pydantic.Field(
        default=1e-09,
        ge=1e-12,
        le=float('inf'),
        validate_default=True,
    )

    """Upper limit on |matrix entries|: values greater than or equal to this will be treated as infinite"""
    LARGE_MATRIX_VALUE: float = pydantic.Field(
        default=1e+15,
        ge=1,
        le=float('inf'),
        validate_default=True,
    )

    """Primal feasibility tolerance"""
    PRIMAL_FEASIBILITY_TOLERANCE: float = pydantic.Field(
        default=1e-07,
        ge=1e-10,
        le=float('inf'),
        validate_default=True,
    )

    """Dual feasibility tolerance"""
    DUAL_FEASIBILITY_TOLERANCE: float = pydantic.Field(
        default=1e-07,
        ge=1e-10,
        le=float('inf'),
        validate_default=True,
    )

    """IPM optimality tolerance"""
    IPM_OPTIMALITY_TOLERANCE: float = pydantic.Field(
        default=1e-08,
        ge=1e-12,
        le=float('inf'),
        validate_default=True,
    )

    """Objective bound for termination of the dual simplex solver"""
    OBJECTIVE_BOUND: float = pydantic.Field(
        default=float('inf'),
        ge=float('-inf'),
        le=float('inf'),
        validate_default=True,
    )

    """Objective target for termination of the MIP solver"""
    OBJECTIVE_TARGET: float = pydantic.Field(
        default=float('-inf'),
        ge=float('-inf'),
        le=float('inf'),
        validate_default=True,
    )

    """Random seed used in HiGHS"""
    RANDOM_SEED: Optional[int] = pydantic.Field(
        default=None,
        ge=0,
        le=2147483647,
        validate_default=True,
    )

    """Number of threads used by HiGHS (0: automatic)"""
    THREADS: int = pydantic.Field(
        default=0,
        ge=0,
        le=2147483647,
        validate_default=True,
    )

    """Exponent of power-of-two bound scaling for model"""
    USER_BOUND_SCALE: int = pydantic.Field(
        default=0,
        ge=-2147483647,
        le=2147483647,
        validate_default=True,
    )

    """Exponent of power-of-two cost scaling for model"""
    USER_COST_SCALE: int = pydantic.Field(
        default=0,
        ge=-2147483647,
        le=2147483647,
        validate_default=True,
    )

    """Strategy for simplex solver [0: Choose; 1: Dual (serial); 2: Dual (PAMI); 3: Dual (SIP); 4: Primal]"""
    SIMPLEX_STRATEGY: int = pydantic.Field(
        default=1,
        ge=0,
        le=4,
        validate_default=True,
    )

    """Simplex scaling strategy: [0: off; 1: choose; 2: equilibration; 3: forced equilibration; 4: max value 0; 5: max value 1]"""
    SIMPLEX_SCALE_STRATEGY: int = pydantic.Field(
        default=1,
        ge=0,
        le=5,
        validate_default=True,
    )

    """Strategy for simplex dual edge weights: [-1: Choose; 0: Dantzig; 1: Devex; 2: Steepest Edge]"""
    SIMPLEX_DUAL_EDGE_WEIGHT_STRATEGY: int = pydantic.Field(
        default=-1,
        ge=-1,
        le=2,
        validate_default=True,
    )

    """Strategy for simplex primal edge weights: [-1: Choose; 0: Dantzig; 1: Devex; 2: Steepest Edge]"""
    SIMPLEX_PRIMAL_EDGE_WEIGHT_STRATEGY: int = pydantic.Field(
        default=-1,
        ge=-1,
        le=2,
        validate_default=True,
    )

    """Iteration limit for simplex solver when solving LPs, but not subproblems in the MIP solver"""
    SIMPLEX_ITERATION_LIMIT: int = pydantic.Field(
        default=2147483647,
        ge=0,
        le=2147483647,
        validate_default=True,
    )

    """Limit on the number of simplex UPDATE operations"""
    SIMPLEX_UPDATE_LIMIT: int = pydantic.Field(
        default=5000,
        ge=0,
        le=2147483647,
        validate_default=True,
    )

    """Maximum level of concurrency in parallel simplex"""
    SIMPLEX_MAX_CONCURRENCY: int = pydantic.Field(
        default=8,
        ge=1,
        le=8,
        validate_default=True,
    )

    """Enables or disables solver output"""
    # OUTPUT_FILE: bool = pydantic.Field(
    #     default=True,
    #     validate_default=True,
    # )

    """Enables or disables console logging"""
    # LOG_TO_CONSOLE: bool = pydantic.Field(
    #     default=True,
    #     validate_default=True,
    # )

    """Solution file"""
    SOLUTION_FILE: str = pydantic.Field(
        default="",
        validate_default=True,
    )

    """Log file"""
    LOG_FILE: str = pydantic.Field(
        default="",
        validate_default=True,
    )

    """Write the primal and dual solution to a file"""
    WRITE_SOLUTION_TO_FILE: bool = pydantic.Field(
        default=False,
        validate_default=True,
    )

    """Style of solution file: [0: HiGHS raw; 1: HiGHS pretty; 2: Glpsol raw; 3: Glpsol pretty; 4: HiGHS sparse raw] (raw = computer-readable, pretty = human-readable)"""
    WRITE_SOLUTION_STYLE: int = pydantic.Field(
        default=0,
        ge=0,
        le=4,
        validate_default=True,
    )

    """Location of cost row for Glpsol file: -2 => Last; -1 => None; 0 => None if empty, otherwise data file location; 1 <= n <= num_row => Location n; n > num_row => Last"""
    GLPSOL_COST_ROW_LOCATION: int = pydantic.Field(
        default=0,
        ge=-2,
        le=2147483647,
        validate_default=True,
    )

    """Write model file"""
    WRITE_MODEL_FILE: str = pydantic.Field(
        default="",
        validate_default=True,
    )

    """Write the model to a file"""
    WRITE_MODEL_TO_FILE: bool = pydantic.Field(
        default=False,
        validate_default=True,
    )

    """Whether MIP symmetry should be detected"""
    MIP_DETECT_SYMMETRY: bool = pydantic.Field(
        default=True,
        validate_default=True,
    )

    """Whether MIP restart is permitted"""
    MIP_ALLOW_RESTART: bool = pydantic.Field(
        default=True,
        validate_default=True,
    )

    """MIP solver max number of nodes"""
    MIP_MAX_NODES: int = pydantic.Field(
        default=2147483647,
        ge=0,
        le=2147483647,
        validate_default=True,
    )

    """MIP solver max number of nodes where estimate is above cutoff bound"""
    MIP_MAX_STALL_NODES: int = pydantic.Field(
        default=2147483647,
        ge=0,
        le=2147483647,
        validate_default=True,
    )

    """Whether improving MIP solutions should be saved"""
    MIP_IMPROVING_SOLUTION_SAVE: bool = pydantic.Field(
        default=False,
        validate_default=True,
    )

    """Whether improving MIP solutions should be reported in sparse format"""
    MIP_IMPROVING_SOLUTION_REPORT_SPARSE: bool = pydantic.Field(
        default=False,
        validate_default=True,
    )

    """File for reporting improving MIP solutions: not reported for an empty string ''"""
    MIP_IMPROVING_SOLUTION_FILE: str = pydantic.Field(
        default="",
        validate_default=True,
    )

    """MIP solver max number of leave nodes"""
    MIP_MAX_LEAVES: int = pydantic.Field(
        default=2147483647,
        ge=0,
        le=2147483647,
        validate_default=True,
    )

    """Limit on the number of improving solutions found to stop the MIP solver prematurely"""
    MIP_MAX_IMPROVING_SOLS: int = pydantic.Field(
        default=2147483647,
        ge=1,
        le=2147483647,
        validate_default=True,
    )

    """Maximal age of dynamic LP rows before they are removed from the LP relaxation in the MIP solver"""
    MIP_LP_AGE_LIMIT: int = pydantic.Field(
        default=10,
        ge=0,
        le=32767,
        validate_default=True,
    )

    """Maximal age of rows in the MIP solver cutpool before they are deleted"""
    MIP_POOL_AGE_LIMIT: int = pydantic.Field(
        default=30,
        ge=0,
        le=1000,
        validate_default=True,
    )

    """Soft limit on the number of rows in the MIP solver cutpool for dynamic age adjustment"""
    MIP_POOL_SOFT_LIMIT: int = pydantic.Field(
        default=10000,
        ge=1,
        le=2147483647,
        validate_default=True,
    )

    """Minimal number of observations before MIP solver pseudo costs are considered reliable"""
    MIP_PSCOST_MINRELIABLE: int = pydantic.Field(
        default=8,
        ge=0,
        le=2147483647,
        validate_default=True,
    )

    """Minimal number of entries in the MIP solver cliquetable before neighbourhood queries of the conflict graph use parallel processing"""
    MIP_MIN_CLIQUETABLE_ENTRIES_FOR_PARALLELISM: int = pydantic.Field(
        default=100000,
        ge=0,
        le=2147483647,
        validate_default=True,
    )

    """MIP feasibility tolerance"""
    MIP_FEASIBILITY_TOLERANCE: float = pydantic.Field(
        default=1e-06,
        ge=1e-10,
        le=float('inf'),
        validate_default=True,
    )

    """Effort spent for MIP heuristics"""
    MIP_HEURISTIC_EFFORT: float = pydantic.Field(
        default=0.05,
        ge=0,
        le=1,
        validate_default=True,
    )

    """Tolerance on relative gap, |ub-lb|/|ub|, to determine whether optimality has been reached for a MIP instance"""
    MIP_REL_GAP: float = pydantic.Field(
        default=0.0001,
        ge=0,
        le=float('inf'),
        validate_default=True,
    )

    """Tolerance on absolute gap of MIP, |ub-lb|, to determine whether optimality has been reached for a MIP instance"""
    MIP_ABS_GAP: float = pydantic.Field(
        default=1e-06,
        ge=0,
        le=float('inf'),
        validate_default=True,
    )

    """MIP minimum logging interval"""
    MIP_MIN_LOGGING_INTERVAL: float = pydantic.Field(
        default=5,
        ge=0,
        le=float('inf'),
        validate_default=True,
    )

    """Iteration limit for IPM solver"""
    IPM_ITERATION_LIMIT: int = pydantic.Field(
        default=2147483647,
        ge=0,
        le=2147483647,
        validate_default=True,
    )

    """Use native termination for PDLP solver: Default = false"""
    PDLP_NATIVE_TERMINATION: bool = pydantic.Field(
        default=False,
        validate_default=True,
    )

    """Scaling option for PDLP solver: Default = true"""
    PDLP_SCALING: bool = pydantic.Field(
        default=True,
        validate_default=True,
    )

    """Iteration limit for PDLP solver"""
    PDLP_ITERATION_LIMIT: int = pydantic.Field(
        default=2147483647,
        ge=0,
        le=2147483647,
        validate_default=True,
    )

    """Restart mode for PDLP solver: 0 => none; 1 => GPU (default); 2 => CPU"""
    PDLP_E_RESTART_METHOD: int = pydantic.Field(
        default=1,
        ge=0,
        le=2,
        validate_default=True,
    )

    """Duality gap tolerance for PDLP solver: Default = 1e-4"""
    PDLP_D_GAP_TOL: float = pydantic.Field(
        default=0.0001,
        ge=1e-12,
        le=float('inf'),
        validate_default=True,
    )


    """Make seed random if None"""
    @pydantic.model_validator(mode="after")
    def _random_seed(self):
        if self.RANDOM_SEED is None:
            try:
                min_int = self.model_fields["RANDOM_SEED"].metadata[0].ge
                max_int = self.model_fields["RANDOM_SEED"].metadata[1].le
            except:
                min_int = 0
                max_int = 2147483647

            self.RANDOM_SEED = np.random.randint(min_int, max_int)

        return self

if __name__ == "__main__":
    s = HiGHS_Settings()

    print(s.model_dump_json())
