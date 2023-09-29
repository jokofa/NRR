#
from typing import Optional, Tuple
import numpy as np

np.warnings.filterwarnings('ignore', category=RuntimeWarning)   # ignore overflow RuntimeWarning in 1st iteration


class SimulatedAnnealing:
    """Simulated annealing strategy with different options
    for cooling schedules and acceptance criteria."""
    def __init__(self,
                 tau_init: float = 1.0,
                 tau_final: float = 0.0,
                 alpha: float = 0.8,
                 cooling_schedule: str = "lin",
                 acceptance_criterion: str = "metropolis",
                 num_max_steps: Optional[int] = None,
                 restart_at_step: int = 0,
                 seed: int = 1234,
                 **kwargs):
        super(SimulatedAnnealing, self).__init__()
        self.tau_init = tau_init
        self.tau_final = tau_final
        self.alpha = alpha
        self.cooling_schedule = cooling_schedule.lower()
        self.acceptance_criterion = acceptance_criterion.lower()
        self.num_max_steps = num_max_steps
        self.restart_at_step = restart_at_step
        self.seed = seed

        self.rng = np.random.default_rng(self.seed)
        self._step_restart_correction = None
        self._steps_no_imp = None
        self.reset()

    def reset(self):
        self._step_restart_correction = 0
        self._steps_no_imp = 0

    def _get_temp(self, step: int) -> float:
        """compute current temperature according to cooling schedule."""
        # "http://what-when-how.com/artificial-intelligence/
        # a-comparison-of-cooling-schedules-for-simulated-annealing-artificial-intelligence/"
        assert step >= 0
        if self.cooling_schedule == "lin":
            return self.tau_init / (1 + step)
        elif self.cooling_schedule == "exp_mult":
            return self.tau_init * self.alpha**step
        elif self.cooling_schedule == "log_mult":
            return self.tau_init / (1 + self.alpha * np.log(step + 1))
        elif self.cooling_schedule == "lin_mult":
            return self.tau_init / (1 + self.alpha * step)
        elif self.cooling_schedule == "quad_mult":
            return self.tau_init / (1 + self.alpha * (step ** 2))
        else:
            assert self.num_max_steps is not None and self.tau_final is not None
            if self.cooling_schedule == "lin_add":
                return (self.tau_final + (self.tau_init - self.tau_final) *
                        ((self.num_max_steps - step)/self.num_max_steps))
            elif self.cooling_schedule == "quad_add":
                return (self.tau_final + (self.tau_init - self.tau_final) *
                        ((self.num_max_steps - step)/self.num_max_steps)**2)
            elif self.cooling_schedule == "exp_add":
                return (
                    self.tau_final + (self.tau_init - self.tau_final) *
                    (1 / (1 + np.exp(
                        (2*np.log(self.tau_init - self.tau_final)/self.num_max_steps) *
                        (step - 0.5*self.num_max_steps))))
                )
            else:
                raise ValueError(f"unknown cooling_schedule: '{self.cooling_schedule}'.")

    def _accept(self, tau: float, e_old: float, e_new: float) -> bool:
        """Decide about acceptance."""
        delta = e_new - e_old
        if delta > 0:
            self._steps_no_imp += 1
        else:
            self._steps_no_imp = 0
        if self.acceptance_criterion == "metropolis":
            # stochastic accepting rule used in Kirkpatrick et al. 1983
            eps = np.exp(delta / tau)
            return (delta < 0) | (self.rng.random(1) > eps)
        elif self.acceptance_criterion == "threshold":
            # simple threshold accepting rule by Dueck & Scheuer 1990
            return delta < tau
        else:
            raise ValueError(f"unknown acceptance_criterion: '{self.acceptance_criterion}'.")

    def _restart(self, step: int):
        """
        <restart_at_step> represents the maximum allowable number of reductions in temperature
        if the value of the objective function has not improved.
        The algorithm restarts if restart_at_step is reached. When the algorithm restarts, the
        current temperature is reset to the initial temperature,
        and the creation of a new initial solution is requested
        to initiate a new SA run.
        The algorithm is terminated once it reaches the
        maximum number of iterations num_max_steps.
        """
        restart = False
        if self.restart_at_step > 0:
            first_step = (step == 0)
            if first_step:  # first iter
                self._step_restart_correction = 0
            restart = self._steps_no_imp >= self.restart_at_step
            if restart:
                self._step_restart_correction = step
            step = step-self._step_restart_correction

        return step, restart

    def check_accept(
            self,
            step: int,
            prev_cost: float,
            new_cost: float
    ) -> Tuple[bool, bool]:
        step, do_restart = self._restart(step)
        tau = self._get_temp(step)
        first_step = (step == 0)
        if first_step:     # first iter
            prev_cost = 1e12  # high cost value ^= inf

        return self._accept(tau, prev_cost, new_cost), do_restart
