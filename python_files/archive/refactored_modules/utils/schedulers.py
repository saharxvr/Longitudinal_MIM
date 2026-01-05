"""
Learning Rate and Mask Probability Schedulers
==============================================

Custom schedulers for training that aren't provided by PyTorch.

Usage:
------
    from utils.schedulers import MaskProbScheduler
    
    scheduler = MaskProbScheduler(epochs=30, steps_per_epoch=100)
    for step in range(total_steps):
        mask_prob = scheduler.step()
"""

from typing import Optional


class MaskProbScheduler:
    """
    Scheduler for mask probability in Masked Image Modeling.
    
    Implements a curriculum learning schedule where masking probability
    starts low, ramps up to a maximum, holds, then optionally decreases.
    
    Schedule phases:
    1. Warmup: Hold at init_val
    2. Ramp up: Linear increase to max_val
    3. Plateau: Hold at max_val
    4. Ramp down: Linear decrease to end_val
    5. Final: Hold at end_val
    
    Parameters
    ----------
    epochs : int
        Total training epochs.
    steps_per_epoch : int
        Number of steps (batches) per epoch.
    init_val : float, default=0.05
        Initial mask probability.
    max_val : float, default=0.7
        Maximum mask probability.
    end_val : float, default=0.7
        Final mask probability.
    perc_on_start : float, default=0.05
        Fraction of training on initial value (warmup).
    perc_on_slope : float, default=0.2
        Fraction of training for ramp up.
    perc_on_max : float, default=0.4
        Fraction of training at maximum.
    perc_on_slope2 : float, default=0.1
        Fraction of training for ramp down.
        
    Examples
    --------
    >>> scheduler = MaskProbScheduler(epochs=30, steps_per_epoch=100)
    >>> for epoch in range(30):
    ...     for step in range(100):
    ...         mask_prob = scheduler.step()
    ...         # Use mask_prob in training
    """
    
    def __init__(
        self,
        epochs: int,
        steps_per_epoch: int,
        init_val: float = 0.05,
        max_val: float = 0.7,
        end_val: float = 0.7,
        perc_on_start: float = 0.05,
        perc_on_slope: float = 0.2,
        perc_on_max: float = 0.4,
        perc_on_slope2: float = 0.1
    ):
        total_steps = epochs * steps_per_epoch
        
        self.init_val = init_val
        self.max_val = max_val
        self.end_val = end_val
        
        # Calculate phase thresholds
        self.th1 = perc_on_start * total_steps
        self.th2 = self.th1 + perc_on_slope * total_steps
        self.th3 = self.th2 + perc_on_max * total_steps
        self.th4 = self.th3 + perc_on_slope2 * total_steps
        
        # Calculate slopes for linear phases
        self.slope1 = (max_val - init_val) / (self.th2 - self.th1) if self.th2 > self.th1 else 0
        self.slope2 = (end_val - max_val) / (self.th4 - self.th3) if self.th4 > self.th3 else 0
        
        self.cur_step = 0
    
    def get_step(self) -> int:
        """Get current step number."""
        return self.cur_step
    
    def set_step(self, step: int) -> None:
        """Set current step number (for resuming training)."""
        self.cur_step = step
    
    def calc_cur_val(self) -> float:
        """Calculate mask probability for current step."""
        if self.cur_step <= self.th1:
            # Warmup phase
            val = self.init_val
        elif self.th1 < self.cur_step <= self.th2:
            # Ramp up phase
            val = (self.cur_step - self.th1) * self.slope1 + self.init_val
        elif self.th2 < self.cur_step <= self.th3:
            # Plateau phase
            val = self.max_val
        elif self.th3 < self.cur_step <= self.th4:
            # Ramp down phase
            val = (self.cur_step - self.th3) * self.slope2 + self.max_val
        else:
            # Final phase
            val = self.end_val
        return val
    
    def step(self) -> float:
        """
        Get current mask probability and advance to next step.
        
        Returns
        -------
        float
            Mask probability for current step.
        """
        val = self.calc_cur_val()
        self.cur_step += 1
        return val
    
    def get_current_value(self) -> float:
        """Get mask probability without advancing step."""
        return self.calc_cur_val()


class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup and cosine decay.
    
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer to schedule.
    warmup_steps : int
        Number of warmup steps.
    total_steps : int
        Total training steps.
    min_lr : float, default=1e-6
        Minimum learning rate at end of decay.
    """
    
    def __init__(
        self,
        warmup_steps: int,
        total_steps: int,
        max_lr: float,
        min_lr: float = 1e-6
    ):
        import math
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.cur_step = 0
        self._math = math
    
    def get_lr(self) -> float:
        """Calculate learning rate for current step."""
        if self.cur_step < self.warmup_steps:
            # Linear warmup
            return self.max_lr * (self.cur_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.cur_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + self._math.cos(self._math.pi * progress))
    
    def step(self) -> float:
        """Get current LR and advance step."""
        lr = self.get_lr()
        self.cur_step += 1
        return lr
