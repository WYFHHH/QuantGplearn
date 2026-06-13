"""Alpha pool utilities inspired by alphagen, adapted to QuantGplearn programs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from .tensor_fitness import batch_pearsonr, clean_factor, normalize_by_day


class TensorFactorCalculator:
    """Calculate single-factor and mutual IC for QuantGplearn programs."""

    def __init__(self, data, normalize: bool = True, cache_factors: bool = True):
        self.data = data
        self.normalize = normalize
        self.cache_factors = cache_factors
        self.factor_cache: dict[str, torch.Tensor] = {}
        if data.target is not None:
            self.target = clean_factor(data.target, data.mask)
            if normalize:
                self.target = normalize_by_day(self.target, data.mask)
        else:
            self.target = None

    @torch.no_grad()
    def evaluate_program(self, program):
        key = program.generate_my_output()
        if self.cache_factors and key in self.factor_cache:
            return self.factor_cache[key]
        factor = clean_factor(program.execute_tensor(self.data), self.data.mask)
        if self.normalize:
            factor = normalize_by_day(factor, self.data.mask)
        if self.cache_factors:
            self.factor_cache[key] = factor.detach()
        return factor

    @torch.no_grad()
    def calc_single_ic(self, program) -> float:
        if self.target is None:
            raise ValueError("target is required to calculate single IC")
        factor = self.evaluate_program(program)
        ic = batch_pearsonr(factor, self.target, self.data.mask)
        out = torch.nanmean(ic)
        return 0.0 if not torch.isfinite(out) else float(out.item())

    @torch.no_grad()
    def calc_mutual_ic(self, p1, p2) -> float:
        f1 = self.evaluate_program(p1)
        f2 = self.evaluate_program(p2)
        ic = batch_pearsonr(f1, f2, self.data.mask)
        out = torch.nanmean(ic)
        return 0.0 if not torch.isfinite(out) else float(out.item())


@dataclass
class GPAlphaPool:
    """Small factor pool with IC threshold and mutual-correlation filtering."""

    capacity: int
    calculator: TensorFactorCalculator
    ic_lower_bound: float | None = None
    mutual_ic_upper_bound: float = 0.7
    programs: list = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    def try_add(self, program) -> bool:
        score = self.calculator.calc_single_ic(program)
        if self.ic_lower_bound is not None and score < self.ic_lower_bound:
            return False
        for old in self.programs:
            mutual = self.calculator.calc_mutual_ic(program, old)
            if abs(mutual) > self.mutual_ic_upper_bound:
                return False
        self.programs.append(program)
        self.scores.append(score)
        order = np.argsort(self.scores)[::-1]
        if len(order) > self.capacity:
            order = order[: self.capacity]
        self.programs = [self.programs[i] for i in order]
        self.scores = [float(self.scores[i]) for i in order]
        return True

    def update(self, programs: Sequence) -> "GPAlphaPool":
        for p in programs:
            self.try_add(p)
        return self

    def to_records(self) -> list[dict]:
        return [
            {"rank": i + 1, "score": score, "expression": program.generate_my_output()}
            for i, (program, score) in enumerate(zip(self.programs, self.scores))
        ]
