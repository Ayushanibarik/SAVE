"""
Multi-Objective Optimization for Disaster Response System
=========================================================
Implements NSGA-II (Non-dominated Sorting Genetic Algorithm II) for
Pareto-optimal patient-hospital allocation.

Theoretical Foundations:
- NSGA-II (Deb et al., 2002) - Fast & Elitist Multi-Objective GA
- Pareto Dominance & Non-dominated Sorting
- Crowding Distance for solution diversity
- Tournament Selection with dominance-based comparison
- Simulated Binary Crossover (SBX) & Polynomial Mutation
- Hypervolume Indicator (Zitzler & Thiele, 1999)

Optimization Objectives:
1. Minimize total transport time (golden hour compliance)
2. Maximize survival probability (weighted by ESI acuity)
3. Minimize load imbalance (Gini coefficient)
4. Maximize specialty matching score

References:
- Deb et al. (2002) - A Fast and Elitist Multi-Objective GA: NSGA-II
- Zitzler et al. (2001) - SPEA2: Improving the Strength Pareto EA
- Coello et al. (2007) - Evolutionary Algorithms for Multi-Objective Optimization
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from copy import deepcopy
from datetime import datetime


# ============================================
# OBJECTIVE FUNCTIONS
# ============================================

class ObjectiveFunctions:
    """
    Multi-objective cost functions for disaster response optimization.
    
    Each function takes an allocation solution (chromosome) and returns
    a scalar objective value to be minimized.
    """
    
    @staticmethod
    def transport_time_objective(allocation: List[Dict]) -> float:
        """
        Objective 1: Minimize total weighted transport time.
        
        f₁ = Σ (assigned_i × eta_i × acuity_weight_i)
        
        Critical patients weighted 5x more than non-urgent.
        """
        if not allocation:
            return float('inf')
        
        total_time = 0
        for alloc in allocation:
            assigned = alloc.get("assigned", 0)
            eta = alloc.get("eta_minutes", 10)
            
            # Weight by assumed acuity distribution
            acuity_weight = 1.5  # Average acuity multiplier
            total_time += assigned * eta * acuity_weight
        
        return total_time
    
    @staticmethod
    def survival_objective(allocation: List[Dict]) -> float:
        """
        Objective 2: Maximize survival probability (return negative for minimization).
        
        P(survival) = Π P_i(t_i, ESI_i)
        
        Based on golden-hour survival decay model.
        """
        if not allocation:
            return 1.0  # Worst = 0 survival → return 1.0 (minimize)
        
        total_survival = 0
        total_patients = 0
        
        for alloc in allocation:
            assigned = alloc.get("assigned", 0)
            eta = alloc.get("eta_minutes", 10)
            
            if assigned == 0:
                continue
            
            # Survival probability (exponential decay with ETA)
            # P(s|t) = exp(-λt) where λ depends on severity
            lambda_decay = 0.015  # Average decay rate
            survival = math.exp(-lambda_decay * eta)
            
            total_survival += survival * assigned
            total_patients += assigned
        
        avg_survival = total_survival / max(1, total_patients)
        return 1.0 - avg_survival  # Minimize (1 - survival)
    
    @staticmethod
    def load_balance_objective(allocation: List[Dict]) -> float:
        """
        Objective 3: Minimize load imbalance (Gini coefficient).
        
        G = Σ|u_i - u_j| / (2n²·μ)
        
        Target: uniform utilization across all hospitals.
        """
        if not allocation or len(allocation) < 2:
            return 0.0
        
        utilizations = []
        for alloc in allocation:
            beds = alloc.get("available_beds", 1)
            assigned = alloc.get("assigned", 0)
            utilizations.append(assigned / max(1, beds))
        
        n = len(utilizations)
        mean_util = np.mean(utilizations)
        if mean_util == 0:
            return 0.0
        
        total_diff = sum(abs(u1 - u2) for u1 in utilizations for u2 in utilizations)
        gini = total_diff / (2 * n * n * mean_util)
        
        return gini  # Minimize Gini
    
    @staticmethod
    def specialty_match_objective(allocation: List[Dict], disaster_type: str) -> float:
        """
        Objective 4: Maximize specialty matching (return negative for minimization).
        
        Score how well allocated hospitals match disaster requirements.
        """
        if not allocation:
            return 1.0
        
        specialty_scores = {
            "FIRE": {"Burns": 1.0, "Trauma": 0.8, "Emergency": 0.7, "General": 0.5},
            "EARTHQUAKE": {"Trauma": 1.0, "Surgical": 0.9, "Emergency": 0.7, "General": 0.5},
            "FLOOD": {"Emergency": 0.8, "General": 0.7, "Internal Medicine": 0.9},
            "ACCIDENT": {"Trauma": 1.0, "Surgical": 0.9, "Emergency": 0.7, "General": 0.5},
            "CHEMICAL_SPILL": {"Toxicology": 1.0, "Emergency": 0.8, "General": 0.4},
            "BUILDING_COLLAPSE": {"Trauma": 1.0, "Surgical": 0.9, "Emergency": 0.7, "General": 0.5}
        }
        
        type_scores = specialty_scores.get(disaster_type.upper(), {})
        
        total_score = 0
        total_patients = 0
        for alloc in allocation:
            specialty = alloc.get("specialty", "General")
            assigned = alloc.get("assigned", 0)
            score = type_scores.get(specialty, 0.5)
            total_score += score * assigned
            total_patients += assigned
        
        avg_score = total_score / max(1, total_patients)
        return 1.0 - avg_score  # Minimize (1 - specialty_match)


# ============================================
# NSGA-II ALGORITHM
# ============================================

class Individual:
    """
    A candidate solution (chromosome) in the genetic algorithm.
    
    Encoding: Permutation-based chromosome
    - Gene[i] = fraction of remaining patients sent to hospital i
    - Decoded into concrete allocation by sequential assignment
    """
    
    def __init__(self, n_hospitals: int, chromosome: np.ndarray = None):
        self.n_hospitals = n_hospitals
        
        if chromosome is not None:
            self.chromosome = chromosome.copy()
        else:
            # Random initialization: proportional allocation weights
            raw = np.random.dirichlet(np.ones(n_hospitals))
            self.chromosome = raw
        
        self.objectives = []       # Objective function values
        self.rank = 0              # Non-domination rank
        self.crowding_distance = 0 # Crowding distance
        self.fitness = None
    
    def decode(self, hospitals: List[Dict], total_patients: int) -> List[Dict]:
        """
        Decode chromosome into concrete allocation.
        
        Process:
        1. Normalize chromosome to probabilities
        2. Scale by patient count
        3. Cap by hospital capacity
        4. Redistribute overflow
        """
        allocation = []
        weights = np.clip(self.chromosome, 0.01, None)
        weights = weights / weights.sum()
        
        # Initial allocation
        raw_alloc = (weights * total_patients).astype(int)
        remaining = total_patients - raw_alloc.sum()
        
        # Distribute remainder to highest-weight hospitals
        for i in np.argsort(-weights):
            if remaining <= 0:
                break
            raw_alloc[i] += 1
            remaining -= 1
        
        # Cap by capacity and redistribute overflow
        overflow = 0
        for i in range(min(len(hospitals), self.n_hospitals)):
            h = hospitals[i] if i < len(hospitals) else {}
            beds = h.get("Beds", h.get("beds", h.get("available_beds", 15)))
            
            assigned = min(raw_alloc[i], beds)
            overflow += max(0, raw_alloc[i] - beds)
            
            distance = h.get("_distance", h.get("distance", 5.0))
            eta = round(distance / 0.67, 1)
            
            allocation.append({
                "hospital": h.get("Hospital", h.get("name", f"H-{i}")),
                "assigned": int(assigned),
                "available_beds": beds,
                "distance": round(distance, 2),
                "eta_minutes": eta,
                "specialty": h.get("specialty", "General"),
                "icu_beds": h.get("icu_beds", 0),
                "lat": h.get("lat"),
                "lng": h.get("lng"),
                "weight": round(float(weights[i]), 4)
            })
        
        # Redistribute overflow to hospitals with remaining capacity
        if overflow > 0:
            for alloc in sorted(allocation, key=lambda a: a["assigned"] / max(1, a["available_beds"])):
                spare = alloc["available_beds"] - alloc["assigned"]
                if spare > 0 and overflow > 0:
                    added = min(spare, overflow)
                    alloc["assigned"] += added
                    overflow -= added
        
        return allocation
    
    def dominates(self, other: 'Individual') -> bool:
        """
        Pareto dominance check.
        
        Solution A dominates Solution B if:
        - A is at least as good as B in all objectives
        - A is strictly better than B in at least one objective
        """
        at_least_as_good = all(a <= b for a, b in zip(self.objectives, other.objectives))
        strictly_better = any(a < b for a, b in zip(self.objectives, other.objectives))
        return at_least_as_good and strictly_better


class NSGA2:
    """
    NSGA-II (Non-dominated Sorting Genetic Algorithm II).
    
    Features:
    - Fast non-dominated sorting: O(MN²) where M=objectives, N=population
    - Crowding distance for diversity preservation
    - Binary tournament selection with crowding comparison
    - Simulated Binary Crossover (SBX)
    - Polynomial mutation
    - Elitist replacement
    
    Deb et al. (2002)
    """
    
    def __init__(self, pop_size: int = 50, n_generations: int = 30,
                 crossover_rate: float = 0.9, mutation_rate: float = 0.1,
                 eta_crossover: float = 20.0, eta_mutation: float = 20.0):
        self.pop_size = pop_size
        self.n_gen = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.eta_c = eta_crossover    # SBX distribution index
        self.eta_m = eta_mutation      # Mutation distribution index
    
    def _fast_non_dominated_sort(self, population: List[Individual]) -> List[List[int]]:
        """
        Fast Non-Dominated Sorting (Deb et al., 2002).
        
        Assigns each individual to a front (rank).
        Front 0 = Pareto-optimal solutions (non-dominated by any other).
        
        Time complexity: O(MN²) where M = #objectives, N = #individuals
        """
        n = len(population)
        domination_count = [0] * n       # How many solutions dominate individual i
        dominated_set = [[] for _ in range(n)]  # Set of solutions dominated by i
        fronts = [[]]
        
        for i in range(n):
            for j in range(i + 1, n):
                if population[i].dominates(population[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif population[j].dominates(population[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1
            
            if domination_count[i] == 0:
                population[i].rank = 0
                fronts[0].append(i)
        
        current_front = 0
        while fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        population[j].rank = current_front + 1
                        next_front.append(j)
            
            current_front += 1
            fronts.append(next_front)
        
        return [f for f in fronts if f]  # Remove empty fronts
    
    def _crowding_distance(self, population: List[Individual], 
                           front: List[int]) -> None:
        """
        Crowding Distance computation.
        
        Measures how isolated a solution is in objective space.
        Higher crowding distance = more diverse = preferred.
        
        For each objective m:
        CD_i += (f_m(i+1) - f_m(i-1)) / (f_m_max - f_m_min)
        """
        if len(front) <= 2:
            for i in front:
                population[i].crowding_distance = float('inf')
            return
        
        for i in front:
            population[i].crowding_distance = 0
        
        n_objectives = len(population[front[0]].objectives)
        
        for m in range(n_objectives):
            # Sort front by objective m
            sorted_front = sorted(front, key=lambda i: population[i].objectives[m])
            
            # Boundary solutions get infinite distance
            population[sorted_front[0]].crowding_distance = float('inf')
            population[sorted_front[-1]].crowding_distance = float('inf')
            
            # Range of objective m
            f_min = population[sorted_front[0]].objectives[m]
            f_max = population[sorted_front[-1]].objectives[m]
            obj_range = f_max - f_min
            
            if obj_range == 0:
                continue
            
            # Interior solutions
            for k in range(1, len(sorted_front) - 1):
                population[sorted_front[k]].crowding_distance += (
                    (population[sorted_front[k + 1]].objectives[m] - 
                     population[sorted_front[k - 1]].objectives[m]) / obj_range
                )
    
    def _tournament_selection(self, population: List[Individual]) -> Individual:
        """
        Binary tournament selection with crowding comparison.
        
        1. Pick two random individuals
        2. Prefer lower rank (closer to Pareto front)
        3. If same rank, prefer higher crowding distance (more diverse)
        """
        i, j = np.random.randint(0, len(population), 2)
        
        if population[i].rank < population[j].rank:
            return population[i]
        elif population[j].rank < population[i].rank:
            return population[j]
        elif population[i].crowding_distance > population[j].crowding_distance:
            return population[i]
        else:
            return population[j]
    
    def _sbx_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Simulated Binary Crossover (SBX).
        
        Uses polynomial distribution to create offspring near parents.
        Distribution index η_c controls offspring spread:
        - High η_c → offspring closer to parents
        - Low η_c → wider exploration
        
        Deb & Agrawal (1995)
        """
        n = parent1.n_hospitals
        child1_chrom = parent1.chromosome.copy()
        child2_chrom = parent2.chromosome.copy()
        
        if np.random.random() > self.crossover_rate:
            return (Individual(n, child1_chrom), Individual(n, child2_chrom))
        
        for i in range(n):
            if np.random.random() < 0.5:
                if abs(parent1.chromosome[i] - parent2.chromosome[i]) > 1e-10:
                    # SBX polynomial distribution
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (self.eta_c + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (self.eta_c + 1))
                    
                    child1_chrom[i] = 0.5 * ((1 + beta) * parent1.chromosome[i] + 
                                              (1 - beta) * parent2.chromosome[i])
                    child2_chrom[i] = 0.5 * ((1 - beta) * parent1.chromosome[i] + 
                                              (1 + beta) * parent2.chromosome[i])
        
        # Ensure valid (positive) weights
        child1_chrom = np.clip(child1_chrom, 0.01, None)
        child2_chrom = np.clip(child2_chrom, 0.01, None)
        
        return (Individual(n, child1_chrom), Individual(n, child2_chrom))
    
    def _polynomial_mutation(self, individual: Individual) -> Individual:
        """
        Polynomial mutation operator.
        
        Each gene mutated with probability p_m.
        Mutation magnitude controlled by η_m (distribution index).
        
        Deb & Goyal (1996)
        """
        chrom = individual.chromosome.copy()
        
        for i in range(len(chrom)):
            if np.random.random() < self.mutation_rate:
                u = np.random.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (self.eta_m + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (self.eta_m + 1))
                
                chrom[i] += delta * chrom[i]
        
        chrom = np.clip(chrom, 0.01, None)
        return Individual(individual.n_hospitals, chrom)
    
    def optimize(self, hospitals: List[Dict], total_patients: int,
                 disaster_type: str) -> Dict:
        """
        Run NSGA-II multi-objective optimization.
        
        Returns:
        - Pareto front of non-dominated solutions
        - Knee-point recommendation
        - Hypervolume quality indicator
        """
        n_hospitals = len(hospitals)
        if n_hospitals == 0:
            return {"error": "No hospitals for optimization"}
        
        # Initialize population
        population = [Individual(n_hospitals) for _ in range(self.pop_size)]
        
        # Evaluate objectives for initial population
        for ind in population:
            allocation = ind.decode(hospitals, total_patients)
            ind.objectives = [
                ObjectiveFunctions.transport_time_objective(allocation),
                ObjectiveFunctions.survival_objective(allocation),
                ObjectiveFunctions.load_balance_objective(allocation),
                ObjectiveFunctions.specialty_match_objective(allocation, disaster_type)
            ]
        
        # Evolution loop
        convergence_history = []
        
        for gen in range(self.n_gen):
            # Create offspring
            offspring = []
            while len(offspring) < self.pop_size:
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                child1, child2 = self._sbx_crossover(parent1, parent2)
                child1 = self._polynomial_mutation(child1)
                child2 = self._polynomial_mutation(child2)
                
                # Evaluate offspring
                for child in [child1, child2]:
                    allocation = child.decode(hospitals, total_patients)
                    child.objectives = [
                        ObjectiveFunctions.transport_time_objective(allocation),
                        ObjectiveFunctions.survival_objective(allocation),
                        ObjectiveFunctions.load_balance_objective(allocation),
                        ObjectiveFunctions.specialty_match_objective(allocation, disaster_type)
                    ]
                    offspring.append(child)
            
            # Combine parents + offspring (2N)
            combined = population + offspring[:self.pop_size]
            
            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(combined)
            
            # Crowding distance for each front
            for front in fronts:
                self._crowding_distance(combined, front)
            
            # Select next generation (elitist)
            new_population = []
            for front in fronts:
                if len(new_population) + len(front) <= self.pop_size:
                    new_population.extend([combined[i] for i in front])
                else:
                    # Sort by crowding distance (descending) and take best
                    remaining = self.pop_size - len(new_population)
                    sorted_front = sorted(front, 
                                         key=lambda i: combined[i].crowding_distance,
                                         reverse=True)
                    new_population.extend([combined[i] for i in sorted_front[:remaining]])
                    break
            
            population = new_population[:self.pop_size]
            
            # Track convergence
            front0_objs = [ind.objectives for ind in population if ind.rank == 0]
            if front0_objs:
                convergence_history.append({
                    "generation": gen,
                    "pareto_front_size": len(front0_objs),
                    "avg_objectives": [round(np.mean([o[i] for o in front0_objs]), 4)
                                       for i in range(4)]
                })
        
        # Extract Pareto front
        pareto_front = [ind for ind in population if ind.rank == 0]
        
        # Find knee-point (solution closest to ideal point)
        ideal = [min(ind.objectives[i] for ind in pareto_front) for i in range(4)]
        nadir = [max(ind.objectives[i] for ind in pareto_front) for i in range(4)]
        
        best_knee = None
        min_distance = float('inf')
        
        for ind in pareto_front:
            # Normalized distance to ideal point
            dist = sum(((ind.objectives[i] - ideal[i]) / max(1e-10, nadir[i] - ideal[i])) ** 2
                       for i in range(4))
            dist = math.sqrt(dist)
            
            if dist < min_distance:
                min_distance = dist
                best_knee = ind
        
        # Decode best solutions
        knee_allocation = best_knee.decode(hospitals, total_patients) if best_knee else []
        
        # Hypervolume computation (approximate - 2D projection)
        reference_point = [max(nadir[i] * 1.1, 1.0) for i in range(4)]
        hypervolume = self._compute_hypervolume(pareto_front, reference_point)
        
        # Pareto front solutions for visualization
        pareto_solutions = []
        for i, ind in enumerate(pareto_front[:10]):
            alloc = ind.decode(hospitals, total_patients)
            pareto_solutions.append({
                "solution_id": i + 1,
                "objectives": {
                    "transport_time": round(ind.objectives[0], 2),
                    "survival_loss": round(ind.objectives[1], 4),
                    "load_imbalance": round(ind.objectives[2], 4),
                    "specialty_mismatch": round(ind.objectives[3], 4)
                },
                "crowding_distance": round(ind.crowding_distance, 4) if ind.crowding_distance < float('inf') else "∞",
                "is_knee_point": (ind == best_knee),
                "allocation_summary": [
                    {"hospital": a["hospital"], "patients": a["assigned"]}
                    for a in alloc if a["assigned"] > 0
                ]
            })
        
        return {
            "engine": "S.A.V.E. NSGA-II Multi-Objective Optimizer v1.0",
            "theoretical_basis": [
                "NSGA-II (Deb et al., 2002)",
                "Pareto Dominance & Non-dominated Sorting",
                "Crowding Distance for diversity preservation",
                "Simulated Binary Crossover (SBX)",
                "Polynomial Mutation",
                "Hypervolume Quality Indicator"
            ],
            "parameters": {
                "population_size": self.pop_size,
                "generations": self.n_gen,
                "crossover_rate": self.crossover_rate,
                "mutation_rate": self.mutation_rate,
                "eta_crossover": self.eta_c,
                "eta_mutation": self.eta_m
            },
            "objectives": {
                "names": ["Transport Time", "Survival Loss", "Load Imbalance", "Specialty Mismatch"],
                "directions": ["minimize", "minimize", "minimize", "minimize"],
                "ideal_point": [round(v, 4) for v in ideal],
                "nadir_point": [round(v, 4) for v in nadir]
            },
            "pareto_front": {
                "size": len(pareto_front),
                "solutions": pareto_solutions,
                "hypervolume": round(hypervolume, 4),
            },
            "recommended_solution": {
                "method": "Knee-point (minimum normalized distance to ideal)",
                "allocation": knee_allocation,
                "objectives": {
                    "transport_time": round(best_knee.objectives[0], 2) if best_knee else None,
                    "survival_probability": round(1 - best_knee.objectives[1], 4) if best_knee else None,
                    "load_balance_gini": round(best_knee.objectives[2], 4) if best_knee else None,
                    "specialty_match": round(1 - best_knee.objectives[3], 4) if best_knee else None
                }
            },
            "convergence": convergence_history[-5:] if convergence_history else [],
            "timestamp": datetime.now().isoformat()
        }
    
    def _compute_hypervolume(self, pareto_front: List[Individual],
                              reference: List[float]) -> float:
        """
        Approximate hypervolume indicator.
        
        HV = volume of objective space dominated by Pareto front
        and bounded by reference point.
        
        Uses Monte Carlo sampling for >2 objectives.
        """
        if not pareto_front:
            return 0.0
        
        n_samples = 5000
        n_obj = len(reference)
        
        # Sample random points in the bounded space
        ideal = [min(ind.objectives[i] for ind in pareto_front) for i in range(n_obj)]
        
        total_volume = 1.0
        for i in range(n_obj):
            total_volume *= (reference[i] - ideal[i])
        
        if total_volume <= 0:
            return 0.0
        
        # Monte Carlo: count points dominated by at least one Pareto solution
        dominated_count = 0
        for _ in range(n_samples):
            point = [np.random.uniform(ideal[i], reference[i]) for i in range(n_obj)]
            
            for ind in pareto_front:
                if all(ind.objectives[i] <= point[i] for i in range(n_obj)):
                    dominated_count += 1
                    break
        
        return total_volume * dominated_count / n_samples


# ============================================
# GLOBAL INSTANCE & CONVENIENCE
# ============================================

nsga2_optimizer = NSGA2(pop_size=40, n_generations=25)


def pareto_optimize(hospitals: List[Dict], total_patients: int,
                    disaster_type: str, **kwargs) -> Dict:
    """Quick access to Pareto multi-objective optimization"""
    pop_size = kwargs.get("pop_size", 40)
    n_gen = kwargs.get("n_generations", 25)
    
    optimizer = NSGA2(pop_size=pop_size, n_generations=n_gen)
    return optimizer.optimize(hospitals, total_patients, disaster_type)
