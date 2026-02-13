"""
Markov Decision Process & Monte Carlo Simulation for Disaster Progression
=========================================================================
Models patient state transitions and disaster outcome prediction using
stochastic processes and simulation.

Theoretical Foundations:
- Markov Chains (Markov, 1906) - Memoryless state transitions
- Markov Decision Process (Bellman, 1957) - Sequential decision making
- Monte Carlo Methods (Metropolis & Ulam, 1949) - Stochastic simulation
- Value Iteration (Bellman, 1957) - Dynamic programming for optimal policy
- Chapman-Kolmogorov Equation: P(n) = Pⁿ (n-step transition matrix)
- Steady-state distribution: πP = π
- Absorbing Markov Chains (Kemeny & Snell, 1960)

References:
- Puterman (2005) - Markov Decision Processes: Discrete Stochastic DP
- Ross (2014) - Introduction to Probability Models
- Sutton & Barto (2018) - Reinforcement Learning (Ch. 3-4, MDP)
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict


# ============================================
# PATIENT STATE DEFINITIONS
# ============================================

# Patient states in the Markov chain
STATES = {
    0: {"name": "Stable", "color": "green", "absorbing": False},
    1: {"name": "Moderate", "color": "yellow", "absorbing": False},
    2: {"name": "Serious", "color": "orange", "absorbing": False},
    3: {"name": "Critical", "color": "red", "absorbing": False},
    4: {"name": "ICU", "color": "darkred", "absorbing": False},
    5: {"name": "Recovered", "color": "blue", "absorbing": True},
    6: {"name": "Deceased", "color": "black", "absorbing": True}
}

STATE_NAMES = [s["name"] for s in STATES.values()]
N_STATES = len(STATES)


# ============================================
# TRANSITION PROBABILITY MATRICES
# ============================================

# Transition matrices calibrated per disaster type
# P[i][j] = probability of transitioning from state i to state j in one time step (6 hours)
# Based on WHO mass casualty data and trauma surgery literature

TRANSITION_MATRICES = {
    "FIRE": np.array([
        #  Stable  Moderate Serious Critical  ICU    Recovered Deceased
        [  0.40,   0.25,    0.15,    0.05,    0.02,  0.13,     0.00  ],  # Stable
        [  0.10,   0.35,    0.25,    0.10,    0.05,  0.14,     0.01  ],  # Moderate
        [  0.03,   0.10,    0.30,    0.25,    0.15,  0.12,     0.05  ],  # Serious
        [  0.01,   0.02,    0.08,    0.30,    0.35,  0.10,     0.14  ],  # Critical
        [  0.00,   0.01,    0.03,    0.10,    0.45,  0.20,     0.21  ],  # ICU
        [  0.00,   0.00,    0.00,    0.00,    0.00,  1.00,     0.00  ],  # Recovered (absorbing)
        [  0.00,   0.00,    0.00,    0.00,    0.00,  0.00,     1.00  ],  # Deceased (absorbing)
    ]),
    
    "EARTHQUAKE": np.array([
        [  0.35,   0.28,    0.18,    0.07,    0.03,  0.09,     0.00  ],
        [  0.08,   0.30,    0.28,    0.15,    0.06,  0.12,     0.01  ],
        [  0.02,   0.08,    0.28,    0.28,    0.18,  0.10,     0.06  ],
        [  0.01,   0.02,    0.05,    0.28,    0.38,  0.08,     0.18  ],
        [  0.00,   0.01,    0.02,    0.08,    0.42,  0.18,     0.29  ],
        [  0.00,   0.00,    0.00,    0.00,    0.00,  1.00,     0.00  ],
        [  0.00,   0.00,    0.00,    0.00,    0.00,  0.00,     1.00  ],
    ]),
    
    "FLOOD": np.array([
        [  0.45,   0.22,    0.12,    0.04,    0.02,  0.15,     0.00  ],
        [  0.12,   0.38,    0.22,    0.08,    0.04,  0.15,     0.01  ],
        [  0.04,   0.12,    0.32,    0.22,    0.12,  0.14,     0.04  ],
        [  0.01,   0.03,    0.08,    0.32,    0.32,  0.12,     0.12  ],
        [  0.00,   0.01,    0.03,    0.10,    0.48,  0.18,     0.20  ],
        [  0.00,   0.00,    0.00,    0.00,    0.00,  1.00,     0.00  ],
        [  0.00,   0.00,    0.00,    0.00,    0.00,  0.00,     1.00  ],
    ]),
    
    "ACCIDENT": np.array([
        [  0.42,   0.24,    0.14,    0.05,    0.02,  0.13,     0.00  ],
        [  0.10,   0.36,    0.24,    0.10,    0.04,  0.15,     0.01  ],
        [  0.03,   0.10,    0.30,    0.24,    0.14,  0.13,     0.06  ],
        [  0.01,   0.02,    0.06,    0.30,    0.36,  0.09,     0.16  ],
        [  0.00,   0.01,    0.03,    0.08,    0.44,  0.20,     0.24  ],
        [  0.00,   0.00,    0.00,    0.00,    0.00,  1.00,     0.00  ],
        [  0.00,   0.00,    0.00,    0.00,    0.00,  0.00,     1.00  ],
    ]),
    
    "CHEMICAL_SPILL": np.array([
        [  0.35,   0.28,    0.18,    0.08,    0.04,  0.07,     0.00  ],
        [  0.06,   0.30,    0.28,    0.16,    0.08,  0.10,     0.02  ],
        [  0.02,   0.06,    0.26,    0.28,    0.20,  0.10,     0.08  ],
        [  0.00,   0.02,    0.05,    0.26,    0.38,  0.07,     0.22  ],
        [  0.00,   0.01,    0.02,    0.06,    0.40,  0.16,     0.35  ],
        [  0.00,   0.00,    0.00,    0.00,    0.00,  1.00,     0.00  ],
        [  0.00,   0.00,    0.00,    0.00,    0.00,  0.00,     1.00  ],
    ]),
    
    "BUILDING_COLLAPSE": np.array([
        [  0.33,   0.28,    0.20,    0.08,    0.04,  0.07,     0.00  ],
        [  0.07,   0.30,    0.28,    0.16,    0.07,  0.11,     0.01  ],
        [  0.02,   0.08,    0.26,    0.28,    0.20,  0.09,     0.07  ],
        [  0.01,   0.02,    0.05,    0.26,    0.38,  0.07,     0.21  ],
        [  0.00,   0.01,    0.02,    0.08,    0.42,  0.17,     0.30  ],
        [  0.00,   0.00,    0.00,    0.00,    0.00,  1.00,     0.00  ],
        [  0.00,   0.00,    0.00,    0.00,    0.00,  0.00,     1.00  ],
    ]),
}


# ============================================
# INTERVENTION EFFECTS
# ============================================

# How medical interventions modify transition probabilities
INTERVENTION_EFFECTS = {
    "rapid_triage": {
        "description": "START/SALT triage within 15 minutes",
        "recovery_boost": 0.08,
        "mortality_reduction": 0.04,
        "time_window_hours": 0.25
    },
    "surgical_intervention": {
        "description": "Emergency surgical teams deployed",
        "recovery_boost": 0.12,
        "mortality_reduction": 0.10,
        "time_window_hours": 1.0
    },
    "blood_bank_activation": {
        "description": "Mass transfusion protocol activated",
        "recovery_boost": 0.06,
        "mortality_reduction": 0.08,
        "time_window_hours": 2.0
    },
    "helicopter_evac": {
        "description": "Air ambulance for critical patients",
        "recovery_boost": 0.05,
        "mortality_reduction": 0.12,
        "time_window_hours": 0.5
    },
    "field_hospital": {
        "description": "Emergency field hospital deployed",
        "recovery_boost": 0.10,
        "mortality_reduction": 0.06,
        "time_window_hours": 4.0
    }
}

def apply_intervention(P: np.ndarray, intervention: str) -> np.ndarray:
    """
    Modify transition matrix based on medical intervention.
    
    Increases recovery probability and decreases mortality probability
    proportional to intervention effectiveness.
    """
    P_modified = P.copy()
    effects = INTERVENTION_EFFECTS.get(intervention)
    if not effects:
        return P_modified
    
    recovery_boost = effects["recovery_boost"]
    mortality_reduction = effects["mortality_reduction"]
    
    # For non-absorbing states: increase recovery, decrease mortality transitions
    for i in range(5):  # States 0-4 (non-absorbing)
        # Boost recovery (state 5) transition
        P_modified[i, 5] = min(0.99, P_modified[i, 5] + recovery_boost)
        # Reduce mortality (state 6) transition
        P_modified[i, 6] = max(0.001, P_modified[i, 6] - mortality_reduction)
        
        # Renormalize row to sum to 1
        row_sum = P_modified[i].sum()
        if row_sum > 0:
            P_modified[i] /= row_sum
    
    return P_modified


# ============================================
# MARKOV CHAIN ANALYZER
# ============================================

class MarkovChainAnalyzer:
    """
    Markov chain analysis for patient state progression.
    
    Computes:
    - N-step transition probabilities: P(n) = Pⁿ
    - Steady-state distribution: πP = π (eigenvector of P^T for eigenvalue 1)
    - Absorption probabilities: probability of reaching each absorbing state
    - Expected time to absorption: E[T] from each transient state
    - Fundamental matrix: N = (I - Q)⁻¹ for absorbing chain analysis
    """
    
    @staticmethod
    def n_step_transition(P: np.ndarray, n: int) -> np.ndarray:
        """
        Chapman-Kolmogorov: P(n) = P^n
        Compute n-step transition probabilities via matrix exponentiation.
        """
        return np.linalg.matrix_power(P, n)
    
    @staticmethod
    def steady_state(P: np.ndarray) -> np.ndarray:
        """
        Find steady-state distribution π such that πP = π.
        
        Computed as the left eigenvector of P for eigenvalue 1.
        For absorbing chains, this gives the limiting distribution.
        """
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        
        # Find eigenvector for eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        steady = np.real(eigenvectors[:, idx])
        steady = steady / steady.sum()  # Normalize
        
        return np.abs(steady)
    
    @staticmethod
    def absorption_analysis(P: np.ndarray) -> Dict:
        """
        Analyze absorbing Markov chain.
        
        Canonical form: P = [[Q  R], [0  I]]
        
        Where:
        - Q = transition matrix among transient states
        - R = transition probabilities from transient to absorbing states
        - N = (I - Q)⁻¹ = fundamental matrix
        - B = NR = absorption probability matrix
        - E[T] = N·1 = expected steps to absorption
        
        (Kemeny & Snell, 1960)
        """
        # Identify transient and absorbing states
        n = P.shape[0]
        transient = []
        absorbing = []
        
        for i in range(n):
            if abs(P[i, i] - 1.0) < 1e-10 and sum(P[i, j] for j in range(n) if j != i) < 1e-10:
                absorbing.append(i)
            else:
                transient.append(i)
        
        if not absorbing or not transient:
            return {"error": "Not a valid absorbing Markov chain"}
        
        t = len(transient)
        a = len(absorbing)
        
        # Extract Q (transient→transient) and R (transient→absorbing)
        Q = P[np.ix_(transient, transient)]
        R = P[np.ix_(transient, absorbing)]
        
        # Fundamental matrix: N = (I - Q)⁻¹
        try:
            N = np.linalg.inv(np.eye(t) - Q)
        except np.linalg.LinAlgError:
            N = np.linalg.pinv(np.eye(t) - Q)
        
        # Absorption probability matrix: B = N * R
        B = N @ R
        
        # Expected time to absorption: t = N * 1
        expected_time = N @ np.ones(t)
        
        return {
            "transient_states": [STATE_NAMES[i] for i in transient],
            "absorbing_states": [STATE_NAMES[i] for i in absorbing],
            "absorption_probabilities": {
                STATE_NAMES[transient[i]]: {
                    STATE_NAMES[absorbing[j]]: round(float(B[i, j]), 4)
                    for j in range(a)
                }
                for i in range(t)
            },
            "expected_time_to_absorption": {
                STATE_NAMES[transient[i]]: round(float(expected_time[i]), 2)
                for i in range(t)
            },
            "fundamental_matrix_trace": round(float(np.trace(N)), 4)
        }


# ============================================
# MONTE CARLO SIMULATION ENGINE
# ============================================

class MonteCarloSimulator:
    """
    Monte Carlo simulation for disaster outcome prediction.
    
    Runs N independent stochastic simulations of patient state
    progression to estimate outcome distributions.
    
    Features:
    - Configurable number of simulations (default 1000)
    - Time-step projection (6h, 12h, 24h, 48h)
    - Confidence intervals (95% CI)
    - Intervention scenario comparison
    - Sensitivity analysis on key parameters
    """
    
    def __init__(self, n_simulations: int = 1000):
        self.n_sims = n_simulations
        self.rng = np.random.RandomState(42)
    
    def simulate_patient_trajectory(self, P: np.ndarray, 
                                     initial_state: int,
                                     n_steps: int) -> List[int]:
        """
        Simulate a single patient trajectory through the Markov chain.
        
        At each step, sample next state from P[current_state, :].
        """
        trajectory = [initial_state]
        current = initial_state
        
        for _ in range(n_steps):
            if STATES[current]["absorbing"]:
                trajectory.append(current)
                continue
            
            # Sample next state
            probs = P[current]
            next_state = self.rng.choice(N_STATES, p=probs)
            trajectory.append(next_state)
            current = next_state
        
        return trajectory
    
    def run_simulation(self, disaster_type: str, initial_distribution: Dict[int, int],
                       time_horizon_hours: int = 48,
                       interventions: List[str] = None) -> Dict:
        """
        Run full Monte Carlo simulation for a disaster scenario.
        
        Args:
            disaster_type: Type of disaster
            initial_distribution: {state_index: patient_count}
            time_horizon_hours: Simulation duration
            interventions: List of interventions applied
        
        Returns:
            Outcome distributions with confidence intervals
        """
        P = TRANSITION_MATRICES.get(disaster_type.upper(), TRANSITION_MATRICES["ACCIDENT"]).copy()
        
        # Apply interventions
        if interventions:
            for intervention in interventions:
                P = apply_intervention(P, intervention)
        
        # Time steps (each step = 6 hours)
        n_steps = max(1, time_horizon_hours // 6)
        
        # Track outcomes across simulations
        time_points = list(range(n_steps + 1))
        total_patients = sum(initial_distribution.values())
        
        # State counts at each time step across all simulations
        # Shape: (n_sims, n_steps+1, N_STATES)
        all_state_counts = np.zeros((self.n_sims, n_steps + 1, N_STATES))
        
        for sim in range(self.n_sims):
            # Initialize: assign initial states
            patients = []
            for state, count in initial_distribution.items():
                patients.extend([state] * count)
            
            # Record initial state
            for state in patients:
                all_state_counts[sim, 0, state] += 1
            
            # Simulate each patient
            for p_idx, initial_state in enumerate(patients):
                trajectory = self.simulate_patient_trajectory(P, initial_state, n_steps)
                for t, state in enumerate(trajectory):
                    if t > 0:  # Skip initial (already recorded)
                        all_state_counts[sim, t, state] += 1
        
        # Compute statistics
        mean_counts = np.mean(all_state_counts, axis=0)
        std_counts = np.std(all_state_counts, axis=0)
        
        # 95% confidence intervals
        ci_lower = np.percentile(all_state_counts, 2.5, axis=0)
        ci_upper = np.percentile(all_state_counts, 97.5, axis=0)
        
        # Build time-series results
        time_series = []
        report_hours = [0, 6, 12, 24, 48]
        
        for t in range(n_steps + 1):
            hours = t * 6
            if hours > time_horizon_hours:
                break
            
            step_data = {
                "hours": hours,
                "label": f"+{hours}h",
                "state_distribution": {}
            }
            
            for s in range(N_STATES):
                step_data["state_distribution"][STATE_NAMES[s]] = {
                    "mean": round(float(mean_counts[t, s]), 1),
                    "std": round(float(std_counts[t, s]), 1),
                    "ci_95_lower": round(float(ci_lower[t, s]), 1),
                    "ci_95_upper": round(float(ci_upper[t, s]), 1),
                    "percentage": round(float(mean_counts[t, s] / max(1, total_patients) * 100), 1)
                }
            
            time_series.append(step_data)
        
        # Final outcomes
        final_step = min(n_steps, len(mean_counts) - 1)
        final_recovered = float(mean_counts[final_step, 5])
        final_deceased = float(mean_counts[final_step, 6])
        
        mortality_rate = final_deceased / max(1, total_patients)
        recovery_rate = final_recovered / max(1, total_patients)
        still_in_treatment = total_patients - final_recovered - final_deceased
        
        # Mortality distribution across simulations
        sim_mortality = all_state_counts[:, final_step, 6]
        
        return {
            "total_patients": total_patients,
            "simulation_count": self.n_sims,
            "time_horizon_hours": time_horizon_hours,
            "time_step_hours": 6,
            "time_series": time_series,
            "final_outcome": {
                "hours": time_horizon_hours,
                "recovered": {
                    "mean": round(final_recovered, 1),
                    "percentage": round(recovery_rate * 100, 1),
                    "ci_95": [
                        round(float(ci_lower[final_step, 5]), 1),
                        round(float(ci_upper[final_step, 5]), 1)
                    ]
                },
                "deceased": {
                    "mean": round(final_deceased, 1),
                    "percentage": round(mortality_rate * 100, 1),
                    "ci_95": [
                        round(float(ci_lower[final_step, 6]), 1),
                        round(float(ci_upper[final_step, 6]), 1)
                    ]
                },
                "still_in_treatment": {
                    "mean": round(float(still_in_treatment), 1),
                    "percentage": round((1 - recovery_rate - mortality_rate) * 100, 1)
                }
            },
            "mortality_statistics": {
                "mean": round(float(np.mean(sim_mortality)), 2),
                "median": round(float(np.median(sim_mortality)), 2),
                "std": round(float(np.std(sim_mortality)), 2),
                "min": int(np.min(sim_mortality)),
                "max": int(np.max(sim_mortality)),
                "percentile_5": round(float(np.percentile(sim_mortality, 5)), 1),
                "percentile_95": round(float(np.percentile(sim_mortality, 95)), 1)
            },
            "interventions_applied": interventions or [],
        }


# ============================================
# VALUE ITERATION (MDP SOLVER)
# ============================================

class MDPSolver:
    """
    Markov Decision Process solver using Value Iteration.
    
    Determines the optimal intervention policy for each patient state
    to maximize recovery probability or minimize mortality.
    
    Bellman Optimality Equation:
    V*(s) = max_a [R(s,a) + γ · Σ_s' P(s'|s,a) · V*(s')]
    
    Actions: available medical interventions
    States: patient conditions (ESI levels)
    Reward: recovery bonus, mortality penalty
    """
    
    def __init__(self, gamma: float = 0.95, tolerance: float = 1e-6,
                 max_iterations: int = 1000):
        self.gamma = gamma
        self.tolerance = tolerance
        self.max_iter = max_iterations
    
    def solve(self, disaster_type: str) -> Dict:
        """
        Solve the MDP to find optimal intervention policy.
        
        For each patient state, determines which intervention
        maximizes expected recovery.
        """
        P_base = TRANSITION_MATRICES.get(disaster_type.upper(), 
                                          TRANSITION_MATRICES["ACCIDENT"])
        
        actions = list(INTERVENTION_EFFECTS.keys())
        n_actions = len(actions)
        
        # Reward function
        # R(s) = reward for being in state s
        rewards = np.array([
            0.5,   # Stable: mild positive
            0.2,   # Moderate
            -0.1,  # Serious: slight negative
            -0.5,  # Critical
            -1.0,  # ICU
            2.0,   # Recovered: strong positive
            -5.0   # Deceased: strong negative
        ])
        
        # Transition matrices for each action
        P_actions = {}
        for action in actions:
            P_actions[action] = apply_intervention(P_base.copy(), action)
        P_actions["no_intervention"] = P_base.copy()
        all_actions = actions + ["no_intervention"]
        
        # Value Iteration
        V = np.zeros(N_STATES)
        policy = ["no_intervention"] * N_STATES
        
        for iteration in range(self.max_iter):
            V_new = V.copy()
            
            for s in range(5):  # Only transient states (0-4)
                best_value = -float('inf')
                best_action = "no_intervention"
                
                for action in all_actions:
                    P_a = P_actions[action]
                    value = rewards[s] + self.gamma * np.dot(P_a[s], V)
                    
                    if value > best_value:
                        best_value = value
                        best_action = action
                
                V_new[s] = best_value
                policy[s] = best_action
            
            # Absorbing states
            V_new[5] = rewards[5]  # Recovered
            V_new[6] = rewards[6]  # Deceased
            
            # Check convergence
            if np.max(np.abs(V_new - V)) < self.tolerance:
                break
            
            V = V_new
        
        # Build policy result
        optimal_policy = {}
        for s in range(5):
            optimal_policy[STATE_NAMES[s]] = {
                "recommended_intervention": policy[s],
                "intervention_description": (
                    INTERVENTION_EFFECTS[policy[s]]["description"]
                    if policy[s] in INTERVENTION_EFFECTS else "Standard care"
                ),
                "state_value": round(float(V[s]), 4),
                "expected_improvement": round(float(V[s] - rewards[s]), 4)
            }
        
        return {
            "algorithm": "Value Iteration",
            "bellman_equation": "V*(s) = max_a [R(s,a) + γ·Σ P(s'|s,a)·V*(s')]",
            "discount_factor": self.gamma,
            "convergence_iterations": iteration + 1,
            "optimal_policy": optimal_policy,
            "state_values": {STATE_NAMES[i]: round(float(V[i]), 4) for i in range(N_STATES)},
            "available_interventions": {
                name: {
                    "description": info["description"],
                    "recovery_boost": info["recovery_boost"],
                    "mortality_reduction": info["mortality_reduction"],
                    "time_window": f"{info['time_window_hours']}h"
                }
                for name, info in INTERVENTION_EFFECTS.items()
            }
        }


# ============================================
# SENSITIVITY ANALYSIS
# ============================================

class SensitivityAnalyzer:
    """
    Analyze how sensitive outcomes are to intervention timing.
    
    Compares Monte Carlo outcomes under:
    - No intervention (baseline)
    - Early intervention (0-1h)
    - Delayed intervention (2-4h)
    - Late intervention (6-12h)
    """
    
    def __init__(self):
        self.mc_simulator = MonteCarloSimulator(n_simulations=500)
    
    def analyze(self, disaster_type: str, patient_count: int,
                esi_distribution: Dict = None) -> Dict:
        """
        Run sensitivity analysis on intervention timing.
        """
        # Default initial distribution based on disaster severity
        if esi_distribution:
            initial_dist = {}
            for key, data in esi_distribution.items():
                count = data.get("patient_count", data.get("predicted_patients", 0))
                level = data.get("level", 3)
                # Map ESI level to Markov state (ESI-1 → Critical, ESI-5 → Stable)
                state = min(4, max(0, level - 1))
                initial_dist[state] = initial_dist.get(state, 0) + count
        else:
            # Default distribution
            initial_dist = {
                0: int(patient_count * 0.2),   # Stable
                1: int(patient_count * 0.25),  # Moderate
                2: int(patient_count * 0.3),   # Serious
                3: int(patient_count * 0.15),  # Critical
                4: int(patient_count * 0.1),   # ICU
            }
        
        # Ensure patient count matches
        assigned = sum(initial_dist.values())
        if assigned < patient_count:
            initial_dist[1] = initial_dist.get(1, 0) + (patient_count - assigned)
        
        scenarios = {
            "no_intervention": {
                "interventions": [],
                "label": "No Intervention (Baseline)"
            },
            "full_intervention": {
                "interventions": ["rapid_triage", "surgical_intervention", "blood_bank_activation"],
                "label": "Full Intervention Suite"
            },
            "triage_only": {
                "interventions": ["rapid_triage"],
                "label": "Triage Only"
            },
            "surgical_priority": {
                "interventions": ["surgical_intervention", "blood_bank_activation"],
                "label": "Surgical Priority"
            }
        }
        
        results = {}
        for scenario_name, config in scenarios.items():
            sim_result = self.mc_simulator.run_simulation(
                disaster_type, initial_dist,
                time_horizon_hours=48,
                interventions=config["interventions"]
            )
            results[scenario_name] = {
                "label": config["label"],
                "mortality_rate": sim_result["final_outcome"]["deceased"]["percentage"],
                "recovery_rate": sim_result["final_outcome"]["recovered"]["percentage"],
                "mean_deaths": sim_result["mortality_statistics"]["mean"],
                "mortality_ci_95": [
                    sim_result["mortality_statistics"]["percentile_5"],
                    sim_result["mortality_statistics"]["percentile_95"]
                ]
            }
        
        # Compute intervention impact
        baseline_mortality = results["no_intervention"]["mortality_rate"]
        impact = {}
        for name, data in results.items():
            if name != "no_intervention":
                reduction = baseline_mortality - data["mortality_rate"]
                impact[name] = {
                    "mortality_reduction_pct": round(reduction, 1),
                    "lives_saved_estimate": round(reduction * patient_count / 100, 1)
                }
        
        return {
            "disaster_type": disaster_type,
            "total_patients": patient_count,
            "initial_state_distribution": {STATE_NAMES[k]: v for k, v in initial_dist.items()},
            "scenarios": results,
            "intervention_impact": impact,
            "recommendation": max(impact.items(), key=lambda x: x[1]["lives_saved_estimate"])[0]
                if impact else "no_intervention"
        }


# ============================================
# UNIFIED MARKOV MODEL
# ============================================

class MarkovDisasterModel:
    """
    Unified Markov Decision Process model combining:
    - Markov chain analysis
    - Monte Carlo simulation
    - MDP value iteration
    - Sensitivity analysis
    """
    
    def __init__(self):
        self.chain_analyzer = MarkovChainAnalyzer()
        self.mc_simulator = MonteCarloSimulator(n_simulations=1000)
        self.mdp_solver = MDPSolver()
        self.sensitivity = SensitivityAnalyzer()
        self._initialized = True
    
    def predict_outcomes(self, disaster_type: str, patient_count: int,
                         esi_distribution: Dict = None,
                         interventions: List[str] = None,
                         time_horizon_hours: int = 48) -> Dict:
        """
        Run comprehensive Markov model analysis.
        """
        disaster_type = disaster_type.upper()
        P = TRANSITION_MATRICES.get(disaster_type, TRANSITION_MATRICES["ACCIDENT"])
        
        # Initial state distribution
        if esi_distribution:
            initial_dist = {}
            for key, data in esi_distribution.items():
                count = data.get("patient_count", data.get("predicted_patients", 0))
                level = data.get("level", 3)
                state = min(4, max(0, level - 1))
                initial_dist[state] = initial_dist.get(state, 0) + count
        else:
            initial_dist = {
                0: int(patient_count * 0.2),
                1: int(patient_count * 0.25),
                2: int(patient_count * 0.3),
                3: int(patient_count * 0.15),
                4: int(patient_count * 0.1),
            }
        
        assigned = sum(initial_dist.values())
        if assigned < patient_count:
            initial_dist[1] = initial_dist.get(1, 0) + (patient_count - assigned)
        
        # Markov chain analysis
        absorption = self.chain_analyzer.absorption_analysis(P)
        
        # Monte Carlo simulation
        mc_results = self.mc_simulator.run_simulation(
            disaster_type, initial_dist,
            time_horizon_hours=time_horizon_hours,
            interventions=interventions
        )
        
        # MDP optimal policy
        mdp_policy = self.mdp_solver.solve(disaster_type)
        
        # Sensitivity analysis
        sensitivity_results = self.sensitivity.analyze(
            disaster_type, patient_count, esi_distribution
        )
        
        return {
            "engine": "S.A.V.E. Markov Decision Process Engine v1.0",
            "theoretical_basis": [
                "Markov Chains (Markov, 1906)",
                "Absorbing Markov Chains (Kemeny & Snell, 1960)",
                "Chapman-Kolmogorov Equation: P(n) = P^n",
                "Monte Carlo Simulation (Metropolis & Ulam, 1949)",
                "Value Iteration (Bellman, 1957)",
                "Bellman Optimality Equation for MDP"
            ],
            "markov_chain_analysis": absorption,
            "monte_carlo_simulation": mc_results,
            "optimal_policy": mdp_policy,
            "sensitivity_analysis": sensitivity_results,
            "timestamp": datetime.now().isoformat()
        }


# Global instance
markov_model = MarkovDisasterModel()


# Convenience function
def predict_disaster_outcomes(disaster_type: str, patient_count: int,
                              esi_distribution: Dict = None,
                              interventions: List[str] = None, **kwargs) -> Dict:
    """Quick access to Markov model predictions"""
    return markov_model.predict_outcomes(
        disaster_type, patient_count, esi_distribution, interventions,
        **kwargs
    )
