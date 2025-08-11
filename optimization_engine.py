# optimization_engine.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.optimize as opt
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import warnings
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize as minimize_moo
from pymoo.termination import get_termination

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Optimization results container"""
    reservoir_levels: Dict[str, List[float]]
    power_generation: Dict[str, List[float]]
    water_releases: Dict[str, List[float]]
    objective_values: Dict[str, float]
    feasible: bool
    solver_status: str

class NileReservoirOptimization(Problem):
    """Multi-objective optimization problem for Nile cascade"""
    
    def __init__(self, dams_data, weather_data, time_steps=12, **kwargs):
        self.dams_data = dams_data
        self.weather_data = weather_data
        self.time_steps = time_steps
        self.dam_names = list(dams_data.dams.keys())
        self.n_dams = len(self.dam_names)
        
        # Decision variables: reservoir levels and releases for each dam at each time step
        n_vars = self.n_dams * self.time_steps * 2  # levels + releases
        
        # Variable bounds (normalized 0-1, will scale in constraints)
        xl = np.zeros(n_vars)
        xu = np.ones(n_vars)
        
        super().__init__(n_var=n_vars, n_obj=3, n_constr=0, xl=xl, xu=xu, **kwargs)
    
    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate objectives and constraints"""
        n_solutions = X.shape[0]
        F = np.zeros((n_solutions, 3))  # 3 objectives
        
        for i, x in enumerate(X):
            try:
                # Decode decision variables
                levels, releases = self._decode_variables(x)
                
                # Calculate objectives
                power_obj = self._calculate_power_objective(levels, releases)
                flood_obj = self._calculate_flood_objective(levels)
                water_security_obj = self._calculate_water_security_objective(releases)
                
                F[i, 0] = -power_obj  # Negative because we want to maximize
                F[i, 1] = flood_obj   # Minimize flood risk
                F[i, 2] = -water_security_obj  # Negative because we want to maximize
                
            except Exception as e:
                # Assign high penalty for infeasible solutions
                F[i, :] = [1e6, 1e6, 1e6]
        
        out["F"] = F
    
    def _decode_variables(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode normalized decision variables to actual levels and releases"""
        # Split variables into levels and releases
        mid_point = len(x) // 2
        levels_norm = x[:mid_point].reshape(self.n_dams, self.time_steps)
        releases_norm = x[mid_point:].reshape(self.n_dams, self.time_steps)
        
        # Scale to actual values
        levels = np.zeros_like(levels_norm)
        releases = np.zeros_like(releases_norm)
        
        for i, dam_name in enumerate(self.dam_names):
            dam_spec = self.dams_data.get_dam_specs(dam_name)
            
            # Scale reservoir levels (MCM)
            levels[i, :] = (dam_spec.min_capacity + 
                          levels_norm[i, :] * (dam_spec.max_capacity - dam_spec.min_capacity))
            
            # Scale releases (m³/s)
            max_release = dam_spec.spillway_capacity
            releases[i, :] = releases_norm[i, :] * max_release
        
        return levels, releases
    
    def _calculate_power_objective(self, levels: np.ndarray, releases: np.ndarray) -> float:
        """Calculate total power generation (maximize)"""
        total_power = 0.0
        
        for i, dam_name in enumerate(self.dam_names):
            dam_spec = self.dams_data.get_dam_specs(dam_name)
            
            for t in range(self.time_steps):
                # Calculate effective head based on reservoir level
                level_ratio = (levels[i, t] - dam_spec.min_capacity) / (dam_spec.max_capacity - dam_spec.min_capacity)
                effective_head = dam_spec.min_head + level_ratio * (dam_spec.max_head - dam_spec.min_head)
                
                # Power generation (MW) = efficiency * density * gravity * flow * head / 1e6
                flow = min(releases[i, t], dam_spec.turbine_capacity * 100)  # m³/s, rough conversion
                power = (dam_spec.efficiency * 1000 * 9.81 * flow * effective_head) / 1e6
                power = min(power, dam_spec.turbine_capacity)  # Cap at installed capacity
                
                total_power += power
        
        return total_power
    
    def _calculate_flood_objective(self, levels: np.ndarray) -> float:
        """Calculate flood risk (minimize)"""
        flood_risk = 0.0
        
        for i, dam_name in enumerate(self.dam_names):
            dam_spec = self.dams_data.get_dam_specs(dam_name)
            
            for t in range(self.time_steps):
                # Flood risk increases exponentially as we approach maximum capacity
                level_ratio = levels[i, t] / dam_spec.max_capacity
                if level_ratio > 0.85:  # Risk increases above 85% capacity
                    flood_risk += np.exp(10 * (level_ratio - 0.85))
        
        return flood_risk
    
    def _calculate_water_security_objective(self, releases: np.ndarray) -> float:
        """Calculate water supply security (maximize)"""
        security_score = 0.0
        
        for i, dam_name in enumerate(self.dam_names):
            dam_spec = self.dams_data.get_dam_specs(dam_name)
            
            for t in range(self.time_steps):
                # Water security based on meeting minimum environmental flows
                if releases[i, t] >= dam_spec.min_env_flow:
                    security_score += 1.0
                else:
                    # Penalty for not meeting minimum flows
                    security_score += releases[i, t] / dam_spec.min_env_flow
        
        return security_score / (self.n_dams * self.time_steps)

class WaterResourceOptimizer:
    """High-level optimizer interface"""
    
    def __init__(self, dams_data, weather_data):
        self.dams_data = dams_data
        self.weather_data = weather_data
        self.results = None
    
    def optimize_cascade(self, 
                        time_horizon: int = 12,
                        population_size: int = 100,
                        generations: int = 100,
                        weights: Dict[str, float] = None) -> OptimizationResult:
        """
        Optimize the Nile cascade system
        
        Args:
            time_horizon: Planning horizon in months
            population_size: NSGA-II population size
            generations: Number of generations
            weights: Objective weights {'power': 0.4, 'flood': 0.3, 'water': 0.3}
        """
        
        if weights is None:
            weights = {'power': 0.4, 'flood': 0.3, 'water': 0.3}
        
        logger.info(f"Starting multi-objective optimization...")
        logger.info(f"Time horizon: {time_horizon} months")
        logger.info(f"Population size: {population_size}")
        logger.info(f"Objectives weights: {weights}")
        
        try:
            # Define optimization problem
            problem = NileReservoirOptimization(
                dams_data=self.dams_data,
                weather_data=self.weather_data,
                time_steps=time_horizon
            )
            
            # Configure NSGA-II algorithm
            algorithm = NSGA2(pop_size=population_size)
            
            # Define termination criteria
            termination = get_termination("n_gen", generations)
            
            # Run optimization
            res = minimize_moo(
                problem,
                algorithm,
                termination,
                verbose=True,
                seed=42
            )
            
            # Process results
            if res.X is not None:
                # Select best solution using weighted sum
                best_idx = self._select_best_solution(res.F, weights)
                best_solution = res.X[best_idx]
                
                # Decode solution
                levels, releases = problem._decode_variables(best_solution)
                
                # Create result object
                result = self._create_optimization_result(
                    levels, releases, res.F[best_idx], True, "Success"
                )
                
                self.results = result
                logger.info("Optimization completed successfully")
                return result
                
            else:
                logger.error("Optimization failed - no feasible solution found")
                return OptimizationResult(
                    reservoir_levels={},
                    power_generation={},
                    water_releases={},
                    objective_values={},
                    feasible=False,
                    solver_status="No feasible solution"
                )
                
        except Exception as e:
            logger.error(f"Optimization error: {str(e)}")
            return OptimizationResult(
                reservoir_levels={},
                power_generation={},
                water_releases={},
                objective_values={},
                feasible=False,
                solver_status=f"Error: {str(e)}"
            )
    
    def _select_best_solution(self, pareto_front: np.ndarray, weights: Dict[str, float]) -> int:
        """Select best solution from Pareto front using weighted sum"""
        # Normalize objectives to 0-1 range
        normalized_front = (pareto_front - pareto_front.min(axis=0)) / (pareto_front.max(axis=0) - pareto_front.min(axis=0) + 1e-10)
        
        # Calculate weighted sum (note: objectives are negated for maximization)
        weighted_scores = (
            -weights['power'] * normalized_front[:, 0] +    # Power (negated, so minimize negative)
            weights['flood'] * normalized_front[:, 1] +     # Flood risk (minimize)
            -weights['water'] * normalized_front[:, 2]      # Water security (negated)
        )
        
        return np.argmin(weighted_scores)
    
    def _create_optimization_result(self, levels: np.ndarray, releases: np.ndarray, 
                                  objectives: np.ndarray, feasible: bool, status: str) -> OptimizationResult:
        """Create optimization result object"""
        
        reservoir_levels = {}
        power_generation = {}
        water_releases = {}
        
        for i, dam_name in enumerate(self.dams_data.dam_names):
            reservoir_levels[dam_name] = levels[i, :].tolist()
            water_releases[dam_name] = releases[i, :].tolist()
            
            # Calculate power generation
            dam_spec = self.dams_data.get_dam_specs(dam_name)
            power_gen = []
            
            for t in range(levels.shape[1]):
                level_ratio = (levels[i, t] - dam_spec.min_capacity) / (dam_spec.max_capacity - dam_spec.min_capacity)
                effective_head = dam_spec.min_head + level_ratio * (dam_spec.max_head - dam_spec.min_head)
                flow = min(releases[i, t], dam_spec.turbine_capacity * 100)
                power = (dam_spec.efficiency * 1000 * 9.81 * flow * effective_head) / 1e6
                power = min(power, dam_spec.turbine_capacity)
                power_gen.append(max(0, power))
            
            power_generation[dam_name] = power_gen
        
        objective_values = {
            'total_power': -objectives[0],  # Convert back to positive
            'flood_risk': objectives[1],
            'water_security': -objectives[2]  # Convert back to positive
        }
        
        return OptimizationResult(
            reservoir_levels=reservoir_levels,
            power_generation=power_generation,
            water_releases=water_releases,
            objective_values=objective_values,
            feasible=feasible,
            solver_status=status
        )

# Usage example
if __name__ == "__main__":
    # Initialize components
    from dam_specifications import NileCascadeDams
    from weather_integration import FreeWeatherDataManager
    
    dams = NileCascadeDams()
    weather = FreeWeatherDataManager()
    weather_data = weather.get_basin_forecast(days=30, use_historical=False)
    
    # Run optimization
    optimizer = WaterResourceOptimizer(dams, weather_data)
    result = optimizer.optimize_cascade(
        time_horizon=6,  # 6 months
        population_size=50,
        generations=50,
        weights={'power': 0.4, 'flood': 0.3, 'water': 0.3}
    )
    
    if result.feasible:
        print("Optimization successful!")
        print(f"Total power generation: {result.objective_values['total_power']:.1f} MW")
        print(f"Flood risk score: {result.objective_values['flood_risk']:.2f}")
        print(f"Water security score: {result.objective_values['water_security']:.2f}")
    else:
        print(f"Optimization failed: {result.solver_status}")