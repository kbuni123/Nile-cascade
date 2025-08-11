# dam_specifications.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class DamSpecifications:
    """Dam technical specifications based on comprehensive research"""
    name: str
    max_capacity: float  # MCM (Million Cubic Meters)
    min_capacity: float  # MCM
    live_capacity: float  # MCM
    turbine_capacity: float  # MW
    min_env_flow: float  # m³/s
    spillway_capacity: float  # m³/s
    efficiency: float  # Turbine efficiency (0-1)
    max_head: float  # meters
    min_head: float  # meters

class NileCascadeDams:
    """Complete Nile cascade dam specifications"""
    
    def __init__(self):
        self.dams = self._initialize_dam_specs()
        self.cascade_order = [
            'GERD', 'Roseires', 'Sennar', 'Merowe', 
            'Jebel_Aulia', 'Aswan_High', 'Khashm_el_Girba'
        ]
    @property
    def dam_names(self):
        return list(self.dams.keys())

    
    def _initialize_dam_specs(self) -> Dict[str, DamSpecifications]:
        """Initialize all dam specifications based on research data"""
        return {
            'GERD': DamSpecifications(
                name="Grand Ethiopian Renaissance Dam",
                max_capacity=74000,  # MCM
                min_capacity=14800,  # MCM (dead storage)
                live_capacity=59200,  # MCM
                turbine_capacity=5150,  # MW
                min_env_flow=35000/(365*24*3600)*1e6,  # Convert from BCM/year to m³/s
                spillway_capacity=15000,  # m³/s
                efficiency=0.90,
                max_head=140,  # meters
                min_head=45   # meters (minimum operating head)
            ),
            
            'Aswan_High': DamSpecifications(
                name="Aswan High Dam",
                max_capacity=162000,  # MCM
                min_capacity=31600,   # MCM (dead storage)
                live_capacity=90700,  # MCM
                turbine_capacity=2100,  # MW
                min_env_flow=55500/(365*24*3600)*1e6,  # Convert from BCM/year
                spillway_capacity=16000,  # m³/s
                efficiency=0.95,
                max_head=74,  # meters
                min_head=30   # meters
            ),
            
            'Roseires': DamSpecifications(
                name="Roseires Dam",
                max_capacity=7400,    # MCM (post-heightening)
                min_capacity=1000,    # MCM (estimated minimum)
                live_capacity=6400,   # MCM
                turbine_capacity=280, # MW
                min_env_flow=800,     # m³/s (estimated)
                spillway_capacity=7500,  # m³/s (low-level outlets)
                efficiency=0.88,
                max_head=45,  # meters
                min_head=20   # meters
            ),
            
            'Sennar': DamSpecifications(
                name="Sennar Dam",
                max_capacity=930,     # MCM (original capacity)
                min_capacity=200,     # MCM (estimated after sedimentation)
                live_capacity=390,    # MCM (current effective capacity)
                turbine_capacity=15,  # MW
                min_env_flow=500,     # m³/s (estimated)
                spillway_capacity=5000,  # m³/s (estimated total capacity)
                efficiency=0.85,
                max_head=25,  # meters
                min_head=10   # meters
            ),
            
            'Merowe': DamSpecifications(
                name="Merowe Dam",
                max_capacity=12500,   # MCM
                min_capacity=2000,    # MCM (estimated)
                live_capacity=10500,  # MCM
                turbine_capacity=1250, # MW
                min_env_flow=1000,    # m³/s (estimated)
                spillway_capacity=8000,  # m³/s (estimated)
                efficiency=0.92,
                max_head=35,  # meters
                min_head=15   # meters
            ),
            
            'Jebel_Aulia': DamSpecifications(
                name="Jebel Aulia Dam",
                max_capacity=5000,    # MCM
                min_capacity=1000,    # MCM (estimated)
                live_capacity=4000,   # MCM
                turbine_capacity=30,  # MW (Hydromatrix system)
                min_env_flow=300,     # m³/s (White Nile minimum)
                spillway_capacity=3000,  # m³/s (estimated)
                efficiency=0.80,      # Lower due to low head
                max_head=15,  # meters
                min_head=5    # meters
            ),
            
            'Khashm_el_Girba': DamSpecifications(
                name="Khashm el-Girba Dam",
                max_capacity=800,     # MCM
                min_capacity=200,     # MCM (estimated)
                live_capacity=600,    # MCM
                turbine_capacity=10,  # MW (upgraded)
                min_env_flow=200,     # m³/s (Atbara River minimum)
                spillway_capacity=2000,  # m³/s (estimated)
                efficiency=0.82,
                max_head=20,  # meters
                min_head=8    # meters
            )
        }
    
    def get_dam_specs(self, dam_name: str) -> Optional[DamSpecifications]:
        """Get specifications for a specific dam"""
        return self.dams.get(dam_name)
    
    def get_cascade_capacity(self) -> float:
        """Get total cascade storage capacity"""
        return sum(dam.max_capacity for dam in self.dams.values())
    
    def get_cascade_power(self) -> float:
        """Get total cascade power generation capacity"""
        return sum(dam.turbine_capacity for dam in self.dams.values())

# Usage
nile_dams = NileCascadeDams()
print(f"Total cascade capacity: {nile_dams.get_cascade_capacity():,.0f} MCM")
print(f"Total cascade power: {nile_dams.get_cascade_power():,.0f} MW")