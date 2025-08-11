# ai_interpreter.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Interpretation:
    """Container for AI-generated interpretations"""
    summary: str
    key_insights: List[str]
    warnings: List[str]
    recommendations: List[str]
    technical_notes: Optional[str] = None

class OptimizationInterpreter:
    """AI module for interpreting optimization results in natural language"""
    
    def __init__(self, dams_data):
        self.dams_data = dams_data
        self.critical_thresholds = {
            'flood_risk': 0.85,  # 85% capacity
            'power_efficiency': 0.70,  # 70% of max power
            'min_flow_violation': 0.90,  # 90% of minimum flow
            'rapid_change': 0.20  # 20% change between periods
        }
    
    def interpret_optimization_results(self, result, weights: Dict[str, float], 
                                      time_horizon: int) -> Interpretation:
        """Generate comprehensive interpretation of optimization results"""
        
        # Analyze overall performance
        summary = self._generate_summary(result, weights, time_horizon)
        
        # Extract key insights
        insights = self._extract_key_insights(result, time_horizon)
        
        # Identify warnings
        warnings = self._identify_warnings(result)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(result, weights)
        
        # Add technical notes
        technical_notes = self._generate_technical_notes(result)
        
        return Interpretation(
            summary=summary,
            key_insights=insights,
            warnings=warnings,
            recommendations=recommendations,
            technical_notes=technical_notes
        )
    
    def _generate_summary(self, result, weights: Dict[str, float], time_horizon: int) -> str:
        """Generate executive summary of optimization results"""
        
        total_power = result.objective_values.get('total_power', 0)
        flood_risk = result.objective_values.get('flood_risk', 0)
        water_security = result.objective_values.get('water_security', 0)
        
        # Determine optimization focus based on weights
        priority = max(weights.items(), key=lambda x: x[1])[0]
        priority_map = {
            'power': 'power generation',
            'flood': 'flood control',
            'water': 'water security'
        }
        
        # Calculate average utilization
        avg_power_util = self._calculate_average_power_utilization(result)
        avg_storage_util = self._calculate_average_storage_utilization(result)
        
        summary = f"""
        The optimization successfully balanced multiple objectives over a {time_horizon}-month planning horizon, 
        with primary focus on {priority_map[priority]} ({weights[priority]*100:.0f}% weight).
        
        The cascade system achieved an average power generation of {total_power:.0f} MW with {avg_power_util:.1f}% 
        utilization of installed capacity. Reservoir storage averaged {avg_storage_util:.1f}% of total capacity, 
        maintaining a flood risk score of {flood_risk:.2f} while ensuring {water_security*100:.1f}% water security 
        compliance across all dams.
        """
        
        return summary.strip()
    
    def _extract_key_insights(self, result, time_horizon: int) -> List[str]:
        """Extract key insights from optimization results"""
        insights = []
        
        # Analyze power generation patterns
        for dam_name, power_gen in result.power_generation.items():
            power_array = np.array(power_gen)
            dam_spec = self.dams_data.get_dam_specs(dam_name)
            
            if dam_spec:
                avg_power = np.mean(power_array)
                max_power = np.max(power_array)
                utilization = (avg_power / dam_spec.turbine_capacity) * 100
                
                if utilization > 80:
                    insights.append(f"🔋 {dam_spec.name} operating at high capacity ({utilization:.1f}% average utilization)")
                elif utilization < 30:
                    insights.append(f"⚡ {dam_spec.name} has significant unused power capacity ({utilization:.1f}% utilization)")
                
                # Check for seasonal patterns
                if time_horizon >= 6:
                    first_half = np.mean(power_array[:time_horizon//2])
                    second_half = np.mean(power_array[time_horizon//2:])
                    if abs(first_half - second_half) / max(first_half, second_half) > 0.3:
                        trend = "increasing" if second_half > first_half else "decreasing"
                        insights.append(f"📊 {dam_spec.name} shows {trend} power generation trend over the planning period")
        
        # Analyze storage patterns
        for dam_name, levels in result.reservoir_levels.items():
            levels_array = np.array(levels)
            dam_spec = self.dams_data.get_dam_specs(dam_name)
            
            if dam_spec:
                # Check for consistent drawdown or filling
                trend = np.polyfit(range(len(levels_array)), levels_array, 1)[0]
                if abs(trend) > (dam_spec.max_capacity - dam_spec.min_capacity) / (time_horizon * 10):
                    direction = "filling" if trend > 0 else "drawing down"
                    insights.append(f"💧 {dam_spec.name} is consistently {direction} over the planning period")
                
                # Check for critical levels
                min_level = np.min(levels_array)
                max_level = np.max(levels_array)
                if min_level < dam_spec.min_capacity * 1.2:
                    insights.append(f"⚠️ {dam_spec.name} approaches minimum operating level in some periods")
                if max_level > dam_spec.max_capacity * 0.9:
                    insights.append(f"🌊 {dam_spec.name} reaches near-maximum capacity, increasing flood risk")
        
        # Analyze water releases
        total_env_compliance = 0
        total_checks = 0
        
        for dam_name, releases in result.water_releases.items():
            dam_spec = self.dams_data.get_dam_specs(dam_name)
            if dam_spec:
                compliance = sum(1 for r in releases if r >= dam_spec.min_env_flow) / len(releases)
                total_env_compliance += compliance
                total_checks += 1
                
                if compliance < 0.8:
                    insights.append(f"🚨 {dam_spec.name} struggles to meet environmental flow requirements ({compliance*100:.0f}% compliance)")
        
        if total_checks > 0:
            avg_compliance = total_env_compliance / total_checks
            if avg_compliance > 0.95:
                insights.append(f"✅ Excellent environmental flow compliance across the cascade ({avg_compliance*100:.0f}%)")
        
        return insights[:8]  # Limit to 8 most important insights
    
    def _identify_warnings(self, result) -> List[str]:
        """Identify potential issues or warnings in the results"""
        warnings = []
        
        # Check for rapid changes in reservoir levels
        for dam_name, levels in result.reservoir_levels.items():
            dam_spec = self.dams_data.get_dam_specs(dam_name)
            if dam_spec:
                levels_array = np.array(levels)
                for i in range(1, len(levels_array)):
                    change_rate = abs(levels_array[i] - levels_array[i-1]) / dam_spec.max_capacity
                    if change_rate > self.critical_thresholds['rapid_change']:
                        warnings.append(f"⚠️ Rapid level change detected at {dam_spec.name} in period {i}")
                        break
        
        # Check for sustained high flood risk
        if result.objective_values.get('flood_risk', 0) > 5:
            warnings.append("🌊 High flood risk score indicates potential overflow conditions")
        
        # Check for power generation anomalies
        for dam_name, power_gen in result.power_generation.items():
            if any(p < 0 for p in power_gen):
                warnings.append(f"❌ Negative power generation detected at {dam_name}")
            
            dam_spec = self.dams_data.get_dam_specs(dam_name)
            if dam_spec and any(p > dam_spec.turbine_capacity * 1.1 for p in power_gen):
                warnings.append(f"⚡ Power generation exceeds installed capacity at {dam_name}")
        
        # Check water balance
        for dam_name, releases in result.water_releases.items():
            dam_spec = self.dams_data.get_dam_specs(dam_name)
            if dam_spec:
                avg_release = np.mean(releases)
                if avg_release < dam_spec.min_env_flow * 0.5:
                    warnings.append(f"💧 Critically low water releases from {dam_spec.name}")
                elif avg_release > dam_spec.spillway_capacity * 0.9:
                    warnings.append(f"🚨 Near spillway capacity releases from {dam_spec.name}")
        
        return warnings[:5]  # Limit to 5 most critical warnings
    
    def _generate_recommendations(self, result, weights: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations based on results"""
        recommendations = []
        
        # Analyze underutilized capacity
        for dam_name, power_gen in result.power_generation.items():
            dam_spec = self.dams_data.get_dam_specs(dam_name)
            if dam_spec:
                avg_power = np.mean(power_gen)
                utilization = avg_power / dam_spec.turbine_capacity
                
                if utilization < 0.5 and weights['power'] > 0.3:
                    recommendations.append(
                        f"Consider increasing water allocation to {dam_spec.name} to improve power generation "
                        f"(currently at {utilization*100:.0f}% capacity)"
                    )
        
        # Storage management recommendations
        for dam_name, levels in result.reservoir_levels.items():
            dam_spec = self.dams_data.get_dam_specs(dam_name)
            if dam_spec:
                avg_level = np.mean(levels)
                level_ratio = (avg_level - dam_spec.min_capacity) / (dam_spec.max_capacity - dam_spec.min_capacity)
                
                if level_ratio > 0.85:
                    recommendations.append(
                        f"Maintain lower average levels at {dam_spec.name} to improve flood control capacity"
                    )
                elif level_ratio < 0.3:
                    recommendations.append(
                        f"Consider maintaining higher reserve levels at {dam_spec.name} for drought resilience"
                    )
        
        # Coordination recommendations
        if len(result.reservoir_levels) > 3:
            # Check for cascade coordination
            downstream_increasing = 0
            upstream_decreasing = 0
            
            cascade_order = ['GERD', 'Roseires', 'Sennar', 'Merowe', 'Aswan_High']
            for i, dam in enumerate(cascade_order[:-1]):
                if dam in result.reservoir_levels and cascade_order[i+1] in result.reservoir_levels:
                    upstream_trend = np.polyfit(range(len(result.reservoir_levels[dam])), 
                                              result.reservoir_levels[dam], 1)[0]
                    downstream_trend = np.polyfit(range(len(result.reservoir_levels[cascade_order[i+1]])), 
                                                result.reservoir_levels[cascade_order[i+1]], 1)[0]
                    
                    if upstream_trend < 0 and downstream_trend > 0:
                        recommendations.append(
                            f"Good cascade coordination observed between {dam} and {cascade_order[i+1]}"
                        )
                        break
        
        # Seasonal recommendations
        if result.objective_values.get('flood_risk', 0) > 2:
            recommendations.append(
                "Pre-release water before peak rainfall season to create flood storage capacity"
            )
        
        if result.objective_values.get('water_security', 0) < 0.8:
            recommendations.append(
                "Review minimum environmental flow requirements and consider adaptive management strategies"
            )
        
        return recommendations[:6]  # Limit to 6 most actionable recommendations
    
    def _generate_technical_notes(self, result) -> str:
        """Generate technical notes for advanced users"""
        
        notes = []
        
        # Calculate system efficiency metrics
        total_power_capacity = sum(self.dams_data.get_dam_specs(dam).turbine_capacity 
                                  for dam in result.power_generation.keys())
        avg_total_power = sum(np.mean(power) for power in result.power_generation.values())
        system_efficiency = (avg_total_power / total_power_capacity) * 100
        
        notes.append(f"System-wide power efficiency: {system_efficiency:.1f}%")
        
        # Calculate water balance metrics
        total_inflow_estimate = sum(np.mean(releases) for releases in result.water_releases.values())
        notes.append(f"Estimated total cascade throughput: {total_inflow_estimate:.0f} m³/s average")
        
        # Storage turnover rate
        total_storage = sum(self.dams_data.get_dam_specs(dam).max_capacity 
                          for dam in result.reservoir_levels.keys())
        avg_storage_used = sum(np.mean(levels) for levels in result.reservoir_levels.values())
        storage_utilization = (avg_storage_used / total_storage) * 100
        notes.append(f"Cascade storage utilization: {storage_utilization:.1f}%")
        
        # Optimization convergence quality
        if result.feasible:
            notes.append("Optimization converged to feasible solution within constraints")
        else:
            notes.append("Optimization did not find fully feasible solution - results may violate some constraints")
        
        return " | ".join(notes)
    
    def _calculate_average_power_utilization(self, result) -> float:
        """Calculate average power utilization across cascade"""
        total_utilization = 0
        count = 0
        
        for dam_name, power_gen in result.power_generation.items():
            dam_spec = self.dams_data.get_dam_specs(dam_name)
            if dam_spec and dam_spec.turbine_capacity > 0:
                avg_power = np.mean(power_gen)
                utilization = (avg_power / dam_spec.turbine_capacity) * 100
                total_utilization += utilization
                count += 1
        
        return total_utilization / count if count > 0 else 0
    
    def _calculate_average_storage_utilization(self, result) -> float:
        """Calculate average storage utilization across cascade"""
        total_utilization = 0
        count = 0
        
        for dam_name, levels in result.reservoir_levels.items():
            dam_spec = self.dams_data.get_dam_specs(dam_name)
            if dam_spec:
                avg_level = np.mean(levels)
                utilization = (avg_level / dam_spec.max_capacity) * 100
                total_utilization += utilization
                count += 1
        
        return total_utilization / count if count > 0 else 0
    
    def interpret_reservoir_levels_chart(self, result, time_horizon: int) -> str:
        """Generate natural language interpretation of reservoir levels chart"""
        
        interpretation = "📊 **Reservoir Levels Analysis:**\n\n"
        
        # Identify patterns
        patterns = []
        for dam_name, levels in result.reservoir_levels.items():
            dam_spec = self.dams_data.get_dam_specs(dam_name)
            if dam_spec:
                levels_array = np.array(levels)
                
                # Trend analysis
                trend = np.polyfit(range(len(levels_array)), levels_array, 1)[0]
                start_level = levels_array[0]
                end_level = levels_array[-1]
                avg_level = np.mean(levels_array)
                
                # Generate description
                if abs(trend) < (dam_spec.max_capacity * 0.01):  # Less than 1% change per period
                    pattern = f"**{dam_spec.name}** maintains stable levels around {avg_level:.0f} MCM"
                elif trend > 0:
                    change_pct = ((end_level - start_level) / start_level) * 100
                    pattern = f"**{dam_spec.name}** shows filling trend (+{change_pct:.1f}% over period)"
                else:
                    change_pct = ((start_level - end_level) / start_level) * 100
                    pattern = f"**{dam_spec.name}** shows drawdown trend (-{change_pct:.1f}% over period)"
                
                # Add operational context
                utilization = (avg_level / dam_spec.max_capacity) * 100
                if utilization > 80:
                    pattern += f" (high storage at {utilization:.0f}% capacity)"
                elif utilization < 40:
                    pattern += f" (conservative storage at {utilization:.0f}% capacity)"
                else:
                    pattern += f" (moderate storage at {utilization:.0f}% capacity)"
                
                patterns.append(pattern)
        
        interpretation += "\n".join(patterns[:5])  # Limit to 5 dams
        
        # Add seasonal context if applicable
        if time_horizon >= 6:
            interpretation += "\n\n**Seasonal Pattern:** "
            interpretation += "The optimization accounts for seasonal variations in inflow and demand, "
            interpretation += "with coordinated reservoir operations to balance competing objectives."
        
        return interpretation
    
    def interpret_power_generation_chart(self, result, weights: Dict[str, float]) -> str:
        """Generate natural language interpretation of power generation chart"""
        
        interpretation = "⚡ **Power Generation Analysis:**\n\n"
        
        # Calculate total and patterns
        total_capacity = sum(self.dams_data.get_dam_specs(dam).turbine_capacity 
                           for dam in result.power_generation.keys())
        
        power_patterns = []
        high_performers = []
        low_performers = []
        
        for dam_name, power_gen in result.power_generation.items():
            dam_spec = self.dams_data.get_dam_specs(dam_name)
            if dam_spec:
                avg_power = np.mean(power_gen)
                max_power = np.max(power_gen)
                min_power = np.min(power_gen)
                utilization = (avg_power / dam_spec.turbine_capacity) * 100
                
                if utilization > 70:
                    high_performers.append((dam_spec.name, utilization, avg_power))
                elif utilization < 40:
                    low_performers.append((dam_spec.name, utilization, avg_power))
        
        # High performers
        if high_performers:
            interpretation += "**High-efficiency operations:**\n"
            for name, util, power in high_performers[:3]:
                interpretation += f"• {name}: {power:.0f} MW average ({util:.0f}% utilization)\n"
        
        # Low performers
        if low_performers and weights['power'] > 0.3:
            interpretation += "\n**Optimization opportunities:**\n"
            for name, util, power in low_performers[:3]:
                interpretation += f"• {name}: Operating at {util:.0f}% capacity - potential for increased generation\n"
        
        # Overall assessment
        total_avg_power = sum(np.mean(p) for p in result.power_generation.values())
        system_utilization = (total_avg_power / total_capacity) * 100
        
        interpretation += f"\n**System Performance:** {total_avg_power:.0f} MW total generation "
        interpretation += f"({system_utilization:.0f}% of {total_capacity:.0f} MW installed capacity)"
        
        if weights['power'] > 0.5:
            interpretation += "\n\n*Note: Power generation was the primary optimization objective.*"
        
        return interpretation
    
    def interpret_water_releases_chart(self, result) -> str:
        """Generate natural language interpretation of water releases chart"""
        
        interpretation = "💧 **Water Release Analysis:**\n\n"
        
        # Analyze environmental flow compliance
        compliance_status = []
        critical_periods = []
        
        for dam_name, releases in result.water_releases.items():
            dam_spec = self.dams_data.get_dam_specs(dam_name)
            if dam_spec:
                releases_array = np.array(releases)
                avg_release = np.mean(releases_array)
                min_release = np.min(releases_array)
                max_release = np.max(releases_array)
                
                # Check environmental compliance
                compliance_rate = sum(1 for r in releases_array if r >= dam_spec.min_env_flow) / len(releases_array)
                
                if compliance_rate == 1.0:
                    status = f"✅ **{dam_spec.name}**: Full environmental flow compliance "
                    status += f"(avg: {avg_release:.0f} m³/s, required: {dam_spec.min_env_flow:.0f} m³/s)"
                elif compliance_rate > 0.8:
                    status = f"⚠️ **{dam_spec.name}**: Mostly compliant ({compliance_rate*100:.0f}%) "
                    status += f"with occasional drops below {dam_spec.min_env_flow:.0f} m³/s"
                else:
                    status = f"🚨 **{dam_spec.name}**: Compliance challenges ({compliance_rate*100:.0f}%) "
                    status += f"- consider operational adjustments"
                
                compliance_status.append(status)
                
                # Check for critical periods
                for i, release in enumerate(releases_array):
                    if release > dam_spec.spillway_capacity * 0.8:
                        critical_periods.append(f"Period {i+1}: High release at {dam_spec.name} "
                                              f"({release:.0f} m³/s, near spillway capacity)")
        
        interpretation += "\n".join(compliance_status[:5])
        
        if critical_periods:
            interpretation += "\n\n**Critical Periods:**\n"
            interpretation += "\n".join(critical_periods[:3])
        
        # Add coordination insight
        interpretation += "\n\n**Cascade Coordination:** "
        interpretation += "Water releases are optimized to balance downstream requirements "
        interpretation += "while maintaining operational flexibility for each reservoir."
        
        return interpretation

# Usage in Streamlit app - add this to app.py after optimization results