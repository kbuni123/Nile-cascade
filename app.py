# app.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dam_specifications import NileCascadeDams
from weather_integration import FreeWeatherDataManager
from optimization_engine import WaterResourceOptimizer
import time

# Configure page
st.set_page_config(
    page_title="Nile Cascade Water Resource Optimizer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache initialization
@st.cache_resource
def initialize_system():
    """Initialize dam specifications and weather manager"""
    dams = NileCascadeDams()
    weather = FreeWeatherDataManager()
    return dams, weather

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_weather_data(days=14, use_historical=False):
    """Load weather data with caching"""
    _, weather = initialize_system()
    return weather.get_basin_forecast(days=days, use_historical=use_historical)

def main():
    # Header
    st.title("🌊 Nile River Cascade Water Resource Optimization Tool")
    st.markdown("""
    **Comprehensive water resource management for the Nile River cascade system**  
    *Integrating GERD, Aswan High Dam, and all major Sudanese dams with multi-objective optimization*
    """)
    
    # Initialize system
    dams, weather_manager = initialize_system()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Optimization Parameters")
    
    # Weather data source selection
    st.sidebar.subheader("📊 Weather Data Source")
    weather_source = st.sidebar.radio(
        "Select Data Source",
        options=["Free APIs", "Historical Patterns"],
        index=0,
        help="""
        **Free APIs**: Uses Open-Meteo and wttr.in (no API key needed)
        **Historical Patterns**: Uses 30-year climate averages with variability
        """
    )
    use_historical = weather_source == "Historical Patterns"
    
    # Time horizon
    time_horizon = st.sidebar.slider(
        "Planning Horizon (months)",
        min_value=3, max_value=24, value=6, step=1,
        help="Planning horizon for optimization (3-24 months)"
    )
    
    # Objective weights
    st.sidebar.subheader("Objective Weights")
    power_weight = st.sidebar.slider("Power Generation", 0.1, 1.0, 0.4, 0.1)
    flood_weight = st.sidebar.slider("Flood Control", 0.1, 1.0, 0.3, 0.1)  
    water_weight = st.sidebar.slider("Water Security", 0.1, 1.0, 0.3, 0.1)
    
    # Normalize weights
    total_weight = power_weight + flood_weight + water_weight
    weights = {
        'power': power_weight / total_weight,
        'flood': flood_weight / total_weight,
        'water': water_weight / total_weight
    }
    
    # Algorithm parameters
    st.sidebar.subheader("Algorithm Settings")
    pop_size = st.sidebar.selectbox("Population Size", [25, 50, 100, 200], index=1)
    generations = st.sidebar.selectbox("Generations", [25, 50, 100, 200], index=1)
    
    # AI Interpretation Settings
    st.sidebar.subheader("🤖 AI Analysis Settings")
    enable_ai = st.sidebar.checkbox("Enable AI Interpretations", value=True, 
                                   help="Generate natural language explanations of results")
    interpretation_detail = st.sidebar.select_slider(
        "Analysis Detail Level",
        options=["Basic", "Standard", "Detailed"],
        value="Standard",
        help="Control the depth of AI analysis"
    )
    
    # Initial reservoir levels
    st.sidebar.subheader("Initial Reservoir Levels (%)")
    initial_levels = {}
    for dam_name in dams.cascade_order:
        if dam_name in dams.dams:
            initial_levels[dam_name] = st.sidebar.slider(
                f"{dams.dams[dam_name].name}",
                min_value=20, max_value=95, value=70, step=5,
                help=f"Initial level as % of capacity"
            )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("🌡️ System Status")
        
        # System status indicators
        status_container = st.container()
        with status_container:
            # Check API availability
            weather_status = "🟢 Online"
            try:
                test_data = weather_manager.fetch_open_meteo_forecast('khartoum', days=1)
                if not test_data:
                    weather_status = "🟡 Historical Mode"
            except:
                weather_status = "🟡 Historical Mode"
            
            st.metric("Weather Data", weather_status)
            st.metric("Optimization Engine", "🟢 Ready")
            st.metric("Total Cascade Capacity", f"{dams.get_cascade_capacity():,.0f} MCM")
            st.metric("Total Power Capacity", f"{dams.get_cascade_power():,.0f} MW")
    
    with col1:
        # Weather forecast display
        st.subheader("🌦️ Weather Forecast")
        
        with st.spinner("Loading weather data..."):
            try:
                weather_data = load_weather_data(days=14, use_historical=use_historical)
                
                if weather_data:
                    # Show data source info
                    if use_historical:
                        st.info("📊 Using historical climate patterns (30-year averages with daily variability)")
                    else:
                        st.success("🌐 Using free weather APIs (Open-Meteo, wttr.in)")
                    
                    # Create weather visualization
                    weather_df_list = []
                    for location, data_list in weather_data.items():
                        for data_point in data_list:
                            weather_df_list.append({
                                'Location': data_point.location,
                                'Date': data_point.date,
                                'Temperature (°C)': (data_point.temperature_min + data_point.temperature_max) / 2,
                                'Precipitation (mm)': data_point.precipitation,
                                'Humidity (%)': data_point.humidity,
                                'Evapotranspiration (mm/day)': data_point.evapotranspiration or 0
                            })
                    
                    if weather_df_list:
                        weather_df = pd.DataFrame(weather_df_list)
                        
                        # Weather plots
                        fig_weather = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=['Temperature', 'Precipitation', 'Humidity', 'Evapotranspiration'],
                            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                   [{"secondary_y": False}, {"secondary_y": False}]]
                        )
                        
                        for location in weather_df['Location'].unique():
                            location_data = weather_df[weather_df['Location'] == location]
                            
                            # Temperature
                            fig_weather.add_trace(
                                go.Scatter(x=location_data['Date'], y=location_data['Temperature (°C)'],
                                         name=f"{location} Temp", mode='lines'),
                                row=1, col=1
                            )
                            
                            # Precipitation
                            fig_weather.add_trace(
                                go.Bar(x=location_data['Date'], y=location_data['Precipitation (mm)'],
                                      name=f"{location} Precip"),
                                row=1, col=2
                            )
                            
                            # Humidity
                            fig_weather.add_trace(
                                go.Scatter(x=location_data['Date'], y=location_data['Humidity (%)'],
                                         name=f"{location} Humidity", mode='lines'),
                                row=2, col=1
                            )
                            
                            # Evapotranspiration
                            fig_weather.add_trace(
                                go.Scatter(x=location_data['Date'], y=location_data['Evapotranspiration (mm/day)'],
                                         name=f"{location} ET", mode='lines'),
                                row=2, col=2
                            )
                        
                        fig_weather.update_layout(height=500, showlegend=False)
                        st.plotly_chart(fig_weather, use_container_width=True)
                    else:
                        st.warning("No weather data available. Using synthetic data for optimization.")
                else:
                    st.warning("⚠️ Weather APIs unavailable. Using historical climate patterns for optimization.")
                    
            except Exception as e:
                st.error(f"Weather data error: {str(e)}")
                weather_data = {}
        
        # Historical Climate Statistics Section (when using historical mode)
        if use_historical:
            st.subheader("📈 Historical Climate Statistics")
            
            # Get historical analysis for a selected location
            hist_location = st.selectbox(
                "Select location for climate statistics",
                options=list(weather_manager.locations.keys()),
                format_func=lambda x: weather_manager.locations[x]['name']
            )
            
            if hist_location:
                hist_stats = weather_manager.get_historical_analysis(hist_location)
                
                # Create historical climate visualization
                fig_hist = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=['Temperature Range', 'Monthly Precipitation', 
                                  'Humidity Pattern', 'Evapotranspiration'],
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                          [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Temperature range
                fig_hist.add_trace(
                    go.Scatter(x=hist_stats['Month'], y=hist_stats['Avg_Temp_Min'],
                             name='Min Temp', mode='lines+markers', line=dict(color='lightblue')),
                    row=1, col=1
                )
                fig_hist.add_trace(
                    go.Scatter(x=hist_stats['Month'], y=hist_stats['Avg_Temp_Max'],
                             name='Max Temp', mode='lines+markers', line=dict(color='coral')),
                    row=1, col=1
                )
                
                # Precipitation
                fig_hist.add_trace(
                    go.Bar(x=hist_stats['Month'], y=hist_stats['Avg_Precipitation'],
                          name='Precipitation', marker_color='blue'),
                    row=1, col=2
                )
                
                # Humidity
                fig_hist.add_trace(
                    go.Scatter(x=hist_stats['Month'], y=hist_stats['Avg_Humidity'],
                             name='Humidity', mode='lines+markers', line=dict(color='green')),
                    row=2, col=1
                )
                
                # Evapotranspiration
                fig_hist.add_trace(
                    go.Scatter(x=hist_stats['Month'], y=hist_stats['Est_Evapotranspiration'],
                             name='ET0', mode='lines+markers', line=dict(color='orange')),
                    row=2, col=2
                )
                
                fig_hist.update_xaxes(title_text="Month", row=2, col=1)
                fig_hist.update_xaxes(title_text="Month", row=2, col=2)
                fig_hist.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
                fig_hist.update_yaxes(title_text="Precipitation (mm)", row=1, col=2)
                fig_hist.update_yaxes(title_text="Humidity (%)", row=2, col=1)
                fig_hist.update_yaxes(title_text="ET0 (mm/day)", row=2, col=2)
                
                fig_hist.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Display statistics table
                st.dataframe(hist_stats, use_container_width=True)
    
    # Optimization section
    st.markdown("---")
    st.subheader("🚀 Optimization Results")
    
    # Run optimization button
    if st.button("🎯 Run Optimization", type="primary", use_container_width=True):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Initializing optimization...")
            progress_bar.progress(10)
            
            # Create optimizer
            optimizer = WaterResourceOptimizer(dams, weather_data or {})
            
            status_text.text("Running multi-objective optimization...")
            progress_bar.progress(30)
            
            # Run optimization
            result = optimizer.optimize_cascade(
                time_horizon=time_horizon,
                population_size=pop_size,
                generations=generations,
                weights=weights
            )
            
            progress_bar.progress(100)
            status_text.text("Optimization completed!")
            
            if result.feasible:
                st.success("✅ Optimization completed successfully!")
                
                # Display objective values
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Power Generation", f"{result.objective_values['total_power']:.0f} MW")
                with col2:
                    st.metric("Flood Risk Score", f"{result.objective_values['flood_risk']:.2f}")
                with col3:
                    st.metric("Water Security Score", f"{result.objective_values['water_security']:.2f}")
                
                # AI Analysis (if enabled)
                if enable_ai:
                    # Initialize AI Interpreter
                    from ai_interpreter import OptimizationInterpreter
                    interpreter = OptimizationInterpreter(dams)
                    
                    # Generate AI interpretation
                    interpretation = interpreter.interpret_optimization_results(
                        result, weights, time_horizon
                    )
                    
                    # Display AI Summary
                    st.markdown("---")
                    st.subheader("🤖 AI Analysis & Insights")
                    
                    # Executive Summary
                    with st.expander("📋 Executive Summary", expanded=True):
                        st.markdown(interpretation.summary)
                    
                    # Key Insights
                    if interpretation.key_insights:
                        with st.expander("💡 Key Insights", expanded=True):
                            for insight in interpretation.key_insights:
                                st.markdown(f"• {insight}")
                    
                    # Warnings
                    if interpretation.warnings:
                        with st.expander("⚠️ Warnings & Alerts", expanded=interpretation_detail == "Detailed"):
                            for warning in interpretation.warnings:
                                st.markdown(f"• {warning}")
                    
                    # Recommendations
                    if interpretation.recommendations:
                        with st.expander("🎯 Recommendations", expanded=interpretation_detail != "Basic"):
                            for rec in interpretation.recommendations:
                                st.markdown(f"• {rec}")
                    
                    # Technical Notes (only for detailed mode)
                    if interpretation.technical_notes and interpretation_detail == "Detailed":
                        with st.expander("🔧 Technical Notes", expanded=False):
                            st.markdown(interpretation.technical_notes)
                
                # Results visualization
                st.subheader("📊 Optimization Results" + (" with AI Interpretation" if enable_ai else ""))
                
                if enable_ai:
                    # Charts with AI interpretations
                    # Reservoir levels chart with AI interpretation
                    col_chart, col_explain = st.columns([2, 1])
                    
                    with col_chart:
                        fig_levels = go.Figure()
                        
                        time_range = list(range(1, time_horizon + 1))
                        for dam_name, levels in result.reservoir_levels.items():
                            fig_levels.add_trace(go.Scatter(
                                x=time_range,
                                y=levels,
                                mode='lines+markers',
                                name=f"{dam_name} Level (MCM)",
                                line=dict(width=3)
                            ))
                        
                        fig_levels.update_layout(
                            title="Optimal Reservoir Levels Over Time",
                            xaxis_title="Time Period (months)",
                            yaxis_title="Storage Level (MCM)",
                            height=400,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_levels, use_container_width=True)
                    
                    with col_explain:
                        st.markdown("### 🤖 AI Interpretation")
                        levels_interpretation = interpreter.interpret_reservoir_levels_chart(result, time_horizon)
                        st.markdown(levels_interpretation)
                    
                    # Power generation chart with AI interpretation
                    col_chart2, col_explain2 = st.columns([2, 1])
                    
                    with col_chart2:
                        fig_power = go.Figure()
                        
                        for dam_name, power in result.power_generation.items():
                            fig_power.add_trace(go.Scatter(
                                x=time_range,
                                y=power,
                                mode='lines+markers',
                                name=f"{dam_name} Power (MW)",
                                fill='tonexty' if dam_name != list(result.power_generation.keys())[0] else None
                            ))
                        
                        fig_power.update_layout(
                            title="Optimal Power Generation Schedule",
                            xaxis_title="Time Period (months)",
                            yaxis_title="Power Generation (MW)",
                            height=400,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_power, use_container_width=True)
                    
                    with col_explain2:
                        st.markdown("### 🤖 AI Interpretation")
                        power_interpretation = interpreter.interpret_power_generation_chart(result, weights)
                        st.markdown(power_interpretation)
                    
                    # Water releases chart with AI interpretation
                    col_chart3, col_explain3 = st.columns([2, 1])
                    
                    with col_chart3:
                        fig_releases = go.Figure()
                        
                        for dam_name, releases in result.water_releases.items():
                            fig_releases.add_trace(go.Scatter(
                                x=time_range,
                                y=releases,
                                mode='lines+markers',
                                name=f"{dam_name} Release (m³/s)"
                            ))
                        
                        fig_releases.update_layout(
                            title="Optimal Water Release Schedule",
                            xaxis_title="Time Period (months)",
                            yaxis_title="Water Release (m³/s)",
                            height=400,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_releases, use_container_width=True)
                    
                    with col_explain3:
                        st.markdown("### 🤖 AI Interpretation")
                        releases_interpretation = interpreter.interpret_water_releases_chart(result)
                        st.markdown(releases_interpretation)
                else:
                    # Standard charts without AI interpretation
                    # Reservoir levels chart
                    fig_levels = go.Figure()
                    
                    time_range = list(range(1, time_horizon + 1))
                    for dam_name, levels in result.reservoir_levels.items():
                        fig_levels.add_trace(go.Scatter(
                            x=time_range,
                            y=levels,
                            mode='lines+markers',
                            name=f"{dam_name} Level (MCM)",
                            line=dict(width=3)
                        ))
                    
                    fig_levels.update_layout(
                        title="Optimal Reservoir Levels Over Time",
                        xaxis_title="Time Period (months)",
                        yaxis_title="Storage Level (MCM)",
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_levels, use_container_width=True)
                    
                    # Power generation chart
                    fig_power = go.Figure()
                    
                    for dam_name, power in result.power_generation.items():
                        fig_power.add_trace(go.Scatter(
                            x=time_range,
                            y=power,
                            mode='lines+markers',
                            name=f"{dam_name} Power (MW)",
                            fill='tonexty' if dam_name != list(result.power_generation.keys())[0] else None
                        ))
                    
                    fig_power.update_layout(
                        title="Optimal Power Generation Schedule",
                        xaxis_title="Time Period (months)",
                        yaxis_title="Power Generation (MW)",
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_power, use_container_width=True)
                    
                    # Water releases chart
                    fig_releases = go.Figure()
                    
                    for dam_name, releases in result.water_releases.items():
                        fig_releases.add_trace(go.Scatter(
                            x=time_range,
                            y=releases,
                            mode='lines+markers',
                            name=f"{dam_name} Release (m³/s)"
                        ))
                    
                    fig_releases.update_layout(
                        title="Optimal Water Release Schedule",
                        xaxis_title="Time Period (months)",
                        yaxis_title="Water Release (m³/s)",
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_releases, use_container_width=True)
                
                # Results summary table
                st.subheader("📋 Detailed Results Summary")
                
                # Create summary dataframe
                summary_data = []
                for i, dam_name in enumerate(result.reservoir_levels.keys()):
                    dam_spec = dams.get_dam_specs(dam_name)
                    avg_level = np.mean(result.reservoir_levels[dam_name])
                    avg_power = np.mean(result.power_generation[dam_name])
                    avg_release = np.mean(result.water_releases[dam_name])
                    
                    summary_data.append({
                        'Dam': dam_spec.name,
                        'Capacity (MCM)': f"{dam_spec.max_capacity:,.0f}",
                        'Avg Level (MCM)': f"{avg_level:,.0f}",
                        'Level Utilization (%)': f"{(avg_level/dam_spec.max_capacity)*100:.1f}",
                        'Avg Power (MW)': f"{avg_power:.0f}",
                        'Power Utilization (%)': f"{(avg_power/dam_spec.turbine_capacity)*100:.1f}",
                        'Avg Release (m³/s)': f"{avg_release:.0f}",
                        'Min Env Flow (m³/s)': f"{dam_spec.min_env_flow:.0f}",
                        'Env Flow Met': "✅" if avg_release >= dam_spec.min_env_flow else "❌"
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Export results
                if st.button("📥 Export Results to CSV"):
                    # Prepare data for export
                    export_data = []
                    for i in range(time_horizon):
                        row = {'Time_Period': i + 1}
                        for dam_name in result.reservoir_levels.keys():
                            row[f"{dam_name}_Level_MCM"] = result.reservoir_levels[dam_name][i]
                            row[f"{dam_name}_Power_MW"] = result.power_generation[dam_name][i]
                            row[f"{dam_name}_Release_m3s"] = result.water_releases[dam_name][i]
                        export_data.append(row)
                    
                    export_df = pd.DataFrame(export_data)
                    csv = export_df.to_csv(index=False)
                    
                    st.download_button(
                        label="Download Results CSV",
                        data=csv,
                        file_name=f"nile_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                
            else:
                st.error(f"❌ Optimization failed: {result.solver_status}")
                st.info("Try adjusting the parameters or check the system status.")
                
        except Exception as e:
            st.error(f"❌ Optimization error: {str(e)}")
            logger.exception("Optimization failed")
        
        finally:
            progress_bar.empty()
            status_text.empty()
    
    # Dam specifications section
    st.markdown("---")
    st.subheader("🏗️ Dam Technical Specifications")
    
    # Dam selection
    selected_dam = st.selectbox(
        "Select Dam for Detailed Specifications",
        options=list(dams.dams.keys()),
        format_func=lambda x: dams.get_dam_specs(x).name
    )
    
    if selected_dam:
        dam_spec = dams.get_dam_specs(selected_dam)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Storage Specifications:**")
            st.write(f"• Maximum Capacity: {dam_spec.max_capacity:,.0f} MCM")
            st.write(f"• Minimum Capacity: {dam_spec.min_capacity:,.0f} MCM")
            st.write(f"• Live Storage: {dam_spec.live_capacity:,.0f} MCM")
            st.write(f"• Storage Efficiency: {(dam_spec.live_capacity/dam_spec.max_capacity)*100:.1f}%")
            
            st.write("**Power Generation:**")
            st.write(f"• Installed Capacity: {dam_spec.turbine_capacity:,.0f} MW")
            st.write(f"• Turbine Efficiency: {dam_spec.efficiency*100:.0f}%")
            st.write(f"• Maximum Head: {dam_spec.max_head:.0f} m")
            st.write(f"• Minimum Head: {dam_spec.min_head:.0f} m")
        
        with col2:
            st.write("**Operational Constraints:**")
            st.write(f"• Min Environmental Flow: {dam_spec.min_env_flow:,.0f} m³/s")
            st.write(f"• Spillway Capacity: {dam_spec.spillway_capacity:,.0f} m³/s")
            
            # Create capacity visualization
            fig_capacity = go.Figure(go.Bar(
                x=['Dead Storage', 'Live Storage'],
                y=[dam_spec.min_capacity, dam_spec.live_capacity],
                marker_color=['lightcoral', 'lightblue']
            ))
            
            fig_capacity.update_layout(
                title=f"{dam_spec.name} - Storage Breakdown",
                yaxis_title="Capacity (MCM)",
                height=300
            )
            
            st.plotly_chart(fig_capacity, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    **About this tool:**  
    This comprehensive water resource management tool integrates free weather forecasts and historical climate data, 
    detailed dam specifications, and multi-objective optimization to support decision-making 
    for the Nile River cascade system. Built with Python, Streamlit, and HiGHS optimization.
    
    **Data Sources:** 
    - **Weather**: Open-Meteo (free, no key needed), wttr.in (free), 30-year historical climate averages
    - **Dam Specifications**: Official government reports, technical literature, engineering documents
    - **Optimization**: HiGHS solver with NSGA-II multi-objective genetic algorithm
    """)

if __name__ == "__main__":
    main()