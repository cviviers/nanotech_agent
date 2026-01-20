"""
Utility functions for exporting plotly figures to various formats
"""
import streamlit as st
from datetime import datetime
import io
import plotly.graph_objects as go

def add_figure_export_button(fig, filename_prefix: str, key: str = None):
    """
    Add an SVG export button below a plotly figure
    
    Args:
        fig: Plotly figure object
        filename_prefix: Prefix for the exported filename
        key: Unique key for the button (required if multiple buttons on same page)
    """
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_filename = f"{filename_prefix}_{timestamp}.svg"
    
    # Create a copy of the figure to modify for export
    fig_export = go.Figure(fig)
    
    # Configure for square export with Arial font, no axes or title
    fig_export.update_layout(
        width=1000,
        height=1000,
        font=dict(family="Arial, sans-serif"),
        margin=dict(l=40, r=40, t=40, b=40),
        title=None,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    
    # Export figure to SVG using kaleido
    svg_bytes = fig_export.to_image(format="svg", width=1000, height=1000)
    
    # Create download button
    st.download_button(
        label="📥 Export as SVG",
        data=svg_bytes,
        file_name=default_filename,
        mime="image/svg+xml",
        key=key,
        use_container_width=False
    )


def display_figure_with_export(fig, filename_prefix: str, key: str = None, use_container_width: bool = True):
    """
    Display a plotly figure with an SVG export button
    
    Args:
        fig: Plotly figure object
        filename_prefix: Prefix for the exported filename
        key: Unique key for the button (required if multiple buttons on same page)
        use_container_width: Whether to use full container width for the chart
    """
    st.plotly_chart(fig, use_container_width=use_container_width)
    add_figure_export_button(fig, filename_prefix, key=key)
