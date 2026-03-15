"""
Page modules for the Novelty Analysis App
"""
from pages.data_loading import page_data_loading
from pages.embedding_processing import page_embedding_processing
from pages.filters import page_filters
from pages.clustering import page_clustering
from pages.gap_analysis import page_gap_analysis
from pages.gap_regions import page_gap_regions
from pages.llm_analysis import page_llm_analysis
from pages.agent_console import page_agent_console
from pages.database_explorer import page_database_explorer
from pages.export import page_export

__all__ = [
    'page_data_loading',
    'page_embedding_processing',
    'page_filters',
    'page_clustering',
    'page_gap_analysis',
    'page_gap_regions',
    'page_llm_analysis',
    'page_agent_console',
    'page_database_explorer',
    'page_export'
]
