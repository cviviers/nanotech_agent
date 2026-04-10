"""
Page modules for the Novelty Analysis App
"""
from app_pages.data_loading import page_data_loading
from app_pages.embedding_processing import page_embedding_processing
from app_pages.filters import page_filters
from app_pages.clustering import page_clustering
from app_pages.gap_analysis import page_gap_analysis
from app_pages.gap_regions import page_gap_regions
from app_pages.llm_analysis import page_llm_analysis
from app_pages.agent_console import page_agent_console
from app_pages.database_explorer import page_database_explorer
from app_pages.export import page_export

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
