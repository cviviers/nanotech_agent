"""
Configuration and constants for the Novelty Analysis App
"""

# ============================================================================
# DOMAIN-SPECIFIC ENTITY HINTS
# ============================================================================

MATERIAL_HINTS = [
    'liposome','plga','gold','agnp','au','iron oxide','magnetite','silica',
    'mesoporous','graphene','go','peg','chitosan','albumin','micelle',
    'dendrimer','hydrogel','quantum dot','nanotube','nanoemulsion'
]

LIGAND_HINTS = [
    'rgd','folate','transferrin','aptamer','peptide','antibody','egf',
    'her2','mannose','galactose','hyaluronic'
]

DISEASE_HINTS = [
    'cancer','glioblastoma','breast','lung','pancreatic','pancreatic cancer',
    'prostate','melanoma','liver','ovarian','colorectal','colorectal cancer',
    'infection','inflammation','chronic inflammation','chronic inflammatory disease',
    'alzheimer','alzheimer\'s disease','neurodegenerative','neurodegenerative disease',
    'inflammatory bowel disease','ibd',
    'rheumatoid arthritis','autoimmune','autoimmunity'
]

DELIVERY_HINTS = [
    'intravenous','iv','oral','oral delivery','intratumoral','inhalation','topical','intranasal',
    'systemic','systemic delivery',
    'local','local delivery','local effects',
    'sustained release','local sustained release',
    'brain delivery',
    'blood brain barrier','barrier passage',
    'barrier penetration','barrier disruption'
]

MODEL_HINTS = [
    'in vitro','in vivo','mouse','murine','rat','xenograft','clinical','phase'
]

# ============================================================================
# APP CONFIGURATION
# ============================================================================

APP_TITLE = "Novelty Analysis App"
APP_ICON = "🔬"
PAGE_LAYOUT = "wide"

# Default configuration
DEFAULT_CONFIG = {
    'data_path': 'data/all_papers.json',
    'embedding_cols': ['qwen', 'bert'],
    'primary_embedding': 'qwen',
    'data_dir': 'data',
    'random_seed': 42
}

# Page navigation
PAGES = [
    "📊 Data & Config",
    "🧬 Embeddings",
    "🎯 Filters",
    "🔬 Clustering",
    "🔍 Gap Analysis",
    "🌉 Gap Regions",
    "🤖 LLM Analysis",
    "📚 Database Explorer",
    "💾 Export"
]
