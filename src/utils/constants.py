"""
Centralized path definitions and global constants for the project.
Defines directory structures and standardized suffixes for Knowledge Graph releases.
"""

from pathlib import Path

# paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "config"
PROMPT_DIR = PROJECT_ROOT / "prompts"
ARTICLE_BUILDER_DIR = DATA_DIR / "runs" / "article_builder"
ABSTRACT_AUGMENTER_DIR = DATA_DIR  / "runs" / "abstract_augmenter"
RELEASES_DIR = DATA_DIR / "releases"

# flags
COMPLETION_FLAG = "_RUN_COMPLETED.SUCCESS"

# suffixes
SUFFIX_BASE_KG = "base"
SUFFIX_ARTICLE_KG = "articles"
SUFFIX_AUGMENTED_KG = "augmented"
SUFFIX_PARTIAL_AUGMENTED_KG = "partial_augmented"
SUFFIX_NORMALIZED_KG = "normalized"
SUFFIX_REFINED_KG = "refined"
SUFFIX_FINAL_KG = "oc_resistance_kg"