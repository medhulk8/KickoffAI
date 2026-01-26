"""
Data module for ASIL project.

Provides data loading, baseline models, and advanced statistics.

Components:
- AdvancedStatsCalculator: Calculate advanced team stats from match data
"""

from src.data.advanced_stats import AdvancedStatsCalculator

__all__ = ["AdvancedStatsCalculator"]
