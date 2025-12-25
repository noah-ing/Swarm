"""
Cognitive Architecture Package
==============================

This package contains the refactored cognitive architecture with
separate components for better modularity and maintainability.

Components:
- CognitiveDatabase: Centralized database management
- UncertaintyAssessor: Task complexity and risk analysis
- StrategySelector: Strategy selection and management
- ReflectionEngine: Post-task reflection and learning
- SkillExtractor: Pattern recognition and skill creation
- KnowledgeRetriever: Historical insights and recommendations
- CognitiveArchitecture: Main orchestrator maintaining backward compatibility

The refactored architecture maintains 100% backward compatibility with
the existing brain.py API while providing better separation of concerns.
"""

from typing import Dict, Union, Any, List, Optional

def format_stats(stats: Dict[str, Union[int, float, str, Any]], separator: str = ': ',
                 line_separator: str = '\n') -> str:
    """
    Format a dictionary of statistics into a readable string.

    Args:
        stats (Dict[str, Union[int, float, str, Any]]): Dictionary of statistics
        separator (str, optional): Separator between key and value. Defaults to ': '.
        line_separator (str, optional): Separator between stat lines. Defaults to '\n'.

    Returns:
        str: Formatted statistics string
    """
    formatted_lines = [
        f"{key}{separator}{value}"
        for key, value in stats.items()
        if value is not None
    ]
    return line_separator.join(formatted_lines)

def get_stats_summary(stats: Dict[str, Union[int, float, str, Any]], 
                      aggregate_func: Optional[callable] = None, 
                      filter_func: Optional[callable] = None) -> Dict[str, Union[int, float, str, Any]]:
    """
    Generate a summary of statistics based on optional aggregation and filtering functions.

    Args:
        stats (Dict[str, Union[int, float, str, Any]]): Dictionary of input statistics
        aggregate_func (Optional[callable], optional): Function to aggregate numeric stats. 
                                                      Defaults to None (no aggregation).
        filter_func (Optional[callable], optional): Function to filter stats. 
                                                   Defaults to None (no filtering).

    Returns:
        Dict[str, Union[int, float, str, Any]]: A summary of statistics
    """
    # Apply filtering if a filter function is provided
    filtered_stats = {k: v for k, v in stats.items() if filter_func is None or filter_func(k, v)}
    
    # Compute summary
    summary = {
        'total_items': len(filtered_stats),
        'keys': list(filtered_stats.keys())
    }
    
    # Compute aggregate statistics for numeric values
    numeric_stats = {k: v for k, v in filtered_stats.items() if isinstance(v, (int, float))}
    
    if numeric_stats and aggregate_func:
        summary['aggregate'] = aggregate_func(list(numeric_stats.values()))
    
    return summary

def calculate_risk_score(metrics: List[float], weights: Optional[List[float]] = None) -> float:
    """
    Calculate a weighted risk score based on input metrics.

    Args:
        metrics (List[float]): A list of risk metrics, typically ranging from 0.0 to 1.0
        weights (Optional[List[float]], optional): Corresponding weights for each metric.
                                                   Defaults to equal weights if not provided.

    Returns:
        float: A weighted risk score between 0.0 and 1.0
    """
    if not metrics:
        return 0.0

    # Use equal weights if no weights are provided
    if weights is None:
        weights = [1.0 / len(metrics)] * len(metrics)

    # Validate inputs
    if len(metrics) != len(weights):
        raise ValueError("Number of metrics must match number of weights")

    # Calculate weighted risk score
    weighted_score = sum(metric * weight for metric, weight in zip(metrics, weights))
    normalized_score = min(max(weighted_score, 0.0), 1.0)  # Clamp between 0 and 1

    return normalized_score

# Import all components
from brain.database import CognitiveDatabase, Reflection
from brain.uncertainty import UncertaintyAssessor, Uncertainty
from brain.strategy import StrategySelector, Strategy
from brain.reflection import ReflectionEngine
from brain.skill_extractor import SkillExtractor
from brain.knowledge import KnowledgeRetriever

# Import main architecture and singleton
from brain.architecture import CognitiveArchitecture, get_brain

# Export key items for easy access
__all__ = [
    'CognitiveArchitecture',
    'get_brain',
    'Uncertainty',
    'Strategy',
    'Reflection',
    'CognitiveDatabase',
    'UncertaintyAssessor',
    'StrategySelector',
    'ReflectionEngine',
    'SkillExtractor',
    'KnowledgeRetriever',
    'format_stats',
    'get_stats_summary',
    'calculate_risk_score'
]

# Version info
__version__ = '2.0.0'