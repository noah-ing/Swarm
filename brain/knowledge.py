"""
Knowledge Retrieval Component
============================

Provides contextual insights and recommendations from historical data.
Extracted from CognitiveArchitecture for focused knowledge management.
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from brain.database import CognitiveDatabase


@dataclass
class KnowledgeInsight:
    """Structured insight from historical data."""
    insight: str
    source: str  # 'reflection', 'skill', 'failure_pattern'
    relevance_score: float
    metadata: Dict[str, Any]


@dataclass
class ModelRecommendation:
    """Model recommendation with reasoning."""
    model: str
    success_rate: float
    avg_tokens: int
    efficiency_score: float
    reasoning: str


class KnowledgeRetriever:
    """
    Retrieves and synthesizes knowledge from cognitive history.
    
    Extracted from CognitiveArchitecture to provide focused knowledge retrieval:
    - Task-relevant insights from past reflections
    - Failure pattern warnings
    - Model recommendations based on performance
    - Skill suggestions from learned patterns
    """
    
    def __init__(self, db: CognitiveDatabase):
        """Initialize with database connection."""
        self.db = db
    
    def get_insights_for_task(self, task: str) -> List[str]:
        """
        Get relevant insights from past reflections for a task.
        
        Extracted from CognitiveArchitecture.get_insights_for_task()
        
        Args:
            task: The task description
            
        Returns:
            List of relevant insights (max 5)
        """
        return self.db.get_insights_for_task(task)
    
    def get_failure_warnings(self, task: str) -> List[str]:
        """
        Get warnings based on past failure patterns.
        
        Extracted from CognitiveArchitecture.get_failure_warnings()
        
        Args:
            task: The task description
            
        Returns:
            List of warning messages (max 3)
        """
        return self.db.get_failure_warnings_for_task(task)
    
    def get_recommended_model(self, task: str, default: str = "sonnet") -> str:
        """
        Recommend a model based on past performance with similar tasks.
        
        Extracted from CognitiveArchitecture.get_recommended_model()
        
        Args:
            task: The task description (currently unused but reserved for future enhancement)
            default: Default model to use if no data available
            
        Returns:
            Recommended model name
        """
        return self.db.get_recommended_model(default)
    
    def get_contextual_knowledge(self, task: str, include_warnings: bool = True) -> Dict[str, Any]:
        """
        Get comprehensive contextual knowledge for a task.
        
        This is an enhanced method that combines multiple knowledge sources
        to provide a complete picture of relevant historical knowledge.
        
        Args:
            task: The task description
            include_warnings: Whether to include failure warnings
            
        Returns:
            Dictionary containing:
            - insights: List of relevant insights
            - warnings: List of failure warnings (if requested)
            - recommended_model: Best model for this type of task
            - similar_tasks_exist: Whether similar tasks have been done before
            - confidence_indicators: Factors affecting confidence in the task
        """
        knowledge = {
            'insights': self.get_insights_for_task(task),
            'recommended_model': self.get_recommended_model(task),
            'similar_tasks_exist': self.db.has_similar_past_solution(task),
        }
        
        if include_warnings:
            knowledge['warnings'] = self.get_failure_warnings(task)
        
        # Add confidence indicators based on historical data
        confidence_indicators = []
        
        if knowledge['similar_tasks_exist']:
            confidence_indicators.append("Similar tasks completed successfully before")
        
        if knowledge.get('warnings'):
            confidence_indicators.append(f"{len(knowledge['warnings'])} failure patterns detected")
        
        if len(knowledge['insights']) >= 3:
            confidence_indicators.append("Rich historical insights available")
        
        knowledge['confidence_indicators'] = confidence_indicators
        
        return knowledge
    
    def get_structured_insights(self, task: str, limit: int = 10) -> List[KnowledgeInsight]:
        """
        Get structured insights with metadata and relevance scores.
        
        Enhanced version that provides more detailed insight information.
        
        Args:
            task: The task description
            limit: Maximum number of insights to return
            
        Returns:
            List of KnowledgeInsight objects with relevance scoring
        """
        task_keywords = set(task.lower().split())
        structured_insights = []
        
        # Get insights from reflections
        for reflection_data in self.db.get_recent_insights(limit=100):
            for insight in reflection_data.get('insights', [])[:2]:
                insight_keywords = set(insight.lower().split())
                relevance = len(task_keywords & insight_keywords) / max(len(task_keywords), 1)
                
                if relevance > 0.1:  # Minimum relevance threshold
                    structured_insights.append(KnowledgeInsight(
                        insight=insight,
                        source='reflection',
                        relevance_score=relevance,
                        metadata={
                            'what_worked': reflection_data.get('what_worked', []),
                            'what_failed': reflection_data.get('what_failed', [])
                        }
                    ))
        
        # Sort by relevance and limit
        structured_insights.sort(key=lambda x: x.relevance_score, reverse=True)
        return structured_insights[:limit]
    
    def get_model_recommendations(self, task: str = None) -> List[ModelRecommendation]:
        """
        Get detailed model recommendations with performance metrics.
        
        Enhanced version that provides reasoning for recommendations.
        
        Args:
            task: Optional task description for task-specific recommendations
            
        Returns:
            List of ModelRecommendation objects sorted by overall score
        """
        recommendations = []
        
        for model_data in self.db.get_model_performance_stats():
            model_name = model_data['model']
            total = model_data['total']
            successes = model_data['successes']
            avg_tokens = model_data['avg_tokens']
            
            if total > 0:
                success_rate = successes / total
                efficiency = 1.0 / (1 + avg_tokens / 10000)
                overall_score = success_rate * 0.7 + efficiency * 0.3
                
                # Generate reasoning
                reasoning_parts = []
                if success_rate >= 0.8:
                    reasoning_parts.append(f"High success rate ({success_rate:.1%})")
                elif success_rate >= 0.6:
                    reasoning_parts.append(f"Good success rate ({success_rate:.1%})")
                else:
                    reasoning_parts.append(f"Moderate success rate ({success_rate:.1%})")
                
                if avg_tokens < 5000:
                    reasoning_parts.append("Very efficient token usage")
                elif avg_tokens < 15000:
                    reasoning_parts.append("Efficient token usage")
                else:
                    reasoning_parts.append("High token usage")
                
                reasoning_parts.append(f"{total} tasks completed")
                
                recommendations.append(ModelRecommendation(
                    model=model_name,
                    success_rate=success_rate,
                    avg_tokens=int(avg_tokens),
                    efficiency_score=efficiency,
                    reasoning="; ".join(reasoning_parts)
                ))
        
        # Sort by overall score (success_rate * 0.7 + efficiency * 0.3)
        recommendations.sort(
            key=lambda x: x.success_rate * 0.7 + x.efficiency_score * 0.3,
            reverse=True
        )
        
        return recommendations
    
    def get_skill_suggestions(self, task: str) -> List[Dict[str, Any]]:
        """
        Get relevant skills that might help with the task.
        
        Args:
            task: The task description
            
        Returns:
            List of relevant skills with their patterns and success counts
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        # Search for skills with patterns similar to the task
        task_keywords = set(task.lower().split())
        
        cursor.execute("""
            SELECT name, trigger_pattern, solution_template, success_count, fail_count
            FROM skills
            WHERE success_count > fail_count
            ORDER BY success_count DESC
            LIMIT 20
        """)
        
        relevant_skills = []
        for row in cursor.fetchall():
            pattern_keywords = set(row[1].lower().split())
            overlap = len(task_keywords & pattern_keywords) / max(len(pattern_keywords), 1)
            
            if overlap > 0.3:  # Relevance threshold
                try:
                    solution_template = json.loads(row[2]) if row[2] else []
                except json.JSONDecodeError:
                    solution_template = []
                
                relevant_skills.append({
                    'name': row[0],
                    'pattern': row[1],
                    'solution_template': solution_template,
                    'success_count': row[3],
                    'fail_count': row[4],
                    'relevance_score': overlap
                })
        
        # Sort by relevance score * success rate
        relevant_skills.sort(
            key=lambda x: x['relevance_score'] * (x['success_count'] / max(x['success_count'] + x['fail_count'], 1)),
            reverse=True
        )
        
        return relevant_skills[:5]  # Top 5 skills
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """
        Get a summary of what the system has learned.
        
        Returns:
            Dictionary containing learning statistics and patterns
        """
        stats = self.db.get_system_stats()
        
        # Calculate derived metrics
        success_rate = 0.0
        if stats['total_reflections'] > 0:
            success_rate = stats['successful_reflections'] / stats['total_reflections']
        
        # Get top performing models
        model_recommendations = self.get_model_recommendations()
        top_models = [
            {
                'model': rec.model,
                'success_rate': rec.success_rate,
                'reasoning': rec.reasoning
            }
            for rec in model_recommendations[:3]
        ]
        
        # Get common failure patterns
        failure_patterns = self.db.get_failure_patterns(min_occurrences=3, limit=5)
        
        return {
            'total_tasks_completed': stats['total_reflections'],
            'overall_success_rate': success_rate,
            'skills_learned': stats['skills_learned'],
            'failure_patterns_identified': stats['failure_patterns_tracked'],
            'average_confidence': stats['avg_confidence'],
            'top_performing_models': top_models,
            'common_failure_causes': [fp['cause'] for fp in failure_patterns],
            'learning_insights': self._generate_learning_insights(stats, success_rate)
        }
    
    def _generate_learning_insights(self, stats: Dict[str, Any], success_rate: float) -> List[str]:
        """Generate high-level insights about the system's learning."""
        insights = []
        
        if success_rate >= 0.8:
            insights.append("System shows high proficiency with most tasks")
        elif success_rate >= 0.6:
            insights.append("System demonstrates good overall performance")
        else:
            insights.append("System is still learning and improving")
        
        if stats['skills_learned'] > 10:
            insights.append(f"Developed {stats['skills_learned']} reusable skills")
        elif stats['skills_learned'] > 0:
            insights.append(f"Beginning to extract reusable patterns ({stats['skills_learned']} skills)")
        
        if stats['failure_patterns_tracked'] > 5:
            insights.append(f"Actively learning from {stats['failure_patterns_tracked']} failure patterns")
        
        if stats['avg_confidence'] >= 0.7:
            insights.append("High confidence in task analysis and reflection")
        elif stats['avg_confidence'] >= 0.5:
            insights.append("Moderate confidence levels indicate balanced self-assessment")
        
        return insights