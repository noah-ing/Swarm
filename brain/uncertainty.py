"""
Uncertainty Assessor Component
==============================

Analyzes task complexity and assesses uncertainty/risk factors.
Extracted from CognitiveArchitecture for better modularity.
"""

import re
from dataclasses import dataclass
from typing import List

from brain.database import CognitiveDatabase


@dataclass
class Uncertainty:
    """Uncertainty assessment for a task."""
    level: str  # low, medium, high, critical
    score: float  # 0.0 to 1.0
    reasons: list[str]
    recommended_action: str  # proceed, cautious, escalate, ask_user
    fallback_strategies: list[str]


class UncertaintyAssessor:
    """
    Analyzes task complexity and assesses uncertainty/risk factors.
    
    Extracted from CognitiveArchitecture to provide focused uncertainty analysis
    using historical failure patterns and heuristic analysis.
    """
    
    def __init__(self, db: CognitiveDatabase):
        """Initialize with database for historical pattern analysis."""
        self.db = db
    
    def assess_uncertainty(self, task: str, context: str = "") -> Uncertainty:
        """
        Assess uncertainty level for a task.
        
        This is the core method extracted from CognitiveArchitecture that analyzes:
        - Vague language patterns
        - Task complexity indicators  
        - Risk keywords
        - Historical failure patterns
        
        Returns guidance on how to proceed based on confidence.
        """
        task_lower = task.lower()
        
        # Factors that increase uncertainty
        uncertainty_factors = []
        score = 0.0
        
        # Vague language
        vague_words = ["somehow", "maybe", "might", "something", "stuff", "thing", "whatever", "etc"]
        if any(w in task_lower for w in vague_words):
            uncertainty_factors.append("Task contains vague language")
            score += 0.2
        
        # Negative/unclear requirements
        if "don't know" in task_lower or "not sure" in task_lower:
            uncertainty_factors.append("Unclear requirements expressed")
            score += 0.3
        
        # Very short task (ambiguous)
        if len(task.split()) < 5:
            uncertainty_factors.append("Task description very brief")
            score += 0.15
        
        # Very long task (complex)
        if len(task.split()) > 100:
            uncertainty_factors.append("Task description very long/complex")
            score += 0.2
        
        # Novel domain indicators
        novel_indicators = ["never done", "first time", "new to", "unfamiliar"]
        if any(n in task_lower for n in novel_indicators):
            uncertainty_factors.append("Novel/unfamiliar domain indicated")
            score += 0.25
        
        # Complex architectural keywords that indicate high complexity
        complex_architectural = [
            "entire system", "whole system", "refactor entire", "complete rewrite",
            "system-wide", "architecture", "refactor", "redesign", "overhaul"
        ]
        
        architectural_matches = [term for term in complex_architectural if term in task_lower]
        if architectural_matches:
            uncertainty_factors.append(f"Complex architectural change indicated: {', '.join(architectural_matches)}")
            score += 0.3  # Architectural changes are inherently complex
        
        # High complexity individual keywords
        high_complexity_keywords = [
            "oauth2", "authentication", "security", "encryption", "authorization",
            "distributed", "microservices", "scalability", "performance", 
            "concurrent", "parallel", "optimization"
        ]
        
        complexity_matches = [term for term in high_complexity_keywords if term in task_lower]
        if complexity_matches:
            uncertainty_factors.append(f"High complexity domain detected: {', '.join(complexity_matches)}")
            score += 0.2 + (0.1 * min(len(complexity_matches) - 1, 2))  # Additional complexity for multiple domains
        
        # Critical high-risk keywords (should immediately elevate to high/critical)
        critical_risk_keywords = ["delete", "remove all", "drop table", "truncate", "rm -rf"]
        production_keywords = ["production", "prod", "live"]
        
        has_critical_risk = any(k in task_lower for k in critical_risk_keywords)
        has_production_context = any(k in task_lower for k in production_keywords)
        
        if has_critical_risk and has_production_context:
            uncertainty_factors.append("CRITICAL: Destructive operation on production system")
            score += 0.6  # This alone should push to critical
        elif has_critical_risk:
            uncertainty_factors.append("High-risk destructive operation detected")
            score += 0.4
        elif has_production_context:
            uncertainty_factors.append("Production environment operation")
            score += 0.3
        
        # Other high-risk keywords
        risk_keywords = ["database", "migration", "deploy", "schema change", "backup"]
        matching_risks = [k for k in risk_keywords if k in task_lower]
        if matching_risks:
            uncertainty_factors.append(f"High-risk operation detected: {', '.join(matching_risks)}")
            score += 0.2 * len(matching_risks)  # Multiple risk factors compound
        
        # Lack of specifics
        if not re.search(r'[\w/]+\.\w+', task):  # No file paths
            if "file" in task_lower or "code" in task_lower:
                uncertainty_factors.append("References files but no specific paths")
                score += 0.15
        
        # Multiple systems/components mentioned
        multi_component_indicators = [
            "multiple", "several", "various", "all", "entire", "complete",
            "system", "component", "module", "service"
        ]
        component_count = sum(1 for indicator in multi_component_indicators if indicator in task_lower)
        if component_count >= 2:
            uncertainty_factors.append("Multiple system components involved")
            score += 0.2
        
        # Check historical failure patterns
        failure_warnings = self.db.get_failure_warnings_for_task(task)
        if failure_warnings:
            uncertainty_factors.append("Similar tasks have failed before")
            score += 0.2
        
        # Clamp score
        score = min(1.0, score)
        
        # Determine level and action - adjusted thresholds for better classification
        if score < 0.2:
            level = "low"
            action = "proceed"
            fallbacks = []
        elif score < 0.4:
            level = "medium"
            action = "cautious"
            fallbacks = ["explore_first", "trial_and_error"]
        elif score < 0.7:
            level = "high"
            action = "escalate"
            fallbacks = ["explore_first", "decompose_sequential"]
        else:
            level = "critical"
            action = "ask_user"
            fallbacks = ["template_match", "explore_first"]
        
        return Uncertainty(
            level=level,
            score=score,
            reasons=uncertainty_factors if uncertainty_factors else ["Task appears straightforward"],
            recommended_action=action,
            fallback_strategies=fallbacks,
        )
    
    def get_complexity_indicators(self, task: str) -> dict[str, List[str]]:
        """
        Get detailed complexity analysis for debugging/explanation purposes.
        
        Returns:
            Dict with 'high', 'medium', 'low' complexity indicators found
        """
        task_lower = task.lower()
        
        indicators = {
            'high': [],
            'medium': [],
            'low': []
        }
        
        # High complexity patterns
        high_patterns = [
            'architect', 'design', 'refactor', 'optimize', 'integrate',
            'migrate', 'scale', 'distributed', 'concurrent', 'parallel',
            'performance', 'security', 'infrastructure', 'pipeline',
            'framework', 'platform', 'system-wide', 'ecosystem',
            'oauth2', 'authentication', 'authorization', 'encryption'
        ]
        
        # Medium complexity patterns
        medium_patterns = [
            'implement', 'create', 'build', 'develop', 'extend',
            'modify', 'update', 'enhance', 'improve', 'add feature',
            'api', 'interface', 'component', 'module', 'service',
            'test', 'validate', 'analyze', 'investigate'
        ]
        
        # Low complexity patterns
        low_patterns = [
            'fix', 'correct', 'adjust', 'rename', 'move', 'copy',
            'delete', 'remove', 'add', 'insert', 'append', 'prepend',
            'format', 'style', 'comment', 'document', 'typo',
            'change', 'replace', 'swap'
        ]
        
        for pattern in high_patterns:
            if pattern in task_lower:
                indicators['high'].append(pattern)
                
        for pattern in medium_patterns:
            if pattern in task_lower:
                indicators['medium'].append(pattern)
                
        for pattern in low_patterns:
            if pattern in task_lower:
                indicators['low'].append(pattern)
        
        return indicators
    
    def get_risk_analysis(self, task: str, context: str = "") -> dict:
        """
        Get detailed risk analysis for a task.
        
        Returns:
            Dict with risk categories and specific risks found
        """
        combined_text = f"{task.lower()} {context.lower()}"
        
        risks = {
            'vague_language': [],
            'unclear_requirements': [],
            'complexity_indicators': [],
            'architectural_complexity': [],
            'critical_operations': [],
            'production_risks': [],
            'high_risk_operations': [],
            'specificity_issues': [],
            'historical_failures': []
        }
        
        # Vague language analysis
        vague_words = ["somehow", "maybe", "might", "something", "stuff", "thing", "whatever", "etc"]
        for word in vague_words:
            if word in combined_text:
                risks['vague_language'].append(word)
        
        # Unclear requirements
        unclear_phrases = ["don't know", "not sure", "unclear", "ambiguous"]
        for phrase in unclear_phrases:
            if phrase in combined_text:
                risks['unclear_requirements'].append(phrase)
        
        # Complexity indicators
        complexity = self.get_complexity_indicators(task)
        risks['complexity_indicators'] = complexity
        
        # Architectural complexity
        architectural_terms = ["entire system", "refactor entire", "architecture", "system-wide"]
        for term in architectural_terms:
            if term in combined_text:
                risks['architectural_complexity'].append(term)
        
        # Critical operations
        critical_ops = ["delete", "remove all", "drop table", "truncate", "rm -rf"]
        for op in critical_ops:
            if op in combined_text:
                risks['critical_operations'].append(op)
        
        # Production risks
        production_terms = ["production", "prod", "live"]
        for term in production_terms:
            if term in combined_text:
                risks['production_risks'].append(term)
        
        # High risk operations
        risk_keywords = ["database", "migration", "deploy", "schema change", "backup"]
        for keyword in risk_keywords:
            if keyword in combined_text:
                risks['high_risk_operations'].append(keyword)
        
        # Specificity issues
        if not re.search(r'[\w/]+\.\w+', task) and ("file" in combined_text or "code" in combined_text):
            risks['specificity_issues'].append("References files but no specific paths")
        
        # Historical failures
        risks['historical_failures'] = self.db.get_failure_warnings_for_task(task)
        
        return risks