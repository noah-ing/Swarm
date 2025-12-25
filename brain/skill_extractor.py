"""
Skill Extractor Component
=========================

Handles pattern recognition and skill creation from successful task executions.
Extracted from the monolithic CognitiveArchitecture class.

The SkillExtractor focuses specifically on:
- Identifying reusable patterns from successful tasks
- Creating and managing skill templates
- Updating failure patterns for learning from mistakes
- Analyzing tool sequences for automation opportunities
"""

import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional

from .database import CognitiveDatabase


class SkillExtractor:
    """
    Specialized component for identifying and extracting reusable skills.
    
    Responsibilities:
    - Analyze successful task patterns for skill extraction opportunities
    - Create new skills when patterns are sufficiently validated
    - Update existing skills with new success data
    - Track failure patterns to avoid repeating mistakes
    - Manage skill lifecycle (creation, validation, refinement)
    """

    def __init__(self, db: CognitiveDatabase):
        """Initialize with database dependency injection."""
        self.db = db

    def maybe_create_skill(self, candidate: Dict[str, Any]) -> Optional[str]:
        """
        Create a skill if the pattern is seen multiple times and meets quality criteria.

        This is the main entry point for skill creation, called from ReflectionEngine
        when skill candidates are identified in successful task reflections.

        Args:
            candidate: Dictionary containing pattern, tools, and metadata

        Returns:
            skill_id if skill was created/updated, None otherwise
        """
        pattern = candidate.get("pattern", "")
        tools = candidate.get("tools", [])
        name = candidate.get("name", "")

        # Validate candidate quality
        if not self._is_valid_skill_candidate(pattern, tools):
            return None

        # Check if similar skill already exists
        existing_skill = self.db.find_similar_skill(pattern)
        
        if existing_skill:
            # Update existing skill success count
            self.db.update_skill_success_count(existing_skill["id"])
            return existing_skill["id"]
        else:
            # Create new skill if pattern is specific and valuable enough
            if self._should_create_new_skill(pattern, tools):
                skill_id = self._generate_skill_id(candidate)
                skill_name = self._generate_skill_name(pattern, tools)
                
                self.db.create_skill(skill_id, skill_name, pattern, tools)
                return skill_id

        return None

    def update_failure_pattern(self, task: str, error: str) -> None:
        """
        Track failure patterns to avoid repeating mistakes.

        Called from ReflectionEngine when tasks fail to build up knowledge
        about what doesn't work and why.

        Args:
            task: The task description that failed
            error: The error message or description
        """
        self.db.store_failure_pattern(task, error)

    def extract_skills_from_reflection(self, reflection_data: Dict[str, Any]) -> List[str]:
        """
        Extract potential skills from a completed reflection.

        Analyzes the reflection data for patterns that could be automated
        or reused in future tasks.

        Args:
            reflection_data: Complete reflection data including tools, outcome, etc.

        Returns:
            List of skill IDs that were created or updated
        """
        skill_ids = []

        # Extract from skill candidates if present
        if "skill_candidates" in reflection_data:
            for candidate in reflection_data["skill_candidates"]:
                skill_id = self.maybe_create_skill(candidate)
                if skill_id:
                    skill_ids.append(skill_id)

        # Analyze tool sequences for additional patterns
        if "tool_calls" in reflection_data:
            tool_patterns = self._analyze_tool_patterns(reflection_data["tool_calls"])
            for pattern in tool_patterns:
                skill_id = self.maybe_create_skill(pattern)
                if skill_id:
                    skill_ids.append(skill_id)

        return skill_ids

    def _is_valid_skill_candidate(self, pattern: str, tools: List[str]) -> bool:
        """
        Validate whether a pattern is worth creating as a skill.

        Checks for minimum quality criteria:
        - Sufficient pattern length for specificity
        - Reasonable tool sequence length
        - Not too generic or too specific
        """
        # Pattern must be substantial enough
        if len(pattern) < 20:
            return False

        # Pattern should not be too long (overly specific)
        if len(pattern) > 500:
            return False

        # Should have reasonable tool sequence
        if not tools or len(tools) > 10:
            return False

        # Avoid overly generic patterns
        generic_patterns = [
            "fix", "update", "change", "modify", "create", "delete",
            "list", "show", "print", "get", "set"
        ]
        pattern_lower = pattern.lower()
        if any(pattern_lower.startswith(generic) for generic in generic_patterns):
            if len(pattern) < 50:  # Too generic and short
                return False

        return True

    def _should_create_new_skill(self, pattern: str, tools: List[str]) -> bool:
        """
        Determine if a new skill should be created based on pattern characteristics.

        Uses heuristics to avoid creating too many similar or low-value skills.
        """
        # Require minimum pattern specificity
        if len(pattern) < 30:
            return False

        # Check for valuable tool patterns
        valuable_tool_patterns = self._identify_valuable_tool_patterns(tools)
        if not valuable_tool_patterns:
            return False

        # Check pattern complexity (not too simple, not too complex)
        complexity_score = self._calculate_pattern_complexity(pattern)
        if complexity_score < 0.3 or complexity_score > 0.8:
            return False

        return True

    def _identify_valuable_tool_patterns(self, tools: List[str]) -> bool:
        """
        Identify if a tool sequence represents a valuable, reusable pattern.
        
        Returns True if the tool sequence shows potential for automation.
        """
        if not tools:
            return False

        # Single tool operations are usually not worth extracting
        if len(tools) == 1:
            return False

        # Look for common valuable patterns
        valuable_patterns = [
            # File operations
            ["read", "write"],
            ["search", "read", "write"],
            ["read", "search", "write"],
            
            # Development workflows
            ["search", "read", "bash"],
            ["read", "write", "bash"],
            
            # Multi-step analysis
            ["search", "read", "search"],
            ["read", "read", "write"],
        ]

        # Check if tool sequence matches valuable patterns
        tool_sequence = [tool.lower() for tool in tools]
        for pattern in valuable_patterns:
            if self._sequence_matches_pattern(tool_sequence, pattern):
                return True

        # Check for repeated tool usage (potential optimization)
        if len(set(tools)) < len(tools):  # Has duplicates
            return True

        return False

    def _sequence_matches_pattern(self, sequence: List[str], pattern: List[str]) -> bool:
        """Check if a tool sequence matches a valuable pattern."""
        if len(sequence) != len(pattern):
            return False
        
        for seq_tool, pattern_tool in zip(sequence, pattern):
            if pattern_tool not in seq_tool:  # Allow partial matches
                return False
        
        return True

    def _calculate_pattern_complexity(self, pattern: str) -> float:
        """
        Calculate complexity score for a pattern.

        Returns value from 0.0 (simple) to 1.0 (complex).
        Used to filter patterns that are too simple or too complex for skills.
        """
        complexity_score = 0.0

        # Length-based complexity
        word_count = len(pattern.split())
        complexity_score += min(0.4, word_count / 50)

        # Technical terms indicate complexity
        technical_terms = [
            "implement", "refactor", "optimize", "integrate", "configure",
            "deploy", "migrate", "transform", "analyze", "architect"
        ]
        pattern_lower = pattern.lower()
        for term in technical_terms:
            if term in pattern_lower:
                complexity_score += 0.1

        # File operations indicate concrete actions
        if "." in pattern and any(ext in pattern for ext in [".py", ".js", ".txt", ".json"]):
            complexity_score += 0.2

        # Conditional logic indicates complexity
        conditional_words = ["if", "when", "unless", "based on", "depending"]
        for word in conditional_words:
            if word in pattern_lower:
                complexity_score += 0.1

        return min(1.0, complexity_score)

    def _generate_skill_id(self, candidate: Dict[str, Any]) -> str:
        """Generate a unique skill ID from candidate data."""
        if "name" in candidate and candidate["name"]:
            return candidate["name"]
        
        pattern = candidate.get("pattern", "")
        return f"skill_{hashlib.md5(pattern.encode()).hexdigest()[:16]}"

    def _generate_skill_name(self, pattern: str, tools: List[str]) -> str:
        """Generate a descriptive name for the skill."""
        # Extract key action from pattern
        pattern_words = pattern.lower().split()
        action_words = ["create", "update", "fix", "delete", "search", "read", "write"]
        
        main_action = "process"
        for word in pattern_words:
            if word in action_words:
                main_action = word
                break

        # Create name based on tools and action
        if len(tools) <= 2:
            tool_desc = "-".join(tools[:2])
        else:
            tool_desc = f"{tools[0]}-multi"

        return f"Auto-skill: {main_action} via {tool_desc}"

    def _analyze_tool_patterns(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze tool call sequences for additional skill extraction opportunities.

        Looks beyond the basic skill candidates to find deeper patterns in tool usage.
        """
        if not tool_calls:
            return []

        patterns = []
        tool_names = [call.get("name") for call in tool_calls if call.get("name")]

        # Look for repeated sequences
        if len(tool_names) >= 3:
            for i in range(len(tool_names) - 2):
                sequence = tool_names[i:i+3]
                if len(set(sequence)) > 1:  # Not all same tool
                    patterns.append({
                        "name": f"sequence_{hashlib.md5(''.join(sequence).encode()).hexdigest()[:8]}",
                        "pattern": f"Tool sequence: {' -> '.join(sequence)}",
                        "tools": sequence
                    })

        # Look for alternating patterns
        if len(tool_names) >= 4:
            for i in range(len(tool_names) - 3):
                if (tool_names[i] == tool_names[i+2] and 
                    tool_names[i+1] == tool_names[i+3] and
                    tool_names[i] != tool_names[i+1]):
                    
                    patterns.append({
                        "name": f"alternating_{hashlib.md5(f'{tool_names[i]}{tool_names[i+1]}'.encode()).hexdigest()[:8]}",
                        "pattern": f"Alternating pattern: {tool_names[i]} <-> {tool_names[i+1]}",
                        "tools": [tool_names[i], tool_names[i+1]]
                    })

        return patterns[:3]  # Limit to most promising patterns

    def get_skill_extraction_stats(self) -> Dict[str, Any]:
        """
        Get statistics about skill extraction performance.
        
        Returns metrics useful for monitoring extraction quality and system health.
        """
        base_stats = self.db.get_system_stats()
        
        # Add skill-specific metrics
        skill_stats = {
            "total_skills": base_stats.get("skills_learned", 0),
            "failure_patterns": base_stats.get("failure_patterns_tracked", 0)
        }

        # Calculate extraction rate if we have reflection data
        if base_stats.get("successful_reflections", 0) > 0:
            skill_stats["skill_extraction_rate"] = (
                skill_stats["total_skills"] / base_stats["successful_reflections"]
            )
        else:
            skill_stats["skill_extraction_rate"] = 0.0

        return skill_stats

    def validate_existing_skills(self) -> Dict[str, Any]:
        """
        Validate existing skills for quality and usefulness.
        
        Returns analysis of skill database health and recommendations.
        """
        # This would analyze existing skills in the database
        # For now, return basic validation info
        return {
            "status": "Skills validation not yet implemented",
            "recommendation": "Consider implementing skill quality scoring"
        }