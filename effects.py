"""
Effect Prediction: Analyze potential impact of changes before making them.

Predicts:
- Which files will be affected by a change
- Risk level of the change
- Potential side effects
- Symbols that might break
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from codebase import get_codebase_analyzer, CodebaseMap, FileInfo


@dataclass
class AffectedFile:
    """A file that would be affected by a change."""
    path: str
    reason: str  # Why this file is affected
    impact_type: str  # "direct", "indirect", "test"
    symbols_at_risk: list[str] = field(default_factory=list)


@dataclass
class EffectPrediction:
    """Prediction of effects from a proposed change."""
    target_files: list[str]  # Files being changed
    affected_files: list[AffectedFile]  # Files that depend on targets
    risk_level: str  # "low", "medium", "high", "critical"
    risk_score: float  # 0.0 to 1.0
    warnings: list[str]  # Specific warnings
    suggestions: list[str]  # How to mitigate risk
    breaking_changes: list[str]  # Potential breaking changes
    test_files: list[str]  # Tests that should be run


class EffectPredictor:
    """
    Predicts the effects of code changes before they're made.

    Uses dependency analysis and symbol tracking to identify:
    - Direct dependents (files that import the changed file)
    - Indirect dependents (files that import dependents)
    - Affected symbols (functions/classes that might break)
    - Risk assessment based on criticality and scope
    """

    # Risk multipliers for different file types
    CRITICALITY = {
        "Application entry point": 3.0,
        "Module entry point": 2.0,
        "API routes": 2.5,
        "Configuration": 2.5,
        "Agent implementation": 2.0,
        "Data models": 2.0,
        "Test file": 0.5,  # Lower risk - tests should fail safely
        "Documentation": 0.1,
        "Source file": 1.0,
    }

    def __init__(self):
        self.analyzer = get_codebase_analyzer()

    def predict(
        self,
        root_path: str,
        target_files: list[str],
        change_description: str = "",
        symbols_changed: Optional[list[str]] = None,
    ) -> EffectPrediction:
        """
        Predict effects of changing the specified files.

        Args:
            root_path: Path to codebase root
            target_files: Files being changed
            change_description: Description of the change
            symbols_changed: Specific symbols being modified

        Returns:
            EffectPrediction with full impact analysis
        """
        codebase_map = self.analyzer.analyze(root_path)

        # Normalize paths
        target_files = [self._normalize_path(f, root_path) for f in target_files]

        # Build reverse dependency graph (who depends on whom)
        reverse_deps = self._build_reverse_dependency_graph(codebase_map)

        # Find all affected files
        affected = self._find_affected_files(
            target_files, reverse_deps, codebase_map, symbols_changed
        )

        # Find test files that should be run
        test_files = self._find_related_tests(target_files, affected, codebase_map)

        # Identify breaking changes
        breaking = self._identify_breaking_changes(
            target_files, symbols_changed, codebase_map, change_description
        )

        # Calculate risk
        risk_score, risk_level = self._calculate_risk(
            target_files, affected, breaking, codebase_map
        )

        # Generate warnings and suggestions
        warnings = self._generate_warnings(
            target_files, affected, breaking, codebase_map
        )
        suggestions = self._generate_suggestions(
            target_files, affected, risk_level, test_files
        )

        return EffectPrediction(
            target_files=target_files,
            affected_files=affected,
            risk_level=risk_level,
            risk_score=risk_score,
            warnings=warnings,
            suggestions=suggestions,
            breaking_changes=breaking,
            test_files=test_files,
        )

    def _normalize_path(self, path: str, root_path: str) -> str:
        """Normalize a path to be relative to root."""
        if Path(path).is_absolute():
            try:
                return str(Path(path).relative_to(root_path))
            except ValueError:
                return path
        return path

    def _build_reverse_dependency_graph(
        self, codebase_map: CodebaseMap
    ) -> dict[str, list[str]]:
        """Build graph of what depends on each file."""
        reverse = {path: [] for path in codebase_map.files}

        for path, deps in codebase_map.dependency_graph.items():
            for dep in deps:
                if dep in reverse:
                    reverse[dep].append(path)

        return reverse

    def _find_affected_files(
        self,
        target_files: list[str],
        reverse_deps: dict[str, list[str]],
        codebase_map: CodebaseMap,
        symbols_changed: Optional[list[str]],
    ) -> list[AffectedFile]:
        """Find all files affected by the change."""
        affected = []
        visited = set(target_files)

        def add_dependents(file: str, depth: int, reason_prefix: str):
            if depth > 3:  # Limit depth to avoid explosion
                return

            for dependent in reverse_deps.get(file, []):
                if dependent in visited:
                    continue
                visited.add(dependent)

                info = codebase_map.files.get(dependent)
                is_test = info and "test" in info.purpose.lower()

                # Find symbols at risk
                symbols_at_risk = []
                if symbols_changed and info:
                    for sym in symbols_changed:
                        if sym in info.imports or any(sym in f for f in info.functions):
                            symbols_at_risk.append(sym)

                affected.append(AffectedFile(
                    path=dependent,
                    reason=f"{reason_prefix}imports {file}",
                    impact_type="test" if is_test else ("direct" if depth == 0 else "indirect"),
                    symbols_at_risk=symbols_at_risk,
                ))

                # Recurse for indirect dependencies
                add_dependents(dependent, depth + 1, f"indirectly depends on {file} via ")

        for target in target_files:
            add_dependents(target, 0, "directly ")

        return affected

    def _find_related_tests(
        self,
        target_files: list[str],
        affected: list[AffectedFile],
        codebase_map: CodebaseMap,
    ) -> list[str]:
        """Find test files related to the changed files."""
        test_files = []

        # Tests in affected files
        for af in affected:
            if af.impact_type == "test":
                test_files.append(af.path)

        # Tests matching target file names
        for target in target_files:
            base_name = Path(target).stem
            for path in codebase_map.files:
                if "test" in path.lower() and base_name in path.lower():
                    if path not in test_files:
                        test_files.append(path)

        return test_files

    def _identify_breaking_changes(
        self,
        target_files: list[str],
        symbols_changed: Optional[list[str]],
        codebase_map: CodebaseMap,
        change_description: str,
    ) -> list[str]:
        """Identify potential breaking changes."""
        breaking = []
        desc_lower = change_description.lower()

        # Check for breaking keywords in description
        breaking_keywords = [
            "remove", "delete", "rename", "change signature",
            "breaking", "deprecate", "refactor", "rewrite"
        ]

        for keyword in breaking_keywords:
            if keyword in desc_lower:
                breaking.append(f"Change description mentions '{keyword}'")
                break

        # Check if changing exported symbols
        if symbols_changed:
            for target in target_files:
                info = codebase_map.files.get(target)
                if info:
                    for sym in symbols_changed:
                        if sym in info.exports:
                            breaking.append(f"Modifying exported symbol '{sym}' in {target}")

        # Check if changing entry points
        for target in target_files:
            if target in codebase_map.entry_points:
                breaking.append(f"Modifying entry point: {target}")

        return breaking

    def _calculate_risk(
        self,
        target_files: list[str],
        affected: list[AffectedFile],
        breaking: list[str],
        codebase_map: CodebaseMap,
    ) -> tuple[float, str]:
        """Calculate risk score and level."""
        score = 0.0

        # Base risk from number of affected files
        direct_count = sum(1 for a in affected if a.impact_type == "direct")
        indirect_count = sum(1 for a in affected if a.impact_type == "indirect")

        score += min(direct_count * 0.1, 0.3)
        score += min(indirect_count * 0.05, 0.2)

        # Risk from breaking changes
        score += min(len(breaking) * 0.15, 0.3)

        # Risk from target file criticality
        for target in target_files:
            info = codebase_map.files.get(target)
            if info:
                criticality = self.CRITICALITY.get(info.purpose, 1.0)
                score += criticality * 0.05

        # Cap at 1.0
        score = min(score, 1.0)

        # Determine level
        if score < 0.2:
            level = "low"
        elif score < 0.4:
            level = "medium"
        elif score < 0.7:
            level = "high"
        else:
            level = "critical"

        return score, level

    def _generate_warnings(
        self,
        target_files: list[str],
        affected: list[AffectedFile],
        breaking: list[str],
        codebase_map: CodebaseMap,
    ) -> list[str]:
        """Generate specific warnings about the change."""
        warnings = []

        # Warning about breaking changes
        for b in breaking:
            warnings.append(f"BREAKING: {b}")

        # Warning about high-impact files
        for target in target_files:
            info = codebase_map.files.get(target)
            if info:
                if info.purpose == "Application entry point":
                    warnings.append(f"Changing application entry point: {target}")
                elif info.purpose == "Configuration":
                    warnings.append(f"Changing configuration file: {target}")

        # Warning about many dependents
        direct_deps = [a for a in affected if a.impact_type == "direct"]
        if len(direct_deps) > 5:
            warnings.append(f"High impact: {len(direct_deps)} files directly depend on changes")

        # Warning about symbols at risk
        all_symbols = set()
        for a in affected:
            all_symbols.update(a.symbols_at_risk)
        if all_symbols:
            warnings.append(f"Symbols at risk: {', '.join(list(all_symbols)[:5])}")

        return warnings

    def _generate_suggestions(
        self,
        target_files: list[str],
        affected: list[AffectedFile],
        risk_level: str,
        test_files: list[str],
    ) -> list[str]:
        """Generate suggestions to mitigate risk."""
        suggestions = []

        if risk_level in ("high", "critical"):
            suggestions.append("Consider making changes incrementally")
            suggestions.append("Review all affected files before committing")

        if test_files:
            suggestions.append(f"Run tests: {', '.join(test_files[:3])}")
        else:
            suggestions.append("Consider adding tests for the changed code")

        if len(affected) > 3:
            suggestions.append("Consider notifying team members about this change")

        direct_affected = [a.path for a in affected if a.impact_type == "direct"]
        if direct_affected:
            suggestions.append(f"Verify behavior in: {', '.join(direct_affected[:3])}")

        return suggestions

    def predict_from_task(
        self,
        root_path: str,
        task: str,
    ) -> EffectPrediction:
        """
        Predict effects from a task description.

        Analyzes the task to determine which files might be changed
        and predicts the effects.
        """
        codebase_map = self.analyzer.analyze(root_path)

        # Find files likely to be changed based on task
        relevant_files = self.analyzer.get_relevant_files(codebase_map, task)

        # Extract symbols mentioned in task
        task_words = set(task.lower().split())
        symbols_mentioned = []

        for path in relevant_files[:5]:
            info = codebase_map.files.get(path)
            if info:
                for func in info.functions:
                    if func.lower() in task_words:
                        symbols_mentioned.append(func)
                for cls in info.classes:
                    if cls.lower() in task_words:
                        symbols_mentioned.append(cls)

        # Predict based on most likely files to change
        return self.predict(
            root_path=root_path,
            target_files=relevant_files[:3],  # Top 3 most relevant
            change_description=task,
            symbols_changed=symbols_mentioned if symbols_mentioned else None,
        )

    def format_prediction(self, prediction: EffectPrediction) -> str:
        """Format prediction for display."""
        lines = []

        # Header
        lines.append(f"## Effect Prediction")
        lines.append(f"Risk Level: {prediction.risk_level.upper()} ({prediction.risk_score:.0%})")
        lines.append("")

        # Target files
        lines.append(f"### Target Files")
        for f in prediction.target_files:
            lines.append(f"- {f}")
        lines.append("")

        # Affected files
        if prediction.affected_files:
            lines.append(f"### Affected Files ({len(prediction.affected_files)})")
            for af in prediction.affected_files[:10]:
                risk_marker = "*" if af.symbols_at_risk else ""
                lines.append(f"- {af.path} ({af.impact_type}){risk_marker}")
                if af.symbols_at_risk:
                    lines.append(f"  Symbols: {', '.join(af.symbols_at_risk)}")
            if len(prediction.affected_files) > 10:
                lines.append(f"  ... and {len(prediction.affected_files) - 10} more")
            lines.append("")

        # Warnings
        if prediction.warnings:
            lines.append("### Warnings")
            for w in prediction.warnings:
                lines.append(f"- {w}")
            lines.append("")

        # Breaking changes
        if prediction.breaking_changes:
            lines.append("### Breaking Changes")
            for b in prediction.breaking_changes:
                lines.append(f"- {b}")
            lines.append("")

        # Suggestions
        if prediction.suggestions:
            lines.append("### Suggestions")
            for s in prediction.suggestions:
                lines.append(f"- {s}")
            lines.append("")

        # Tests
        if prediction.test_files:
            lines.append("### Tests to Run")
            for t in prediction.test_files:
                lines.append(f"- {t}")

        return "\n".join(lines)


# Global instance
_predictor: Optional[EffectPredictor] = None


def get_effect_predictor() -> EffectPredictor:
    """Get or create the global effect predictor."""
    global _predictor
    if _predictor is None:
        _predictor = EffectPredictor()
    return _predictor
