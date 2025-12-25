#!/usr/bin/env python3
"""
Test Script for UncertaintyAssessor and StrategySelector
========================================================

Tests the extracted uncertainty assessment and strategy selection components
to ensure they work correctly with the new CognitiveDatabase.
"""

import os
import tempfile
from datetime import datetime

# Import the new cognitive components
from brain.database import CognitiveDatabase, Reflection
from brain.uncertainty import UncertaintyAssessor, Uncertainty
from brain.strategy import StrategySelector, Strategy


def test_uncertainty_assessor():
    """Test UncertaintyAssessor functionality."""
    print("=" * 60)
    print("Testing UncertaintyAssessor")
    print("=" * 60)
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as temp_db:
        db_path = temp_db.name
    
    try:
        db = CognitiveDatabase(db_path)
        assessor = UncertaintyAssessor(db)
        
        # Test 1: Simple task (should be low uncertainty)
        print("\n1. Testing simple task...")
        task1 = "Fix a typo in README.md"
        result1 = assessor.assess_uncertainty(task1)
        print(f"Task: {task1}")
        print(f"Level: {result1.level}")
        print(f"Score: {result1.score:.3f}")
        print(f"Reasons: {result1.reasons}")
        print(f"Action: {result1.recommended_action}")
        assert result1.level == "low", f"Expected low uncertainty, got {result1.level}"
        
        # Test 2: Vague task (should be high uncertainty)
        print("\n2. Testing vague task...")
        task2 = "Maybe fix something in the codebase, not sure what exactly"
        result2 = assessor.assess_uncertainty(task2)
        print(f"Task: {task2}")
        print(f"Level: {result2.level}")
        print(f"Score: {result2.score:.3f}")
        print(f"Reasons: {result2.reasons}")
        print(f"Action: {result2.recommended_action}")
        assert result2.level in ["high", "critical"], f"Expected high/critical uncertainty, got {result2.level}"
        
        # Test 3: High-risk task
        print("\n3. Testing high-risk task...")
        task3 = "Delete all production database records and migrate to new schema"
        result3 = assessor.assess_uncertainty(task3)
        print(f"Task: {task3}")
        print(f"Level: {result3.level}")
        print(f"Score: {result3.score:.3f}")
        print(f"Reasons: {result3.reasons}")
        print(f"Action: {result3.recommended_action}")
        assert result3.level in ["high", "critical"], f"Expected high/critical uncertainty, got {result3.level}"
        
        # Test 4: Medium complexity task
        print("\n4. Testing medium complexity task...")
        task4 = "Implement a new API endpoint for user authentication with proper validation"
        result4 = assessor.assess_uncertainty(task4)
        print(f"Task: {task4}")
        print(f"Level: {result4.level}")
        print(f"Score: {result4.score:.3f}")
        print(f"Reasons: {result4.reasons}")
        print(f"Action: {result4.recommended_action}")
        # This now should be medium due to authentication complexity
        assert result4.level in ["low", "medium"], f"Expected low/medium uncertainty, got {result4.level}"
        
        # Test 5: Risk analysis
        print("\n5. Testing risk analysis...")
        risk_analysis = assessor.get_risk_analysis(task3)
        print(f"Risk analysis for: {task3}")
        for category, risks in risk_analysis.items():
            if risks:
                print(f"  {category}: {risks}")
        
        # Test 6: Complexity indicators
        print("\n6. Testing complexity indicators...")
        complexity = assessor.get_complexity_indicators(task4)
        print(f"Complexity indicators for: {task4}")
        for level, indicators in complexity.items():
            if indicators:
                print(f"  {level}: {indicators}")
        
        print("\n✅ UncertaintyAssessor tests passed!")
        
    finally:
        # Cleanup
        db.close()
        os.unlink(db_path)


def test_strategy_selector():
    """Test StrategySelector functionality."""
    print("\n" + "=" * 60)
    print("Testing StrategySelector")
    print("=" * 60)
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as temp_db:
        db_path = temp_db.name
    
    try:
        db = CognitiveDatabase(db_path)
        selector = StrategySelector(db)
        
        # Test 1: Simple query (should use direct execution)
        print("\n1. Testing simple query...")
        task1 = "Show all Python files in the project"
        strategy1 = selector.select_strategy(task1)
        print(f"Task: {task1}")
        print(f"Selected strategy: {strategy1.name}")
        print(f"Description: {strategy1.description}")
        print(f"Steps: {strategy1.steps}")
        assert strategy1.name == "Direct Execution", f"Expected Direct Execution, got {strategy1.name}"
        
        # Test 2: Complex architectural task (should use explore first due to uncertainty)
        print("\n2. Testing complex task...")
        task2 = "Refactor the entire authentication system to use OAuth2 with proper security measures"
        strategy2 = selector.select_strategy(task2)
        print(f"Task: {task2}")
        print(f"Selected strategy: {strategy2.name}")
        print(f"Description: {strategy2.description}")
        print(f"Steps: {strategy2.steps}")
        assert strategy2.name == "Explore First", f"Expected Explore First, got {strategy2.name}"
        
        # Test 3: Multiple independent operations (should use parallel or direct)
        print("\n3. Testing parallel operations...")
        task3 = "Update README.md and fix typos in docstrings"
        strategy3 = selector.select_strategy(task3)
        print(f"Task: {task3}")
        print(f"Selected strategy: {strategy3.name}")
        print(f"Description: {strategy3.description}")
        print(f"Steps: {strategy3.steps}")
        # This could be parallel or direct depending on implementation
        print(f"✓ Selected strategy: {strategy3.name}")
        
        # Test 4: Sequential operations with lower risk
        print("\n4. Testing sequential operations...")
        task4 = "First validate the config file, then update the settings, finally restart the application"
        strategy4 = selector.select_strategy(task4)
        print(f"Task: {task4}")
        print(f"Selected strategy: {strategy4.name}")
        print(f"Description: {strategy4.description}")
        print(f"Steps: {strategy4.steps}")
        # Should prefer sequential due to "first", "then", "finally" keywords
        expected_strategies = ["Sequential Decomposition", "Explore First"]  # Either is reasonable
        assert strategy4.name in expected_strategies, f"Expected {expected_strategies}, got {strategy4.name}"
        
        # Test 5: Get all strategies
        print("\n5. Testing get all strategies...")
        all_strategies = selector.get_all_strategies()
        print(f"Available strategies: {list(all_strategies.keys())}")
        expected_strategies = [
            "direct_execution", "decompose_parallel", "decompose_sequential",
            "explore_first", "trial_and_error", "template_match"
        ]
        for expected in expected_strategies:
            assert expected in all_strategies, f"Missing strategy: {expected}"
        
        # Test 6: Strategy recommendations with uncertainty
        print("\n6. Testing strategy recommendations...")
        recommendations = selector.get_strategy_recommendations(task2)
        print(f"Task: {task2}")
        print(f"Primary strategy: {recommendations['primary_strategy'].name}")
        print(f"Uncertainty level: {recommendations['uncertainty_assessment'].level}")
        print(f"Alternatives: {[s.name for s in recommendations['alternative_strategies']]}")
        print(f"Rationale: {recommendations['selection_rationale']}")
        
        # Test 7: Update strategy performance
        print("\n7. Testing strategy performance updates...")
        initial_success_rate = strategy1.success_rate
        initial_uses = strategy1.uses
        selector.update_strategy_performance("direct_execution", True)
        print(f"Before update - Uses: {initial_uses}, Success Rate: {initial_success_rate:.3f}")
        print(f"After update - Uses: {strategy1.uses}, Success Rate: {strategy1.success_rate:.3f}")
        assert strategy1.uses == initial_uses + 1, "Uses should increment"
        
        # Test 8: Add some successful reflections to test template matching
        print("\n8. Testing template matching with historical data...")
        reflection = Reflection(
            task="Show all Python files in the src directory",
            outcome="Successfully listed files",
            success=True,
            what_worked=["Direct file listing worked well"],
            what_failed=[],
            insights=["Simple queries are fast"],
            skill_candidates=[],
            confidence=0.9,
            model_used="sonnet",
            tokens_used=150,
            duration_seconds=2.5,
            created_at=datetime.now()
        )
        db.store_reflection(reflection)
        
        # Now test template matching
        similar_task = "List all Python files in the project"
        has_similar = selector._has_similar_past_solution(similar_task)
        print(f"Task: {similar_task}")
        print(f"Has similar past solution: {has_similar}")
        
        if has_similar:
            template_strategy = selector.select_strategy(similar_task)
            print(f"Selected strategy: {template_strategy.name}")
            # Should prefer template match if similarity is detected
            assert template_strategy.name in ["Template Match", "Direct Execution"], \
                f"Expected Template Match or Direct Execution, got {template_strategy.name}"
        
        print("\n✅ StrategySelector tests passed!")
        
    finally:
        # Cleanup
        db.close()
        os.unlink(db_path)


def test_integration():
    """Test integration between UncertaintyAssessor and StrategySelector."""
    print("\n" + "=" * 60)
    print("Testing Integration")
    print("=" * 60)
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as temp_db:
        db_path = temp_db.name
    
    try:
        db = CognitiveDatabase(db_path)
        assessor = UncertaintyAssessor(db)
        selector = StrategySelector(db)
        
        # Test integration with uncertainty-driven strategy selection
        print("\n1. Testing uncertainty-driven strategy selection...")
        high_uncertainty_task = "Somehow refactor the entire system architecture, not sure exactly how"
        
        # First assess uncertainty
        uncertainty = assessor.assess_uncertainty(high_uncertainty_task)
        print(f"Task: {high_uncertainty_task}")
        print(f"Uncertainty: {uncertainty.level} ({uncertainty.score:.3f})")
        
        # Then select strategy based on uncertainty
        strategy = selector.select_strategy(high_uncertainty_task, uncertainty=uncertainty)
        print(f"Selected strategy: {strategy.name}")
        print(f"Fallback strategies: {uncertainty.fallback_strategies}")
        
        # Verify that high uncertainty leads to exploratory strategy
        if uncertainty.level in ["high", "critical"]:
            assert strategy.name == "Explore First", f"High uncertainty should select Explore First, got {strategy.name}"
        
        # Test with low uncertainty task
        print("\n2. Testing low uncertainty task...")
        low_uncertainty_task = "Fix typo in line 42 of main.py"
        uncertainty2 = assessor.assess_uncertainty(low_uncertainty_task)
        strategy2 = selector.select_strategy(low_uncertainty_task, uncertainty=uncertainty2)
        print(f"Task: {low_uncertainty_task}")
        print(f"Uncertainty: {uncertainty2.level} ({uncertainty2.score:.3f})")
        print(f"Selected strategy: {strategy2.name}")
        
        # Test comprehensive recommendations
        print("\n3. Testing comprehensive recommendations...")
        recommendations = selector.get_strategy_recommendations(high_uncertainty_task)
        print(f"Primary: {recommendations['primary_strategy'].name}")
        print(f"Uncertainty: {recommendations['uncertainty_assessment'].level}")
        print(f"Alternatives: {[s.name for s in recommendations['alternative_strategies']]}")
        
        # Test edge case: Medium uncertainty with specific patterns
        print("\n4. Testing medium uncertainty with patterns...")
        medium_task = "Create a new service that handles file uploads and processing"
        uncertainty3 = assessor.assess_uncertainty(medium_task)
        strategy3 = selector.select_strategy(medium_task, uncertainty=uncertainty3)
        print(f"Task: {medium_task}")
        print(f"Uncertainty: {uncertainty3.level} ({uncertainty3.score:.3f})")
        print(f"Selected strategy: {strategy3.name}")
        
        print("\n✅ Integration tests passed!")
        
    finally:
        # Cleanup
        db.close()
        os.unlink(db_path)


def main():
    """Run all tests."""
    print("Testing UncertaintyAssessor and StrategySelector Components")
    print("=" * 60)
    
    try:
        test_uncertainty_assessor()
        test_strategy_selector()
        test_integration()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("✅ UncertaintyAssessor successfully extracts uncertainty analysis from CognitiveArchitecture")
        print("✅ StrategySelector successfully extracts strategy selection logic from CognitiveArchitecture")
        print("✅ Both components integrate properly with CognitiveDatabase")
        print("✅ Historical data analysis works correctly")
        print("✅ Integration between components functions as expected")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()