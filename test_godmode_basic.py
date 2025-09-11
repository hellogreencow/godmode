#!/usr/bin/env python3
"""
Basic test script for GodMode system functionality.

This script provides basic tests to verify the core functionality
of the GodMode hierarchical reasoning system.
"""

import asyncio
import sys
import time
from typing import List

from godmode import (
    GodModeEngine,
    HierarchicalReasoningModel,
    Problem,
    ReasoningType,
)


def create_test_problem() -> Problem:
    """Create a simple test problem."""
    return Problem(
        title="Test Problem",
        description="Find an efficient way to organize a team of 10 people to complete a project in 3 months",
        problem_type="planning",
        domain="project_management",
        constraints=["3 month deadline", "10 team members", "limited budget"],
        objectives=["Complete project on time", "Maximize efficiency", "Minimize costs"]
    )


async def test_godmode_engine():
    """Test the main GodMode engine."""
    print("🧠 Testing GodMode Engine...")
    
    try:
        # Initialize engine
        engine = GodModeEngine(
            memory_size=1000,
            reasoning_depth=3,
            enable_gpu=False,
        )
        print("✅ Engine initialized successfully")
        
        # Create test problem
        problem = create_test_problem()
        print(f"📝 Created test problem: {problem.title}")
        
        # Test problem solving
        start_time = time.time()
        response = await engine.solve_problem(
            problem=problem,
            reasoning_type=ReasoningType.HIERARCHICAL,
            max_time=10.0,
            min_confidence=0.3
        )
        solve_time = time.time() - start_time
        
        print(f"⏱️  Solving time: {solve_time:.2f} seconds")
        
        # Check results
        if response.solution:
            print(f"✅ Solution found with confidence: {response.solution.confidence:.1%}")
            print(f"📊 Quality score: {response.solution.get_overall_quality():.1%}")
            print(f"🔄 Reasoning steps: {response.total_steps}")
            print(f"📈 Success rate: {response.success_rate:.1%}")
        else:
            print("❌ No solution found")
            return False
        
        # Test memory system
        print("\n🧠 Testing Memory System...")
        
        # Store test data
        await engine.memory.store({
            "text": "Project management requires clear communication",
            "type": "insight"
        }, importance=0.8)
        
        # Retrieve data
        results = await engine.memory.retrieve("project management", top_k=2)
        if results:
            print(f"✅ Memory retrieval successful: {len(results)} results")
        else:
            print("⚠️  No memory results found")
        
        # Get statistics
        stats = engine.get_statistics()
        print(f"📊 Engine statistics: {len(stats)} metrics tracked")
        
        return True
        
    except Exception as e:
        print(f"❌ Engine test failed: {e}")
        return False


def test_hierarchical_model():
    """Test the hierarchical reasoning model."""
    print("\n🏗️  Testing Hierarchical Reasoning Model...")
    
    try:
        # Initialize model
        model = HierarchicalReasoningModel(
            embedding_dim=128,  # Smaller for testing
            use_pretrained=False,  # Faster initialization
        )
        print("✅ Hierarchical model initialized successfully")
        
        # Create test problem
        problem = create_test_problem()
        
        # Test problem solving
        start_time = time.time()
        result = model.solve_problem(problem)
        solve_time = time.time() - start_time
        
        print(f"⏱️  Solving time: {solve_time:.2f} seconds")
        
        # Check results
        if result["solutions"]:
            print(f"✅ {len(result['solutions'])} solutions generated")
            print(f"🎯 Overall confidence: {result['confidence']:.1%}")
            
            best_solution = result["solutions"][0]
            print(f"📊 Best solution confidence: {best_solution.confidence:.1%}")
        else:
            print("❌ No solutions generated")
            return False
        
        # Check reasoning trace
        if result["reasoning_trace"]:
            trace = result["reasoning_trace"]
            print(f"🔄 Reasoning levels used: {len(trace['levels_activated'])}")
            print(f"📈 Processing steps: {trace['processing_steps']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hierarchical model test failed: {e}")
        return False


async def test_reasoning_comparison():
    """Test and compare different reasoning approaches."""
    print("\n⚖️  Testing Reasoning Comparison...")
    
    try:
        engine = GodModeEngine(memory_size=500, reasoning_depth=2, enable_gpu=False)
        problem = create_test_problem()
        
        reasoning_types = [
            ReasoningType.FORWARD,
            ReasoningType.BACKWARD,
            ReasoningType.HIERARCHICAL,
        ]
        
        results = {}
        
        for reasoning_type in reasoning_types:
            start_time = time.time()
            response = await engine.solve_problem(
                problem=problem,
                reasoning_type=reasoning_type,
                max_time=5.0,
                min_confidence=0.2
            )
            solve_time = time.time() - start_time
            
            results[reasoning_type.value] = {
                "time": solve_time,
                "success": response.solution is not None,
                "confidence": response.solution.confidence if response.solution else 0,
                "steps": response.total_steps,
            }
            
            print(f"  {reasoning_type.value}: {solve_time:.2f}s, "
                  f"Success: {'✅' if response.solution else '❌'}, "
                  f"Confidence: {response.solution.confidence:.1%}" if response.solution else "0%")
        
        # Find best approach
        best_approach = max(results.items(), key=lambda x: x[1]["confidence"])
        print(f"🏆 Best approach: {best_approach[0]} with {best_approach[1]['confidence']:.1%} confidence")
        
        return True
        
    except Exception as e:
        print(f"❌ Reasoning comparison test failed: {e}")
        return False


def test_basic_imports():
    """Test that all basic imports work."""
    print("📦 Testing Basic Imports...")
    
    try:
        from godmode.models.core import Problem, Solution, CognitiveLevel
        from godmode.core.engine import GodModeEngine
        from godmode.experimental.hierarchical_reasoning import HierarchicalReasoningModel
        from godmode.core.memory import CognitiveMemory
        
        print("✅ All core imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False


async def run_all_tests():
    """Run all basic tests."""
    print("🚀 Starting GodMode Basic Tests\n")
    
    tests = [
        ("Basic Imports", test_basic_imports, False),
        ("GodMode Engine", test_godmode_engine, True),
        ("Hierarchical Model", test_hierarchical_model, False),
        ("Reasoning Comparison", test_reasoning_comparison, True),
    ]
    
    results = []
    
    for test_name, test_func, is_async in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            if is_async:
                result = await test_func()
            else:
                result = test_func()
            
            results.append((test_name, result))
            
        except Exception as e:
            print(f"❌ Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! GodMode is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
