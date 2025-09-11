#!/usr/bin/env python3
"""
Demo script showing GODMODE usage
"""

import asyncio
import json
from godmode import GodmodeEngine
from godmode.schemas import Command, InitCommand


async def demo():
    """Run a simple demo of GODMODE"""
    print("🧠 GODMODE Demo - Question Foresight Engine")
    print("=" * 50)
    
    # Initialize engine
    engine = GodmodeEngine()
    
    # Example question
    question = "Should I switch careers from software engineering to product management?"
    context = """
    I'm a senior software engineer with 6 years experience at tech companies.
    I enjoy the strategic aspects of my work but find pure coding less fulfilling lately.
    I have some product sense but no formal PM experience.
    I'm considering the career change but worried about starting over.
    """
    
    print(f"🤔 Current Question: {question}")
    print(f"📝 Context: {context.strip()}")
    print("\n" + "=" * 50)
    
    # Create INIT command
    command = Command(
        command_type="INIT",
        data=InitCommand(
            current_question=question,
            context=context
        )
    )
    
    try:
        # Process the command
        print("⚡ Processing question through GODMODE...")
        response = await engine.process_command(command)
        
        # Display results
        print("\n🎯 GODMODE Response:")
        print("-" * 30)
        print(f"💬 Chat Reply: {response.chat_reply}")
        
        print(f"\n📊 Generated {len(response.graph_update.priors)} prior questions:")
        for i, prior in enumerate(response.graph_update.priors, 1):
            print(f"  {i}. [{prior.id}] {prior.text}")
            print(f"     → {prior.delta_nuance}")
            print(f"     → Info gain: {prior.expected_info_gain:.2f}, Confidence: {prior.confidence:.2f}")
        
        print(f"\n🛤️  Generated {len(response.graph_update.scenarios)} scenario lanes:")
        for scenario in response.graph_update.scenarios:
            print(f"\n  🎭 {scenario.name} ({scenario.id})")
            print(f"     📖 {scenario.description}")
            for j, question in enumerate(scenario.lane, 1):
                print(f"     {j}. [{question.id}] {question.text}")
                print(f"        → {question.cognitive_move.upper()}: {question.delta_nuance}")
        
        print(f"\n🧵 Created {len(response.graph_update.threads)} exploration threads")
        for thread in response.graph_update.threads:
            print(f"  • {thread.thread_id}: {thread.summary}")
        
        print(f"\n🔗 Ontology extracted:")
        print(f"  • {len(response.ontology_update.entities)} entities")
        print(f"  • {len(response.ontology_update.relations)} relations")
        
        # Show some entities
        if response.ontology_update.entities:
            print("  📋 Key entities:")
            for entity in response.ontology_update.entities[:5]:
                print(f"    - {entity.name} ({entity.type})")
        
        print(f"\n⏱️  Processing time: {response.graph_update.meta.budgets_used.time_s:.2f}s")
        print("✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(demo())