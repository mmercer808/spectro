"""
Example systems demonstrating serializable contexts library usage.

This package contains complete example implementations showing how to use
the library for real-world applications including game narratives,
distributed computing, and AI collaboration.
"""

from matts.examples.game_narrative_system import GameNarrativeSystem, demo_game_narrative_system
from matts.examples.distributed_workers_system import DistributedWorkerSystem, demo_distributed_worker_system  
from matts.examples.ai_collaboration_system import AICollaborationSystem, demo_ai_collaboration_system
from matts.examples.usage_patterns import (
    distributed_processing_example,
    game_narrative_example, 
    ai_collaboration_example,
    UsageExamples
)
from matts.examples.demo_system import (
    run_complete_library_demo,
    run_interactive_demo,
    run_performance_benchmark
)

# Version and metadata
__version__ = "1.0.0"
__author__ = "Serializable Context Library Team"
__description__ = "Example systems for the serializable context library"

# Main example systems
__all__ = [
    # System classes
    'GameNarrativeSystem',
    'DistributedWorkerSystem',
    'AICollaborationSystem',
    'UsageExamples',
    
    # Demo functions for individual systems
    'demo_game_narrative_system',
    'demo_distributed_worker_system',
    'demo_ai_collaboration_system',
    
    # Basic usage examples
    'distributed_processing_example',
    'game_narrative_example',
    'ai_collaboration_example',
    
    # Comprehensive demo runners
    'run_complete_library_demo',
    'run_interactive_demo',
    'run_performance_benchmark'
]

# Example categories for easy discovery
EXAMPLE_CATEGORIES = {
    'game_development': {
        'systems': ['GameNarrativeSystem'],
        'demos': ['demo_game_narrative_system', 'game_narrative_example'],
        'description': 'Game development with serializable NPCs, quests, and narrative state'
    },
    'distributed_computing': {
        'systems': ['DistributedWorkerSystem'],
        'demos': ['demo_distributed_worker_system', 'distributed_processing_example'],
        'description': 'Distributed task execution with serializable worker contexts'
    },
    'ai_collaboration': {
        'systems': ['AICollaborationSystem'],
        'demos': ['demo_ai_collaboration_system', 'ai_collaboration_example'],
        'description': 'Multi-agent AI collaboration with reasoning context sharing'
    },
    'comprehensive': {
        'demos': ['run_complete_library_demo', 'run_interactive_demo', 'run_performance_benchmark'],
        'description': 'Complete demonstrations of all library features'
    }
}

# Quick start recommendations
QUICK_START_EXAMPLES = [
    ('Basic Usage', 'UsageExamples', 'Simple examples of core functionality'),
    ('Game Narrative', 'demo_game_narrative_system', 'RPG-style game with serializable state'),
    ('Distributed Workers', 'demo_distributed_worker_system', 'Multi-worker distributed computing'),
    ('AI Collaboration', 'demo_ai_collaboration_system', 'Multi-agent reasoning and collaboration'),
    ('Complete Demo', 'run_complete_library_demo', 'All features demonstrated together')
]

def get_example_info(category: str = None) -> dict:
    """
    Get information about available examples.
    
    Args:
        category: Optional category filter ('game_development', 'distributed_computing', 
                 'ai_collaboration', 'comprehensive')
    
    Returns:
        Dictionary with example information
    """
    if category:
        if category in EXAMPLE_CATEGORIES:
            return EXAMPLE_CATEGORIES[category]
        else:
            return {'error': f'Unknown category: {category}', 'available': list(EXAMPLE_CATEGORIES.keys())}
    
    return EXAMPLE_CATEGORIES

def list_quick_start_examples() -> list:
    """Get list of recommended quick start examples."""
    return QUICK_START_EXAMPLES

def print_examples_overview():
    """Print a nice overview of all available examples."""
    print("🚀 Serializable Context Library - Examples Overview")
    print("=" * 55)
    print()
    
    print("📚 Quick Start Examples:")
    for name, function_name, description in QUICK_START_EXAMPLES:
        print(f"  • {name}: {description}")
        print(f"    Function: {function_name}")
    print()
    
    print("🗂️  Example Categories:")
    for category, info in EXAMPLE_CATEGORIES.items():
        print(f"  • {category.replace('_', ' ').title()}:")
        print(f"    {info['description']}")
        if 'systems' in info:
            print(f"    Systems: {', '.join(info['systems'])}")
        print(f"    Demos: {', '.join(info['demos'])}")
    print()
    
    print("💡 Usage:")
    print("  from examples import run_complete_library_demo")
    print("  await run_complete_library_demo()")
    print()
    print("  # Or run individual examples:")
    print("  from examples import demo_game_narrative_system")
    print("  game_system = await demo_game_narrative_system()")

# Convenience function for running examples
async def run_example(example_name: str, *args, **kwargs):
    """
    Run a specific example by name.
    
    Args:
        example_name: Name of the example function to run
        *args, **kwargs: Arguments to pass to the example function
    
    Returns:
        Result of the example execution
    """
    if example_name in globals():
        example_func = globals()[example_name]
        if callable(example_func):
            if asyncio.iscoroutinefunction(example_func):
                return await example_func(*args, **kwargs)
            else:
                return example_func(*args, **kwargs)
        else:
            raise ValueError(f"{example_name} is not callable")
    else:
        raise ValueError(f"Example '{example_name}' not found. Available: {list(__all__)}")

# Auto-import for convenience
import asyncio