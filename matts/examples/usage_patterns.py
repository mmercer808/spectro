#!/usr/bin/env python3
"""
Usage Examples and Patterns

Simple examples of how to use the library.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any

# matts.functional_interface
from matts import (
    create_context, transmit_context, emit_signal,
    quick_start, health_check
)
# matts.signal_system
from matts import SignalType, SignalPriority

# =============================================================================
# BASIC USAGE EXAMPLES
# =============================================================================

async def distributed_processing_example():
    """Example: Distributed processing with context transmission."""
    print("🌐 Distributed Processing Example")
    
    # Create contexts
    worker1 = create_context("worker_1")
    worker2 = create_context("worker_2")
    
    # Add data
    await worker1.context_chain.update_context_data({
        'worker_type': 'processor',
        'data': [1, 2, 3, 4, 5]
    })
    
    # Transmit to worker2
    result = await transmit_context(worker1.context_id, worker2.context_id)
    
    return {'success': result.get('handled', False)}

async def game_narrative_example():
    """Example: Game narrative system."""
    print("🎮 Game Narrative Example")
    
    narrative = create_context("game_narrative")
    
    await narrative.context_chain.update_context_data({
        'player_name': 'Hero',
        'location': 'village',
        'inventory': ['sword', 'potion']
    })
    
    return {'context_id': narrative.context_id}

async def ai_collaboration_example():
    """Example: AI collaboration."""
    print("🤖 AI Collaboration Example")
    
    analyzer = create_context("ai_analyzer")
    generator = create_context("ai_generator")
    
    await analyzer.context_chain.update_context_data({
        'analysis_result': {'sentiment': 'positive', 'confidence': 0.8}
    })
    
    result = await transmit_context(analyzer.context_id, generator.context_id)
    
    return {'collaboration_success': result.get('handled', False)}

# =============================================================================
# USAGE EXAMPLES CLASS
# =============================================================================

class UsageExamples:
    """Collection of usage examples."""
    
    @staticmethod
    async def basic_setup():
        """Basic library setup example."""
        library, signal_line, context = await quick_start()
        return {
            'library': library,
            'signal_line': signal_line,
            'context': context
        }
    
    @staticmethod
    async def signal_example():
        """Basic signal usage example."""
        await emit_signal(
            SignalType.CUSTOM,
            source_context_id="example",
            data={'message': 'Hello World'}
        )
        return {'signal_sent': True}

# Export functions for direct import
__all__ = [
    'distributed_processing_example',
    'game_narrative_example', 
    'ai_collaboration_example',
    'UsageExamples'
]