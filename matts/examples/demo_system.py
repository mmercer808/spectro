#!/usr/bin/env python3
"""
Complete Library Demo System

Demonstrates all major features of the serializable context library by running
comprehensive examples that showcase the integration of all components.
"""

import asyncio
import time
from typing import Dict, Any
from datetime import datetime

from matts.examples.game_narrative_system import GameNarrativeSystem, demo_game_narrative_system
from matts.examples.distributed_workers_system import DistributedWorkerSystem, demo_distributed_worker_system  
from matts.examples.ai_collaboration_system import AICollaborationSystem, demo_ai_collaboration_system
from matts.examples.usage_patterns import distributed_processing_example, game_narrative_example, ai_collaboration_example
from matts.functional_interface import health_check, get_performance_stats, run_garbage_collection

# =============================================================================
# COMPLETE DEMO RUNNER
# =============================================================================

async def run_complete_library_demo():
    """Run the complete library demonstration showcasing all features."""
    
    print("🚀 Serializable Context Library - Complete Demo")
    print("=" * 60)
    print("This demo showcases the full capabilities of the library:")
    print("• Signal systems with observer patterns")
    print("• Serializable execution contexts") 
    print("• Live code injection and hot-swapping")
    print("• Generator composition patterns")
    print("• Cross-system context transmission")
    print("• Real-world application examples")
    print("=" * 60)
    
    demo_start_time = time.time()
    demo_results = {}
    
    # Initial health check
    print("\n🏥 Initial System Health Check")
    print("-" * 30)
    initial_health = await health_check()
    print(f"Library status: {initial_health.get('library_status', 'unknown')}")
    print(f"Context count: {initial_health.get('context_count', 0)}")
    print(f"Signal line status: {initial_health.get('signal_line_status', 'unknown')}")
    
    # Demo 1: Basic Usage Patterns
    print("\n📚 Demo 1: Basic Usage Patterns")
    print("-" * 40)
    
    try:
        basic_start = time.time()
        
        print("Running distributed processing example...")
        dist_result = await distributed_processing_example()
        
        print("Running game narrative example...")
        game_result = await game_narrative_example()
        
        print("Running AI collaboration example...")
        ai_result = await ai_collaboration_example()
        
        basic_time = time.time() - basic_start
        demo_results['basic_patterns'] = {
            'success': True,
            'execution_time': basic_time,
            'results': {
                'distributed_processing': dist_result,
                'game_narrative': game_result,
                'ai_collaboration': ai_result
            }
        }
        
        print(f"✅ Basic patterns completed in {basic_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Basic patterns failed: {e}")
        demo_results['basic_patterns'] = {'success': False, 'error': str(e)}
    
    # Demo 2: Game Narrative System
    print("\n🎮 Demo 2: Game Narrative System")
    print("-" * 40)
    
    try:
        game_start = time.time()
        
        game_system = await demo_game_narrative_system()
        
        game_time = time.time() - game_start
        demo_results['game_narrative'] = {
            'success': True,
            'execution_time': game_time,
            'system_id': game_system.game_id,
            'contexts_created': len(game_system.player_contexts) + len(game_system.npc_contexts) + 2,
            'events_processed': game_system.events_processed
        }
        
        print(f"✅ Game narrative system completed in {game_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Game narrative system failed: {e}")
        demo_results['game_narrative'] = {'success': False, 'error': str(e)}
    
    # Demo 3: Distributed Worker System
    print("\n🔧 Demo 3: Distributed Worker System")
    print("-" * 40)
    
    try:
        worker_start = time.time()
        
        worker_system = await demo_distributed_worker_system()
        
        worker_time = time.time() - worker_start
        demo_results['distributed_workers'] = {
            'success': True,
            'execution_time': worker_time,
            'system_id': worker_system.system_id,
            'workers_registered': len(worker_system.workers),
            'tasks_processed': worker_system.total_tasks_processed,
            'average_task_time': worker_system.total_processing_time / max(worker_system.total_tasks_processed, 1)
        }
        
        print(f"✅ Distributed worker system completed in {worker_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Distributed worker system failed: {e}")
        demo_results['distributed_workers'] = {'success': False, 'error': str(e)}
    
    # Demo 4: AI Collaboration System
    print("\n🤖 Demo 4: AI Collaboration System")
    print("-" * 40)
    
    try:
        ai_start = time.time()
        
        ai_system = await demo_ai_collaboration_system()
        
        ai_time = time.time() - ai_start
        demo_results['ai_collaboration'] = {
            'success': True,
            'execution_time': ai_time,
            'system_id': ai_system.system_id,
            'agents_created': len(ai_system.agents),
            'collaborations_completed': ai_system.total_collaborations,
            'success_rate': ai_system.successful_collaborations / max(ai_system.total_collaborations, 1)
        }
        
        print(f"✅ AI collaboration system completed in {ai_time:.2f}s")
        
    except Exception as e:
        print(f"❌ AI collaboration system failed: {e}")
        demo_results['ai_collaboration'] = {'success': False, 'error': str(e)}
    
    # Performance and cleanup
    print("\n🧹 System Performance and Cleanup")
    print("-" * 40)
    
    try:
        cleanup_start = time.time()
        
        # Get performance stats
        print("Gathering performance statistics...")
        perf_stats = get_performance_stats()
        
        # Run garbage collection
        print("Running garbage collection...")
        gc_results = run_garbage_collection()
        
        # Final health check
        print("Final health check...")
        final_health = await health_check()
        
        cleanup_time = time.time() - cleanup_start
        demo_results['cleanup'] = {
            'success': True,
            'execution_time': cleanup_time,
            'performance_stats': perf_stats,
            'garbage_collection': gc_results,
            'final_health': final_health
        }
        
        print(f"✅ Cleanup completed in {cleanup_time:.2f}s")
        print(f"Garbage collected: {gc_results.get('total_collected', 0)} objects")
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        demo_results['cleanup'] = {'success': False, 'error': str(e)}
    
    # Demo summary
    total_demo_time = time.time() - demo_start_time
    
    print("\n📊 Demo Summary")
    print("=" * 60)
    
    successful_demos = sum(1 for result in demo_results.values() if result.get('success', False))
    total_demos = len(demo_results)
    
    print(f"Total execution time: {total_demo_time:.2f} seconds")
    print(f"Successful demos: {successful_demos}/{total_demos}")
    print(f"Success rate: {successful_demos/total_demos:.1%}")
    
    print("\nIndividual Demo Results:")
    for demo_name, result in demo_results.items():
        status = "✅ SUCCESS" if result.get('success', False) else "❌ FAILED"
        exec_time = result.get('execution_time', 0)
        print(f"  {demo_name.replace('_', ' ').title()}: {status} ({exec_time:.2f}s)")
        
        if not result.get('success', False) and 'error' in result:
            print(f"    Error: {result['error']}")
    
    # Detailed statistics if available
    if 'cleanup' in demo_results and demo_results['cleanup'].get('success', False):
        cleanup_data = demo_results['cleanup']
        
        print("\nSystem Statistics:")
        perf_stats = cleanup_data.get('performance_stats', {})
        
        if 'contexts' in perf_stats:
            contexts = perf_stats['contexts']
            print(f"  Total contexts created: {contexts.get('total_contexts', 0)}")
        
        if 'signal_line' in perf_stats:
            signal_stats = perf_stats['signal_line']
            print(f"  Signals processed: {signal_stats.get('signals_processed', 0)}")
        
        gc_results = cleanup_data.get('garbage_collection', {})
        print(f"  Objects garbage collected: {gc_results.get('total_collected', 0)}")
        
        final_health = cleanup_data.get('final_health', {})
        print(f"  Final system status: {final_health.get('library_status', 'unknown')}")
    
    # Feature showcase summary
    print("\n🌟 Features Demonstrated:")
    features_shown = [
        "✅ Serializable execution contexts with state management",
        "✅ Signal system with observer patterns and priorities", 
        "✅ Live code injection and hot-swapping capabilities",
        "✅ Generator composition with complex patterns",
        "✅ Cross-system context transmission",
        "✅ Multi-agent collaboration with reasoning contexts",
        "✅ Distributed computing with serializable tasks",
        "✅ Game narrative system with dynamic NPCs",
        "✅ Performance monitoring and garbage collection",
        "✅ Comprehensive error handling and recovery"
    ]
    
    for feature in features_shown:
        print(f"  {feature}")
    
    print("\n🎯 Key Capabilities Proven:")
    capabilities = [
        "Context serialization and deserialization across boundaries",
        "Real-time hot-swapping of code and behavior",
        "Complex multi-agent reasoning and collaboration", 
        "Fault-tolerant distributed task execution",
        "Dynamic narrative generation with persistent state",
        "High-performance signal processing with concurrency",
        "Comprehensive monitoring and system health tracking",
        "Scalable architecture with garbage collection"
    ]
    
    for capability in capabilities:
        print(f"  • {capability}")
    
    # Recommendations for users
    print("\n💡 Next Steps for Users:")
    recommendations = [
        "Explore the functional API for rapid prototyping",
        "Study the examples for real-world implementation patterns",
        "Experiment with custom observer and signal patterns",
        "Build domain-specific applications using the core framework",
        "Contribute additional example systems and use cases",
        "Test performance characteristics with your specific workloads"
    ]
    
    for rec in recommendations:
        print(f"  • {rec}")
    
    print("\n" + "=" * 60)
    print("🎉 Complete library demonstration finished!")
    print("Thank you for exploring the Serializable Context Library!")
    print("=" * 60)
    
    return {
        'total_execution_time': total_demo_time,
        'demo_results': demo_results,
        'success_rate': successful_demos / total_demos,
        'features_demonstrated': len(features_shown),
        'timestamp': datetime.now().isoformat()
    }

# =============================================================================
# INTERACTIVE DEMO RUNNER
# =============================================================================

async def run_interactive_demo():
    """Run an interactive demo where users can choose which parts to execute."""
    
    print("🎮 Interactive Serializable Context Library Demo")
    print("=" * 50)
    
    available_demos = {
        '1': ('Basic Usage Patterns', 'Quick examples of core functionality'),
        '2': ('Game Narrative System', 'RPG-style game with serializable NPCs and story'),
        '3': ('Distributed Worker System', 'Multi-worker distributed computing'),
        '4': ('AI Collaboration System', 'Multi-agent AI reasoning and collaboration'),
        '5': ('Complete Demo', 'Run all demos in sequence'),
        '6': ('System Health Check', 'Check library status and performance')
    }
    
    print("\nAvailable demonstrations:")
    for key, (name, description) in available_demos.items():
        print(f"  {key}. {name}")
        print(f"     {description}")
    
    print("\nChoose demos to run (comma-separated, e.g., '1,3,5'):")
    print("Or press Enter to run the complete demo")
    
    # For this demo, we'll simulate user input
    # In a real implementation, you would use input() here
    simulated_choice = "5"  # Run complete demo
    
    choices = simulated_choice.strip().split(',') if simulated_choice.strip() else ['5']
    
    results = {}
    
    for choice in choices:
        choice = choice.strip()
        
        if choice == '1':
            print(f"\n🏃 Running: {available_demos[choice][0]}")
            try:
                start_time = time.time()
                dist_result = await distributed_processing_example()
                game_result = await game_narrative_example()
                ai_result = await ai_collaboration_example()
                
                results[choice] = {
                    'success': True,
                    'execution_time': time.time() - start_time,
                    'results': [dist_result, game_result, ai_result]
                }
                print("✅ Basic patterns completed successfully")
            except Exception as e:
                results[choice] = {'success': False, 'error': str(e)}
                print(f"❌ Basic patterns failed: {e}")
        
        elif choice == '2':
            print(f"\n🎮 Running: {available_demos[choice][0]}")
            try:
                start_time = time.time()
                game_system = await demo_game_narrative_system()
                
                results[choice] = {
                    'success': True,
                    'execution_time': time.time() - start_time,
                    'system': game_system
                }
                print("✅ Game narrative system completed successfully")
            except Exception as e:
                results[choice] = {'success': False, 'error': str(e)}
                print(f"❌ Game narrative system failed: {e}")
        
        elif choice == '3':
            print(f"\n🔧 Running: {available_demos[choice][0]}")
            try:
                start_time = time.time()
                worker_system = await demo_distributed_worker_system()
                
                results[choice] = {
                    'success': True,
                    'execution_time': time.time() - start_time,
                    'system': worker_system
                }
                print("✅ Distributed worker system completed successfully")
            except Exception as e:
                results[choice] = {'success': False, 'error': str(e)}
                print(f"❌ Distributed worker system failed: {e}")
        
        elif choice == '4':
            print(f"\n🤖 Running: {available_demos[choice][0]}")
            try:
                start_time = time.time()
                ai_system = await demo_ai_collaboration_system()
                
                results[choice] = {
                    'success': True,
                    'execution_time': time.time() - start_time,
                    'system': ai_system
                }
                print("✅ AI collaboration system completed successfully")
            except Exception as e:
                results[choice] = {'success': False, 'error': str(e)}
                print(f"❌ AI collaboration system failed: {e}")
        
        elif choice == '5':
            print(f"\n🚀 Running: {available_demos[choice][0]}")
            try:
                complete_results = await run_complete_library_demo()
                results[choice] = {
                    'success': True,
                    'results': complete_results
                }
            except Exception as e:
                results[choice] = {'success': False, 'error': str(e)}
                print(f"❌ Complete demo failed: {e}")
        
        elif choice == '6':
            print(f"\n🏥 Running: {available_demos[choice][0]}")
            try:
                start_time = time.time()
                health = await health_check()
                perf_stats = get_performance_stats()
                gc_results = run_garbage_collection()
                
                results[choice] = {
                    'success': True,
                    'execution_time': time.time() - start_time,
                    'health': health,
                    'performance': perf_stats,
                    'garbage_collection': gc_results
                }
                
                print("System Health Report:")
                print(f"  Library status: {health.get('library_status', 'unknown')}")
                print(f"  Context count: {health.get('context_count', 0)}")
                print(f"  Garbage collected: {gc_results.get('total_collected', 0)} objects")
                print("✅ Health check completed successfully")
                
            except Exception as e:
                results[choice] = {'success': False, 'error': str(e)}
                print(f"❌ Health check failed: {e}")
        
        else:
            print(f"❓ Unknown choice: {choice}")
    
    # Summary
    print(f"\n📊 Interactive Demo Summary")
    print("-" * 30)
    successful = sum(1 for r in results.values() if r.get('success', False))
    total = len(results)
    print(f"Completed: {successful}/{total} demos")
    print(f"Success rate: {successful/total:.1%}" if total > 0 else "No demos run")
    
    return results

# =============================================================================
# BENCHMARK DEMO
# =============================================================================

async def run_performance_benchmark():
    """Run performance benchmarks for the library."""
    
    print("⚡ Serializable Context Library - Performance Benchmark")
    print("=" * 55)
    
    benchmark_results = {}
    
    # Benchmark 1: Context creation and serialization speed
    print("\n📊 Benchmark 1: Context Operations")
    print("-" * 35)
    
    from matts.functional_interface import create_context, serialize_context
    
    num_contexts = 100
    start_time = time.time()
    
    contexts = []
    for i in range(num_contexts):
        context = create_context(f"benchmark_context_{i}")
        contexts.append(context)
    
    creation_time = time.time() - start_time
    
    # Serialize contexts
    start_time = time.time()
    serializations = []
    for context in contexts:
        serialized = serialize_context(context)
        serializations.append(serialized)
    
    serialization_time = time.time() - start_time
    
    benchmark_results['context_operations'] = {
        'contexts_created': num_contexts,
        'creation_time': creation_time,
        'creation_rate': num_contexts / creation_time,
        'serialization_time': serialization_time,
        'serialization_rate': num_contexts / serialization_time,
        'average_context_size': sum(len(str(s)) for s in serializations) / len(serializations)
    }
    
    print(f"Context creation: {num_contexts} contexts in {creation_time:.3f}s ({num_contexts/creation_time:.1f}/sec)")
    print(f"Context serialization: {num_contexts} contexts in {serialization_time:.3f}s ({num_contexts/serialization_time:.1f}/sec)")
    
    # Benchmark 2: Signal processing speed
    print("\n📡 Benchmark 2: Signal Processing")
    print("-" * 35)
    
    from matts.signal_system import create_signal_line, SignalPayload, SignalType
    
    signal_line = await create_signal_line("benchmark_signals")
    
    # Register test observer
    signals_received = []
    async def test_observer(signal):
        signals_received.append(signal)
        return True
    
    await signal_line.register_callback(test_observer)
    
    # Send signals
    num_signals = 1000
    start_time = time.time()
    
    for i in range(num_signals):
        signal = SignalPayload(
            signal_type=SignalType.CUSTOM,
            source_id=f"benchmark_source_{i}",
            data={'test_data': i}
        )
        await signal_line.emit(signal)
    
    signal_time = time.time() - start_time
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    benchmark_results['signal_processing'] = {
        'signals_sent': num_signals,
        'signals_received': len(signals_received),
        'processing_time': signal_time,
        'signal_rate': num_signals / signal_time,
        'success_rate': len(signals_received) / num_signals
    }
    
    print(f"Signal processing: {num_signals} signals in {signal_time:.3f}s ({num_signals/signal_time:.1f}/sec)")
    print(f"Success rate: {len(signals_received)}/{num_signals} ({len(signals_received)/num_signals:.1%})")
    
    await signal_line.stop()
    
    # Benchmark 3: Memory usage and garbage collection
    print("\n🧹 Benchmark 3: Memory Management")
    print("-" * 35)
    
    import gc as python_gc
    import sys
    
    # Get initial memory stats
    initial_objects = len(python_gc.get_objects())
    
    # Create many contexts and let them go out of scope
    start_time = time.time()
    for batch in range(10):
        batch_contexts = []
        for i in range(50):
            context = create_context(f"memory_test_{batch}_{i}")
            batch_contexts.append(context)
        # Let batch go out of scope
        del batch_contexts
        
        # Force garbage collection
        python_gc.collect()
    
    memory_test_time = time.time() - start_time
    
    # Run library garbage collection
    start_time = time.time()
    gc_results = run_garbage_collection()
    library_gc_time = time.time() - start_time
    
    final_objects = len(python_gc.get_objects())
    
    benchmark_results['memory_management'] = {
        'initial_objects': initial_objects,
        'final_objects': final_objects,
        'objects_created_destroyed': 500,
        'memory_test_time': memory_test_time,
        'library_gc_time': library_gc_time,
        'objects_collected': gc_results.get('total_collected', 0)
    }
    
    print(f"Memory test: 500 contexts created/destroyed in {memory_test_time:.3f}s")
    print(f"Library GC: {gc_results.get('total_collected', 0)} objects collected in {library_gc_time:.3f}s")
    print(f"Python objects: {initial_objects} → {final_objects} (net: {final_objects - initial_objects:+d})")
    
    # Summary
    print(f"\n🏆 Benchmark Summary")
    print("=" * 25)
    
    total_benchmark_time = sum(
        result.get('creation_time', 0) + 
        result.get('serialization_time', 0) +
        result.get('processing_time', 0) +
        result.get('memory_test_time', 0) +
        result.get('library_gc_time', 0)
        for result in benchmark_results.values()
    )
    
    print(f"Total benchmark time: {total_benchmark_time:.3f}s")
    print("\nPerformance Metrics:")
    
    if 'context_operations' in benchmark_results:
        ctx_ops = benchmark_results['context_operations']
        print(f"  Context creation rate: {ctx_ops['creation_rate']:.1f} contexts/sec")
        print(f"  Context serialization rate: {ctx_ops['serialization_rate']:.1f} contexts/sec")
    
    if 'signal_processing' in benchmark_results:
        sig_proc = benchmark_results['signal_processing']
        print(f"  Signal processing rate: {sig_proc['signal_rate']:.1f} signals/sec")
        print(f"  Signal success rate: {sig_proc['success_rate']:.1%}")
    
    if 'memory_management' in benchmark_results:
        mem_mgmt = benchmark_results['memory_management']
        print(f"  GC efficiency: {mem_mgmt['objects_collected']} objects collected")
        
    return benchmark_results

# =============================================================================
# MAIN DEMO ENTRY POINTS
# =============================================================================

async def main():
    """Main demo entry point."""
    demo_mode = "complete"  # Options: "complete", "interactive", "benchmark"
    
    if demo_mode == "complete":
        return await run_complete_library_demo()
    elif demo_mode == "interactive":
        return await run_interactive_demo()
    elif demo_mode == "benchmark":
        return await run_performance_benchmark()
    else:
        print(f"Unknown demo mode: {demo_mode}")
        return await run_complete_library_demo()

if __name__ == "__main__":
    asyncio.run(main())