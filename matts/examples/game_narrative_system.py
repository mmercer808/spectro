#!/usr/bin/env python3
"""
Game Narrative System Example

Demonstrates using serializable contexts for a dynamic game narrative system
where story state, character progression, and world events are serializable
and can be transmitted between game clients, saved/loaded, and modified
at runtime.
"""

import asyncio
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# matts.functional_interface
from matts import (
    create_context, get_context, transmit_context, emit_signal,
    update_context_data, get_context_data, create_generator_composer,
    add_relationship, hot_swap_context_version
)
# matts.signal_system
from matts import SignalType, SignalPriority
# matts.graph_system
from matts import RelationshipType
# matts.generator_system
from matts import GeneratorCompositionPattern

# =============================================================================
# GAME ENUMS AND DATA STRUCTURES
# =============================================================================

class GameEventType(Enum):
    """Types of game events."""
    PLAYER_ACTION = "player_action"
    NPC_INTERACTION = "npc_interaction"
    STORY_PROGRESSION = "story_progression"
    WORLD_CHANGE = "world_change"
    QUEST_UPDATE = "quest_update"
    ITEM_ACQUISITION = "item_acquisition"
    COMBAT_EVENT = "combat_event"

class QuestStatus(Enum):
    """Quest completion status."""
    NOT_STARTED = "not_started"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class GameCharacter:
    """Game character data."""
    character_id: str
    name: str
    level: int = 1
    health: int = 100
    inventory: List[str] = field(default_factory=list)
    location: str = "starting_village"
    experience: int = 0
    attributes: Dict[str, int] = field(default_factory=dict)

@dataclass
class Quest:
    """Quest definition."""
    quest_id: str
    title: str
    description: str
    status: QuestStatus = QuestStatus.NOT_STARTED
    requirements: List[str] = field(default_factory=list)
    rewards: List[str] = field(default_factory=list)
    progress: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GameEvent:
    """Game event data."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: GameEventType = GameEventType.PLAYER_ACTION
    timestamp: datetime = field(default_factory=datetime.now)
    source_character: Optional[str] = None
    target_character: Optional[str] = None
    location: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# GAME NARRATIVE SYSTEM
# =============================================================================

class GameNarrativeSystem:
    """
    Complete game narrative system using serializable contexts.
    
    Features:
    - Player state serialization and transmission
    - Dynamic story progression with hot-swapping
    - NPC behavior contexts that can be modified at runtime
    - Quest system with serializable state
    - World events that trigger context changes
    """
    
    def __init__(self, game_id: str = None):
        self.game_id = game_id or f"game_{uuid.uuid4()}"
        
        # Core contexts
        self.world_context_id: Optional[str] = None
        self.player_contexts: Dict[str, str] = {}  # player_id -> context_id
        self.npc_contexts: Dict[str, str] = {}     # npc_id -> context_id
        self.quest_context_id: Optional[str] = None
        
        # Game state
        self.active_players: Dict[str, GameCharacter] = {}
        self.game_events: List[GameEvent] = []
        self.world_state: Dict[str, Any] = {}
        
        # Narrative generators
        self.story_composer = None
        
        # System stats
        self.created_at = datetime.now()
        self.events_processed = 0
    
    async def initialize_game(self) -> Dict[str, str]:
        """Initialize the game narrative system."""
        print(f"🎮 Initializing Game Narrative System: {self.game_id}")
        
        # Create world context
        world_context = create_context(f"{self.game_id}_world")
        self.world_context_id = world_context.context_id
        
        # Initialize world state
        await update_context_data(self.world_context_id, {
            'game_id': self.game_id,
            'world_time': 0,
            'weather': 'clear',
            'active_events': [],
            'locations': {
                'starting_village': {
                    'name': 'Peaceful Village',
                    'description': 'A quiet village where adventures begin',
                    'npcs': ['village_elder', 'merchant'],
                    'available_quests': ['gather_herbs', 'find_lost_cat']
                },
                'dark_forest': {
                    'name': 'Dark Forest',
                    'description': 'A mysterious forest full of ancient secrets',
                    'npcs': ['forest_guardian'],
                    'dangers': ['wolves', 'ancient_spirits']
                }
            },
            'global_flags': {}
        })
        
        # Create quest context
        quest_context = create_context(f"{self.game_id}_quests")
        self.quest_context_id = quest_context.context_id
        
        # Initialize quest system
        await update_context_data(self.quest_context_id, {
            'available_quests': {
                'gather_herbs': Quest(
                    quest_id='gather_herbs',
                    title='Gather Healing Herbs',
                    description='The village healer needs 5 healing herbs from the forest',
                    requirements=['visit_forest', 'collect_herbs:5'],
                    rewards=['experience:50', 'gold:100', 'potion:health']
                ).__dict__,
                'find_lost_cat': Quest(
                    quest_id='find_lost_cat',
                    title='Find the Lost Cat',
                    description='Help the elderly woman find her missing cat',
                    requirements=['talk_to_elder', 'search_village', 'find_cat'],
                    rewards=['experience:25', 'reputation:village:+10']
                ).__dict__
            },
            'completed_quests': {},
            'quest_chains': {
                'hero_journey': ['gather_herbs', 'find_lost_cat', 'dark_forest_mystery']
            }
        })
        
        # Set up story generator composition
        self.story_composer = create_generator_composer(self.world_context_id)
        await self._setup_narrative_generators()
        
        # Create relationships between contexts
        add_relationship(
            self.world_context_id, 
            self.quest_context_id,
            RelationshipType.CONTAINS,
            {'relationship': 'world_contains_quests'}
        )
        
        print(f"✅ Game initialized with world context: {self.world_context_id}")
        
        return {
            'world_context_id': self.world_context_id,
            'quest_context_id': self.quest_context_id,
            'game_id': self.game_id
        }
    
    async def create_player(self, player_id: str, character_name: str) -> GameCharacter:
        """Create a new player character with serializable context."""
        print(f"👤 Creating player: {character_name} ({player_id})")
        
        # Create player context
        player_context = create_context(f"{self.game_id}_player_{player_id}")
        self.player_contexts[player_id] = player_context.context_id
        
        # Create character
        character = GameCharacter(
            character_id=player_id,
            name=character_name,
            attributes={'strength': 10, 'dexterity': 10, 'intelligence': 10}
        )
        
        # Initialize player context data
        await update_context_data(player_context.context_id, {
            'character': character.__dict__,
            'session_data': {
                'login_time': datetime.now().isoformat(),
                'last_action': None,
                'action_history': []
            },
            'narrative_state': {
                'current_story_beat': 'introduction',
                'dialogue_history': [],
                'choices_made': [],
                'reputation': {'village': 0, 'forest_guardians': 0}
            },
            'gameplay_flags': {
                'tutorial_completed': False,
                'first_quest_given': False,
                'can_enter_forest': False
            }
        })
        
        # Create relationship with world
        add_relationship(
            self.world_context_id,
            player_context.context_id,
            RelationshipType.CONTAINS,
            {'relationship': 'world_contains_player', 'player_id': player_id}
        )
        
        self.active_players[player_id] = character
        
        # Emit player creation event
        await emit_signal(
            SignalType.CUSTOM,
            source_context_id=player_context.context_id,
            data={
                'event_type': 'player_created',
                'player_id': player_id,
                'character_name': character_name
            }
        )
        
        print(f"✅ Player created with context: {player_context.context_id}")
        return character
    
    async def create_npc(self, npc_id: str, npc_data: Dict[str, Any]) -> str:
        """Create an NPC with serializable AI behavior context."""
        print(f"🤖 Creating NPC: {npc_id}")
        
        # Create NPC context
        npc_context = create_context(f"{self.game_id}_npc_{npc_id}")
        self.npc_contexts[npc_id] = npc_context.context_id
        
        # Initialize NPC behavior and dialogue
        await update_context_data(npc_context.context_id, {
            'npc_id': npc_id,
            'name': npc_data.get('name', npc_id),
            'personality': npc_data.get('personality', 'friendly'),
            'dialogue_tree': npc_data.get('dialogue_tree', {
                'greeting': 'Hello, traveler!',
                'quest_offer': 'I have a task for you...',
                'farewell': 'Safe travels!'
            }),
            'behavior_state': {
                'current_mood': 'neutral',
                'last_interaction': None,
                'interaction_count': 0,
                'relationship_with_players': {}
            },
            'ai_behavior': {
                'aggression': npc_data.get('aggression', 0),
                'helpfulness': npc_data.get('helpfulness', 5),
                'memory_span': npc_data.get('memory_span', 10),
                'decision_patterns': npc_data.get('decision_patterns', [])
            },
            'location': npc_data.get('location', 'starting_village'),
            'schedule': npc_data.get('schedule', {
                'morning': 'village_square',
                'afternoon': 'market',
                'evening': 'tavern'
            })
        })
        
        # Create relationship with world
        add_relationship(
            self.world_context_id,
            npc_context.context_id,
            RelationshipType.CONTAINS,
            {'relationship': 'world_contains_npc', 'npc_id': npc_id}
        )
        
        print(f"✅ NPC created with context: {npc_context.context_id}")
        return npc_context.context_id
    
    async def process_player_action(self, player_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """Process a player action and update narrative state."""
        if player_id not in self.player_contexts:
            raise ValueError(f"Player {player_id} not found")
        
        player_context_id = self.player_contexts[player_id]
        
        print(f"⚡ Processing action for {player_id}: {action.get('type', 'unknown')}")
        
        # Create game event
        event = GameEvent(
            event_type=GameEventType.PLAYER_ACTION,
            source_character=player_id,
            location=action.get('location'),
            data=action
        )
        self.game_events.append(event)
        self.events_processed += 1
        
        # Get current player data
        player_data = get_context_data(player_context_id)
        current_character = player_data.get('character', {})
        
        # Process different action types
        action_result = {'success': False, 'message': '', 'effects': []}
        
        action_type = action.get('type')
        
        if action_type == 'move':
            action_result = await self._process_move_action(
                player_id, player_context_id, action, current_character
            )
        elif action_type == 'interact':
            action_result = await self._process_interact_action(
                player_id, player_context_id, action, current_character
            )
        elif action_type == 'quest_action':
            action_result = await self._process_quest_action(
                player_id, player_context_id, action, current_character
            )
        elif action_type == 'combat':
            action_result = await self._process_combat_action(
                player_id, player_context_id, action, current_character
            )
        
        # Update player context with action result
        await update_context_data(player_context_id, {
            'session_data.last_action': {
                'action': action,
                'result': action_result,
                'timestamp': datetime.now().isoformat()
            }
        })
        
        # Emit action processed signal
        await emit_signal(
            SignalType.CUSTOM,
            source_context_id=player_context_id,
            data={
                'event_type': 'action_processed',
                'player_id': player_id,
                'action': action,
                'result': action_result
            }
        )
        
        return action_result
    
    async def save_game_state(self, player_id: str) -> Dict[str, Any]:
        """Save complete game state for a player (serializable context transmission)."""
        if player_id not in self.player_contexts:
            raise ValueError(f"Player {player_id} not found")
        
        print(f"💾 Saving game state for player: {player_id}")
        
        player_context_id = self.player_contexts[player_id]
        
        # Create save context
        save_context = create_context(f"{self.game_id}_save_{player_id}_{datetime.now().timestamp()}")
        
        # Transmit player context to save context
        player_transmission = await transmit_context(player_context_id, save_context.context_id)
        
        # Transmit world state to save context
        world_transmission = await transmit_context(self.world_context_id, save_context.context_id)
        
        # Transmit quest state to save context
        quest_transmission = await transmit_context(self.quest_context_id, save_context.context_id)
        
        # Add save metadata
        await update_context_data(save_context.context_id, {
            'save_metadata': {
                'save_time': datetime.now().isoformat(),
                'game_id': self.game_id,
                'player_id': player_id,
                'save_version': '1.0',
                'events_processed': self.events_processed
            },
            'transmission_results': {
                'player_transmission': player_transmission,
                'world_transmission': world_transmission,
                'quest_transmission': quest_transmission
            }
        })
        
        save_result = {
            'save_context_id': save_context.context_id,
            'save_time': datetime.now().isoformat(),
            'success': all([
                player_transmission.get('handled', False),
                world_transmission.get('handled', False),
                quest_transmission.get('handled', False)
            ])
        }
        
        print(f"✅ Game state saved: {save_result}")
        return save_result
    
    async def modify_npc_behavior_runtime(self, npc_id: str, 
                                        behavior_changes: Dict[str, Any]) -> bool:
        """Hot-swap NPC behavior at runtime using context versioning."""
        if npc_id not in self.npc_contexts:
            return False
        
        npc_context_id = self.npc_contexts[npc_id]
        
        print(f"🔄 Hot-swapping NPC behavior for: {npc_id}")
        
        # Create new behavior version
        version_id = f"behavior_v{datetime.now().timestamp()}"
        
        success = hot_swap_context_version(
            npc_context_id,
            None,  # Use root node
            version_id,
            {'context_data': behavior_changes}
        )
        
        if success:
            print(f"✅ NPC {npc_id} behavior updated to version: {version_id}")
            
            # Emit behavior change signal
            await emit_signal(
                SignalType.CUSTOM,
                source_context_id=npc_context_id,
                data={
                    'event_type': 'npc_behavior_changed',
                    'npc_id': npc_id,
                    'new_version': version_id,
                    'changes': behavior_changes
                }
            )
        
        return success
    
    async def _setup_narrative_generators(self):
        """Set up story generation using generator composition."""
        # This would set up the actual story generation logic
        # For demo purposes, we'll create a simple pattern
        
        self.story_composer.register_generator_factory('intro_sequence', self._create_intro_generator)
        self.story_composer.register_generator_factory('quest_generator', self._create_quest_generator)
        self.story_composer.register_generator_factory('dialogue_generator', self._create_dialogue_generator)
        
        # Create story progression pattern
        self.story_composer.create_composition_pattern(
            'main_story',
            GeneratorCompositionPattern.SEQUENTIAL,
            [
                {'name': 'intro_sequence', 'config': {'length': 'short'}},
                {'name': 'quest_generator', 'config': {'difficulty': 'easy'}},
                {'name': 'dialogue_generator', 'config': {'style': 'friendly'}}
            ]
        )
    
    def _create_intro_generator(self, config):
        """Create intro sequence generator."""
        def intro_generator():
            length = config.get('length', 'medium')
            if length == 'short':
                yield "Welcome to the adventure! Your journey begins now."
            else:
                yield "A new hero emerges in the peaceful village..."
                yield "The villagers look up as you approach..."
                yield "Your destiny awaits!"
        return intro_generator()
    
    def _create_quest_generator(self, config):
        """Create quest generation logic."""
        def quest_generator():
            difficulty = config.get('difficulty', 'medium')
            if difficulty == 'easy':
                yield {'quest_type': 'fetch', 'complexity': 'simple'}
            else:
                yield {'quest_type': 'mystery', 'complexity': 'branching'}
        return quest_generator()
    
    def _create_dialogue_generator(self, config):
        """Create dialogue generation logic."""
        def dialogue_generator():
            style = config.get('style', 'neutral')
            if style == 'friendly':
                yield "Hello there, brave adventurer!"
            else:
                yield "Greetings, traveler."
        return dialogue_generator()
    
    async def _process_move_action(self, player_id: str, context_id: str, 
                                  action: Dict[str, Any], character: Dict[str, Any]) -> Dict[str, Any]:
        """Process player movement action."""
        destination = action.get('destination')
        
        # Update character location
        await update_context_data(context_id, {
            'character.location': destination
        })
        
        return {
            'success': True,
            'message': f"Moved to {destination}",
            'effects': [f'location_changed:{destination}']
        }
    
    async def _process_interact_action(self, player_id: str, context_id: str,
                                     action: Dict[str, Any], character: Dict[str, Any]) -> Dict[str, Any]:
        """Process player interaction action."""
        target = action.get('target')
        
        if target in self.npc_contexts:
            # Interact with NPC
            npc_context_id = self.npc_contexts[target]
            npc_data = get_context_data(npc_context_id)
            
            greeting = npc_data.get('dialogue_tree', {}).get('greeting', 'Hello!')
            
            return {
                'success': True,
                'message': f'{target} says: "{greeting}"',
                'effects': ['dialogue_started']
            }
        
        return {
            'success': False,
            'message': f"Cannot interact with {target}",
            'effects': []
        }
    
    async def _process_quest_action(self, player_id: str, context_id: str,
                                   action: Dict[str, Any], character: Dict[str, Any]) -> Dict[str, Any]:
        """Process quest-related action."""
        quest_id = action.get('quest_id')
        quest_action = action.get('quest_action')  # 'accept', 'complete', 'abandon'
        
        quest_data = get_context_data(self.quest_context_id)
        available_quests = quest_data.get('available_quests', {})
        
        if quest_id in available_quests and quest_action == 'accept':
            # Accept quest
            await update_context_data(context_id, {
                f'active_quests.{quest_id}': available_quests[quest_id]
            })
            
            return {
                'success': True,
                'message': f"Quest '{quest_id}' accepted!",
                'effects': [f'quest_accepted:{quest_id}']
            }
        
        return {
            'success': False,
            'message': f"Cannot perform quest action: {quest_action}",
            'effects': []
        }
    
    async def _process_combat_action(self, player_id: str, context_id: str,
                                    action: Dict[str, Any], character: Dict[str, Any]) -> Dict[str, Any]:
        """Process combat action."""
        target = action.get('target')
        attack_type = action.get('attack_type', 'basic')
        
        # Simple combat simulation
        damage = 10 if attack_type == 'basic' else 15
        
        return {
            'success': True,
            'message': f"Attacked {target} for {damage} damage!",
            'effects': [f'damage_dealt:{damage}', f'combat_with:{target}']
        }
    
    async def get_game_statistics(self) -> Dict[str, Any]:
        """Get comprehensive game statistics."""
        return {
            'game_id': self.game_id,
            'created_at': self.created_at.isoformat(),
            'active_players': len(self.active_players),
            'total_npcs': len(self.npc_contexts),
            'events_processed': self.events_processed,
            'contexts': {
                'world_context': self.world_context_id,
                'quest_context': self.quest_context_id,
                'player_contexts': len(self.player_contexts),
                'npc_contexts': len(self.npc_contexts)
            },
            'recent_events': [
                {
                    'event_id': event.event_id,
                    'type': event.event_type.value,
                    'timestamp': event.timestamp.isoformat(),
                    'source': event.source_character
                }
                for event in self.game_events[-10:]  # Last 10 events
            ]
        }

# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

async def demo_game_narrative_system():
    """Demonstrate the game narrative system."""
    print("\n🎮 Game Narrative System Demo")
    print("=" * 50)
    
    # Create game system
    game = GameNarrativeSystem("demo_rpg")
    
    # Initialize game
    await game.initialize_game()
    
    # Create player
    hero = await game.create_player("player1", "Eldric the Brave")
    print(f"Created hero: {hero.name}")
    
    # Create NPCs
    await game.create_npc("village_elder", {
        'name': 'Elder Wiseman',
        'personality': 'wise',
        'dialogue_tree': {
            'greeting': 'Welcome, young hero. The village needs your help.',
            'quest_offer': 'Dark creatures stir in the forest. Will you investigate?',
            'farewell': 'May the light guide your path.'
        },
        'helpfulness': 8,
        'location': 'starting_village'
    })
    
    await game.create_npc("merchant", {
        'name': 'Trader Bob',
        'personality': 'greedy',
        'dialogue_tree': {
            'greeting': 'Looking to buy some fine wares?',
            'quest_offer': 'I need someone to retrieve my stolen goods...',
            'farewell': 'Come back when you have more coin!'
        },
        'helpfulness': 3,
        'location': 'starting_village'
    })
    
    # Process some player actions
    print("\n📋 Processing Player Actions:")
    
    # Move action
    move_result = await game.process_player_action("player1", {
        'type': 'move',
        'destination': 'village_square',
        'location': 'starting_village'
    })
    print(f"Move result: {move_result}")
    
    # Interaction action
    interact_result = await game.process_player_action("player1", {
        'type': 'interact',
        'target': 'village_elder',
        'location': 'village_square'
    })
    print(f"Interact result: {interact_result}")
    
    # Quest action
    quest_result = await game.process_player_action("player1", {
        'type': 'quest_action',
        'quest_id': 'gather_herbs',
        'quest_action': 'accept',
        'location': 'village_square'
    })
    print(f"Quest result: {quest_result}")
    
    # Demonstrate hot-swapping NPC behavior
    print("\n🔄 Demonstrating Hot-Swap NPC Behavior:")
    behavior_change = await game.modify_npc_behavior_runtime("village_elder", {
        'ai_behavior.helpfulness': 10,
        'dialogue_tree.greeting': 'Ah, the chosen hero returns! I sense great power in you.',
        'behavior_state.current_mood': 'excited'
    })
    print(f"Behavior change success: {behavior_change}")
    
    # Save game state
    print("\n💾 Saving Game State:")
    save_result = await game.save_game_state("player1")
    print(f"Save result: {save_result}")
    
    # Get game statistics
    print("\n📊 Game Statistics:")
    stats = await game.get_game_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")
    
    return game

if __name__ == "__main__":
    # Add parent directory to Python path when running directly
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    asyncio.run(demo_game_narrative_system())