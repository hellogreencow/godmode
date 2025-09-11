"""
Advanced cognitive memory system for hierarchical reasoning.

This module implements a sophisticated memory architecture that includes
working memory, long-term memory, and episodic memory with attention mechanisms.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

from pydantic import Field

from godmode.models.core import (
    BaseGodModeModel,
    MemoryState,
    TensorData,
    AttentionWeights,
    AttentionMechanism,
)


logger = logging.getLogger(__name__)


class MemoryItem(BaseGodModeModel):
    """Individual item stored in memory."""
    
    content: Dict[str, Any]
    embedding: Optional[TensorData] = None
    
    # Metadata
    memory_type: str  # working, long_term, episodic
    importance: float = Field(0.5, ge=0.0, le=1.0)
    recency: float = Field(1.0, ge=0.0, le=1.0)
    frequency: int = Field(1, ge=1)
    
    # Temporal information
    first_stored: float = Field(default_factory=time.time)
    last_accessed: float = Field(default_factory=time.time)
    
    # Associations
    related_items: Set[UUID] = Field(default_factory=set)
    tags: Set[str] = Field(default_factory=set)
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.frequency += 1
        self.recency = 1.0  # Reset recency
    
    def decay_recency(self, decay_rate: float = 0.01):
        """Decay recency over time."""
        time_since_access = time.time() - self.last_accessed
        self.recency *= np.exp(-decay_rate * time_since_access)
    
    def get_activation(self) -> float:
        """Calculate activation level based on importance, recency, and frequency."""
        return (self.importance * 0.4 + 
                self.recency * 0.4 + 
                min(1.0, self.frequency / 10.0) * 0.2)


class AttentionModule(torch.nn.Module):
    """Neural attention module for memory retrieval."""
    
    def __init__(
        self,
        embedding_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        # Multi-head attention layers
        self.query_projection = torch.nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = torch.nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = torch.nn.Linear(embedding_dim, embedding_dim)
        self.output_projection = torch.nn.Linear(embedding_dim, embedding_dim)
        
        # Dropout and normalization
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(embedding_dim)
        
    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of attention module.
        
        Args:
            query: Query tensor [batch_size, seq_len, embedding_dim]
            keys: Key tensor [batch_size, memory_size, embedding_dim]
            values: Value tensor [batch_size, memory_size, embedding_dim]
            mask: Optional attention mask
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        batch_size, seq_len, _ = query.shape
        memory_size = keys.shape[1]
        
        # Project to multi-head space
        Q = self.query_projection(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key_projection(keys).view(batch_size, memory_size, self.num_heads, self.head_dim)
        V = self.value_projection(values).view(batch_size, memory_size, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        K = K.transpose(1, 2)  # [batch_size, num_heads, memory_size, head_dim]
        V = V.transpose(1, 2)  # [batch_size, num_heads, memory_size, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embedding_dim
        )
        
        # Final projection
        output = self.output_projection(attended)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + query)
        
        # Average attention weights across heads
        avg_attention = attention_weights.mean(dim=1)
        
        return output, avg_attention


class WorkingMemory:
    """Working memory with limited capacity and attention mechanisms."""
    
    def __init__(
        self,
        capacity: int = 7,  # Miller's magic number
        embedding_dim: int = 512,
        decay_rate: float = 0.1,
    ):
        self.capacity = capacity
        self.embedding_dim = embedding_dim
        self.decay_rate = decay_rate
        
        # Memory storage
        self.items: Dict[UUID, MemoryItem] = {}
        self.access_order = deque(maxlen=capacity)
        
        # Attention mechanism
        self.attention = AttentionModule(embedding_dim)
        
        # Statistics
        self.total_items_stored = 0
        self.total_retrievals = 0
        self.cache_hits = 0
        
    def store(self, content: Dict[str, Any], importance: float = 0.5) -> UUID:
        """Store item in working memory."""
        # Create memory item
        item = MemoryItem(
            content=content,
            memory_type="working",
            importance=importance,
        )
        
        # Generate embedding if content is suitable
        if isinstance(content.get("text"), str):
            item.embedding = self._generate_embedding(content["text"])
        
        # Check capacity and evict if necessary
        if len(self.items) >= self.capacity:
            self._evict_least_important()
        
        # Store item
        self.items[item.id] = item
        self.access_order.append(item.id)
        self.total_items_stored += 1
        
        logger.debug(f"Stored item {item.id} in working memory")
        return item.id
    
    def retrieve(self, query: Union[str, Dict[str, Any]], top_k: int = 3) -> List[MemoryItem]:
        """Retrieve items from working memory based on query."""
        self.total_retrievals += 1
        
        if not self.items:
            return []
        
        # Generate query embedding
        if isinstance(query, str):
            query_embedding = self._generate_embedding(query)
        else:
            query_text = query.get("text", str(query))
            query_embedding = self._generate_embedding(query_text)
        
        # Calculate similarities
        similarities = []
        for item_id, item in self.items.items():
            if item.embedding is not None:
                similarity = self._calculate_similarity(query_embedding, item.embedding)
                # Boost by activation level
                boosted_similarity = similarity * (1 + item.get_activation())
                similarities.append((boosted_similarity, item))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        results = [item for _, item in similarities[:top_k]]
        
        # Update access statistics
        for item in results:
            item.update_access()
            if item.id in self.access_order:
                self.access_order.remove(item.id)
            self.access_order.append(item.id)
        
        return results
    
    def update(self, item_id: UUID, content: Dict[str, Any]) -> bool:
        """Update existing item in working memory."""
        if item_id not in self.items:
            return False
        
        item = self.items[item_id]
        item.content.update(content)
        item.update_access()
        
        # Regenerate embedding if text content changed
        if "text" in content:
            item.embedding = self._generate_embedding(content["text"])
        
        return True
    
    def remove(self, item_id: UUID) -> bool:
        """Remove item from working memory."""
        if item_id not in self.items:
            return False
        
        del self.items[item_id]
        if item_id in self.access_order:
            self.access_order.remove(item_id)
        
        return True
    
    def get_state(self) -> Dict[str, Any]:
        """Get current working memory state."""
        return {
            "capacity": self.capacity,
            "current_size": len(self.items),
            "utilization": len(self.items) / self.capacity,
            "items": {str(item_id): item.content for item_id, item in self.items.items()},
            "access_order": [str(item_id) for item_id in self.access_order],
        }
    
    def decay_all(self):
        """Apply decay to all items in working memory."""
        for item in self.items.values():
            item.decay_recency(self.decay_rate)
    
    def _evict_least_important(self):
        """Evict the least important item from working memory."""
        if not self.items:
            return
        
        # Find item with lowest activation
        min_activation = float('inf')
        evict_id = None
        
        for item_id, item in self.items.items():
            activation = item.get_activation()
            if activation < min_activation:
                min_activation = activation
                evict_id = item_id
        
        if evict_id:
            logger.debug(f"Evicting item {evict_id} with activation {min_activation}")
            self.remove(evict_id)
    
    def _generate_embedding(self, text: str) -> TensorData:
        """Generate embedding for text content."""
        # Simple bag-of-words embedding for now
        # In practice, this would use a pre-trained language model
        words = text.lower().split()
        vocab_size = 10000  # Simplified vocabulary
        
        # Create simple hash-based embedding
        embedding = np.zeros(self.embedding_dim)
        for word in words:
            word_hash = hash(word) % vocab_size
            embedding[word_hash % self.embedding_dim] += 1.0
        
        # Normalize
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
        
        return TensorData(
            data=torch.tensor(embedding, dtype=torch.float32),
            shape=list(embedding.shape),
            dtype="float32",
        )
    
    def _calculate_similarity(self, embedding1: TensorData, embedding2: TensorData) -> float:
        """Calculate cosine similarity between embeddings."""
        vec1 = embedding1.data.numpy().reshape(1, -1)
        vec2 = embedding2.data.numpy().reshape(1, -1)
        
        similarity = cosine_similarity(vec1, vec2)[0, 0]
        return float(similarity)


class LongTermMemory:
    """Long-term memory with persistent storage and hierarchical organization."""
    
    def __init__(
        self,
        embedding_dim: int = 512,
        consolidation_threshold: float = 0.7,
    ):
        self.embedding_dim = embedding_dim
        self.consolidation_threshold = consolidation_threshold
        
        # Memory storage organized by categories
        self.semantic_memory: Dict[str, Dict[UUID, MemoryItem]] = defaultdict(dict)
        self.procedural_memory: Dict[UUID, MemoryItem] = {}
        
        # Indexing for fast retrieval
        self.category_index: Dict[str, Set[UUID]] = defaultdict(set)
        self.tag_index: Dict[str, Set[UUID]] = defaultdict(set)
        
        # Statistics
        self.total_items = 0
        self.consolidation_events = 0
        
    def store(
        self,
        content: Dict[str, Any],
        category: str = "general",
        importance: float = 0.5,
        tags: Optional[Set[str]] = None,
    ) -> UUID:
        """Store item in long-term memory."""
        
        # Create memory item
        item = MemoryItem(
            content=content,
            memory_type="long_term",
            importance=importance,
            tags=tags or set(),
        )
        
        # Generate embedding
        if isinstance(content.get("text"), str):
            item.embedding = self._generate_embedding(content["text"])
        
        # Store in appropriate category
        self.semantic_memory[category][item.id] = item
        
        # Update indices
        self.category_index[category].add(item.id)
        for tag in item.tags:
            self.tag_index[tag].add(item.id)
        
        self.total_items += 1
        logger.debug(f"Stored item {item.id} in long-term memory category: {category}")
        
        return item.id
    
    def retrieve(
        self,
        query: Union[str, Dict[str, Any]],
        category: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        top_k: int = 10,
    ) -> List[MemoryItem]:
        """Retrieve items from long-term memory."""
        
        # Generate query embedding
        if isinstance(query, str):
            query_embedding = self._generate_embedding(query)
            query_text = query
        else:
            query_text = query.get("text", str(query))
            query_embedding = self._generate_embedding(query_text)
        
        # Collect candidate items
        candidates = set()
        
        if category:
            candidates.update(self.category_index.get(category, set()))
        else:
            for cat_items in self.category_index.values():
                candidates.update(cat_items)
        
        if tags:
            tag_candidates = set()
            for tag in tags:
                tag_candidates.update(self.tag_index.get(tag, set()))
            if candidates:
                candidates &= tag_candidates
            else:
                candidates = tag_candidates
        
        # Calculate similarities
        similarities = []
        for item_id in candidates:
            item = self._get_item_by_id(item_id)
            if item and item.embedding:
                similarity = self._calculate_similarity(query_embedding, item.embedding)
                # Boost by importance and activation
                boosted_similarity = similarity * (1 + item.importance) * (1 + item.get_activation())
                similarities.append((boosted_similarity, item))
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        results = [item for _, item in similarities[:top_k]]
        
        # Update access statistics
        for item in results:
            item.update_access()
        
        return results
    
    def consolidate_from_working_memory(self, working_memory: WorkingMemory):
        """Consolidate important items from working memory to long-term memory."""
        consolidated = 0
        
        for item_id, item in working_memory.items.items():
            # Check if item meets consolidation criteria
            if (item.importance >= self.consolidation_threshold or 
                item.frequency >= 3 or
                item.get_activation() >= 0.8):
                
                # Determine category based on content
                category = self._categorize_content(item.content)
                
                # Store in long-term memory
                self.store(
                    content=item.content,
                    category=category,
                    importance=item.importance,
                    tags=item.tags,
                )
                
                consolidated += 1
        
        self.consolidation_events += 1
        logger.info(f"Consolidated {consolidated} items from working memory")
        
        return consolidated
    
    def _get_item_by_id(self, item_id: UUID) -> Optional[MemoryItem]:
        """Get item by ID from any category."""
        for category_items in self.semantic_memory.values():
            if item_id in category_items:
                return category_items[item_id]
        
        if item_id in self.procedural_memory:
            return self.procedural_memory[item_id]
        
        return None
    
    def _categorize_content(self, content: Dict[str, Any]) -> str:
        """Automatically categorize content based on its properties."""
        text = content.get("text", "").lower()
        content_type = content.get("type", "").lower()
        
        # Simple keyword-based categorization
        if any(word in text for word in ["problem", "solution", "reasoning"]):
            return "reasoning"
        elif any(word in text for word in ["fact", "knowledge", "information"]):
            return "knowledge"
        elif any(word in text for word in ["procedure", "method", "algorithm"]):
            return "procedural"
        elif content_type:
            return content_type
        else:
            return "general"
    
    def _generate_embedding(self, text: str) -> TensorData:
        """Generate embedding for text content."""
        # Reuse the same embedding generation as WorkingMemory
        words = text.lower().split()
        vocab_size = 10000
        
        embedding = np.zeros(self.embedding_dim)
        for word in words:
            word_hash = hash(word) % vocab_size
            embedding[word_hash % self.embedding_dim] += 1.0
        
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
        
        return TensorData(
            data=torch.tensor(embedding, dtype=torch.float32),
            shape=list(embedding.shape),
            dtype="float32",
        )
    
    def _calculate_similarity(self, embedding1: TensorData, embedding2: TensorData) -> float:
        """Calculate cosine similarity between embeddings."""
        vec1 = embedding1.data.numpy().reshape(1, -1)
        vec2 = embedding2.data.numpy().reshape(1, -1)
        
        similarity = cosine_similarity(vec1, vec2)[0, 0]
        return float(similarity)


class EpisodicMemory:
    """Episodic memory for storing temporal sequences and experiences."""
    
    def __init__(self, max_episodes: int = 1000):
        self.max_episodes = max_episodes
        
        # Episode storage
        self.episodes: Dict[UUID, Dict[str, Any]] = {}
        self.episode_order = deque(maxlen=max_episodes)
        
        # Temporal indexing
        self.time_index: Dict[str, List[UUID]] = defaultdict(list)  # date -> episode_ids
        
    def store_episode(
        self,
        events: List[Dict[str, Any]],
        context: Dict[str, Any],
        importance: float = 0.5,
    ) -> UUID:
        """Store an episode (sequence of events) in episodic memory."""
        
        episode_id = uuid4()
        timestamp = time.time()
        
        episode = {
            "id": episode_id,
            "events": events,
            "context": context,
            "importance": importance,
            "timestamp": timestamp,
            "duration": self._calculate_episode_duration(events),
        }
        
        # Store episode
        self.episodes[episode_id] = episode
        self.episode_order.append(episode_id)
        
        # Update time index
        date_key = time.strftime("%Y-%m-%d", time.localtime(timestamp))
        self.time_index[date_key].append(episode_id)
        
        # Evict old episodes if necessary
        if len(self.episodes) > self.max_episodes:
            self._evict_oldest_episode()
        
        logger.debug(f"Stored episode {episode_id} with {len(events)} events")
        return episode_id
    
    def retrieve_episodes(
        self,
        query: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
        context_filter: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve episodes based on various criteria."""
        
        candidates = list(self.episodes.values())
        
        # Filter by date range
        if date_range:
            start_date, end_date = date_range
            start_timestamp = time.mktime(time.strptime(start_date, "%Y-%m-%d"))
            end_timestamp = time.mktime(time.strptime(end_date, "%Y-%m-%d"))
            
            candidates = [
                ep for ep in candidates
                if start_timestamp <= ep["timestamp"] <= end_timestamp
            ]
        
        # Filter by context
        if context_filter:
            candidates = [
                ep for ep in candidates
                if self._matches_context(ep["context"], context_filter)
            ]
        
        # Score by relevance if query provided
        if query:
            scored_episodes = []
            for episode in candidates:
                score = self._calculate_episode_relevance(episode, query)
                scored_episodes.append((score, episode))
            
            # Sort by score and return top-k
            scored_episodes.sort(key=lambda x: x[0], reverse=True)
            return [episode for _, episode in scored_episodes[:top_k]]
        else:
            # Sort by importance and recency
            candidates.sort(
                key=lambda ep: (ep["importance"], ep["timestamp"]),
                reverse=True
            )
            return candidates[:top_k]
    
    def _calculate_episode_duration(self, events: List[Dict[str, Any]]) -> float:
        """Calculate duration of an episode."""
        if len(events) < 2:
            return 0.0
        
        timestamps = [event.get("timestamp", 0) for event in events]
        return max(timestamps) - min(timestamps)
    
    def _evict_oldest_episode(self):
        """Evict the oldest episode to make room."""
        if self.episode_order:
            oldest_id = self.episode_order.popleft()
            if oldest_id in self.episodes:
                del self.episodes[oldest_id]
    
    def _matches_context(
        self, 
        episode_context: Dict[str, Any], 
        filter_context: Dict[str, Any]
    ) -> bool:
        """Check if episode context matches filter criteria."""
        for key, value in filter_context.items():
            if key not in episode_context:
                return False
            if episode_context[key] != value:
                return False
        return True
    
    def _calculate_episode_relevance(self, episode: Dict[str, Any], query: str) -> float:
        """Calculate relevance score of episode to query."""
        query_words = set(query.lower().split())
        
        # Score based on events
        event_score = 0.0
        for event in episode["events"]:
            event_text = str(event.get("description", "")).lower()
            event_words = set(event_text.split())
            overlap = len(query_words & event_words)
            event_score += overlap / len(query_words) if query_words else 0
        
        # Score based on context
        context_text = str(episode["context"]).lower()
        context_words = set(context_text.split())
        context_overlap = len(query_words & context_words)
        context_score = context_overlap / len(query_words) if query_words else 0
        
        # Combine scores
        total_score = (event_score + context_score) * episode["importance"]
        return total_score


class CognitiveMemory:
    """
    Unified cognitive memory system integrating working, long-term, and episodic memory.
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        working_memory_capacity: int = 7,
        embedding_dim: int = 512,
    ):
        self.capacity = capacity
        self.embedding_dim = embedding_dim
        
        # Initialize memory subsystems
        self.working_memory = WorkingMemory(
            capacity=working_memory_capacity,
            embedding_dim=embedding_dim,
        )
        self.long_term_memory = LongTermMemory(
            embedding_dim=embedding_dim,
        )
        self.episodic_memory = EpisodicMemory(
            max_episodes=capacity // 10,
        )
        
        # Memory management
        self._consolidation_timer = 0
        self._consolidation_interval = 100  # Consolidate every 100 operations
        
        # Statistics
        self.stats = {
            "total_stores": 0,
            "total_retrievals": 0,
            "consolidations": 0,
            "working_memory_evictions": 0,
        }
    
    async def store(
        self,
        content: Dict[str, Any],
        memory_type: str = "working",
        importance: float = 0.5,
        category: Optional[str] = None,
        tags: Optional[Set[str]] = None,
    ) -> UUID:
        """Store content in appropriate memory system."""
        self.stats["total_stores"] += 1
        
        if memory_type == "working":
            item_id = self.working_memory.store(content, importance)
        elif memory_type == "long_term":
            item_id = self.long_term_memory.store(
                content, category or "general", importance, tags
            )
        else:
            # Default to working memory
            item_id = self.working_memory.store(content, importance)
        
        # Check if consolidation is needed
        self._consolidation_timer += 1
        if self._consolidation_timer >= self._consolidation_interval:
            await self._periodic_consolidation()
            self._consolidation_timer = 0
        
        return item_id
    
    async def retrieve(
        self,
        query: Union[str, Dict[str, Any]],
        memory_types: List[str] = None,
        top_k: int = 5,
        **kwargs,
    ) -> List[MemoryItem]:
        """Retrieve items from specified memory systems."""
        self.stats["total_retrievals"] += 1
        
        if memory_types is None:
            memory_types = ["working", "long_term"]
        
        all_results = []
        
        # Retrieve from working memory
        if "working" in memory_types:
            working_results = self.working_memory.retrieve(query, top_k)
            all_results.extend(working_results)
        
        # Retrieve from long-term memory
        if "long_term" in memory_types:
            lt_results = self.long_term_memory.retrieve(
                query,
                category=kwargs.get("category"),
                tags=kwargs.get("tags"),
                top_k=top_k,
            )
            all_results.extend(lt_results)
        
        # Remove duplicates and sort by activation
        unique_results = {item.id: item for item in all_results}
        sorted_results = sorted(
            unique_results.values(),
            key=lambda x: x.get_activation(),
            reverse=True,
        )
        
        return sorted_results[:top_k]
    
    async def store_episode(
        self,
        events: List[Dict[str, Any]],
        context: Dict[str, Any],
        importance: float = 0.5,
    ) -> UUID:
        """Store an episode in episodic memory."""
        return self.episodic_memory.store_episode(events, context, importance)
    
    async def retrieve_episodes(
        self,
        query: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
        context_filter: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve episodes from episodic memory."""
        return self.episodic_memory.retrieve_episodes(
            query, date_range, context_filter, top_k
        )
    
    def get_memory_state(self) -> MemoryState:
        """Get current state of all memory systems."""
        working_state = self.working_memory.get_state()
        
        return MemoryState(
            working_memory=working_state["items"],
            long_term_memory={"total_items": self.long_term_memory.total_items},
            episodic_memory=[],  # Could include recent episodes
            working_memory_capacity=self.working_memory.capacity,
            memory_utilization=working_state["utilization"],
            current_focus=list(working_state["items"].keys())[0] if working_state["items"] else None,
        )
    
    async def _periodic_consolidation(self):
        """Perform periodic memory consolidation."""
        logger.info("Performing memory consolidation")
        
        # Consolidate working memory to long-term memory
        consolidated = self.long_term_memory.consolidate_from_working_memory(
            self.working_memory
        )
        
        # Apply decay to working memory
        self.working_memory.decay_all()
        
        self.stats["consolidations"] += 1
        logger.info(f"Consolidated {consolidated} items")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            **self.stats,
            "working_memory_size": len(self.working_memory.items),
            "working_memory_utilization": len(self.working_memory.items) / self.working_memory.capacity,
            "long_term_memory_size": self.long_term_memory.total_items,
            "episodic_memory_size": len(self.episodic_memory.episodes),
        }
    
    async def shutdown(self):
        """Shutdown memory system and perform final consolidation."""
        logger.info("Shutting down cognitive memory system")
        
        # Final consolidation
        await self._periodic_consolidation()
        
        logger.info("Memory system shutdown complete")
