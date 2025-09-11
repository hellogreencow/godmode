"""Memory architecture for GODMODE."""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from ..models.core import Question, Thread, Entity, Relation
from ..models.responses import GodmodeResponse


class MemoryManager:
    """
    Memory architecture for GODMODE with short-term, long-term, and recall capabilities.
    
    - Short-term: Recent questions, active threads, current session
    - Long-term: Resolved paths, stable preferences, calibrated weights
    - Recall agent: Selects relevant memory shards for context
    """
    
    def __init__(self, max_short_term_items: int = 100):
        # Short-term memory (current session)
        self.short_term_questions: List[Question] = []
        self.short_term_threads: List[Thread] = []
        self.current_session_responses: List[GodmodeResponse] = []
        self.max_short_term_items = max_short_term_items
        
        # Long-term memory (persistent across sessions)
        self.resolved_paths: Dict[str, Dict[str, Any]] = {}  # path_hash -> path_data
        self.user_preferences: Dict[str, float] = {}  # preference -> weight
        self.calibrated_weights: Dict[str, float] = {}  # model_component -> weight
        self.successful_patterns: List[Dict[str, Any]] = []
        
        # Recall indexes
        self.question_embeddings: Dict[str, List[float]] = {}  # question_id -> embedding
        self.topic_clusters: Dict[str, List[str]] = {}  # topic -> [question_ids]
        
        # Statistics for calibration
        self.user_click_patterns: Dict[str, int] = {}  # question_type -> click_count
        self.successful_outcomes: List[Dict[str, Any]] = []
        
        # Session metadata
        self.session_start = datetime.now()
        self.last_activity = datetime.now()
    
    def add_short_term_question(self, question: Question) -> None:
        """Add a question to short-term memory."""
        self.short_term_questions.append(question)
        self.last_activity = datetime.now()
        
        # Trim if exceeding limit
        if len(self.short_term_questions) > self.max_short_term_items:
            # Move oldest to long-term if significant
            old_question = self.short_term_questions.pop(0)
            self._archive_to_long_term(old_question)
    
    def add_short_term_thread(self, thread: Thread) -> None:
        """Add a thread to short-term memory."""
        self.short_term_threads.append(thread)
        self.last_activity = datetime.now()
    
    def add_response(self, response: GodmodeResponse) -> None:
        """Add a response to current session memory."""
        self.current_session_responses.append(response)
        self.last_activity = datetime.now()
        
        # Extract patterns for learning
        self._extract_response_patterns(response)
    
    def record_user_click(self, question_id: str, question_type: str) -> None:
        """Record user click for preference learning."""
        if question_type not in self.user_click_patterns:
            self.user_click_patterns[question_type] = 0
        self.user_click_patterns[question_type] += 1
        
        # Update preferences
        self._update_preferences_from_click(question_type)
    
    def record_successful_outcome(self, thread_id: str, outcome_data: Dict[str, Any]) -> None:
        """Record a successful outcome for pattern learning."""
        outcome = {
            "thread_id": thread_id,
            "timestamp": datetime.now().isoformat(),
            "outcome_data": outcome_data,
            "session_context": self._get_session_context()
        }
        self.successful_outcomes.append(outcome)
        
        # Extract successful patterns
        self._extract_successful_patterns(outcome)
    
    def get_relevant_context(self, current_question: str, max_items: int = 10) -> Dict[str, List[Any]]:
        """
        Recall agent: Get relevant context for current question.
        
        Returns relevant questions, threads, and patterns from memory.
        """
        context = {
            "relevant_questions": [],
            "relevant_threads": [],
            "relevant_patterns": [],
            "user_preferences": dict(self.user_preferences)
        }
        
        # Find semantically similar questions
        similar_questions = self._find_similar_questions(current_question, max_items // 2)
        context["relevant_questions"] = similar_questions
        
        # Find relevant threads
        relevant_threads = self._find_relevant_threads(current_question, max_items // 3)
        context["relevant_threads"] = relevant_threads
        
        # Find applicable patterns
        applicable_patterns = self._find_applicable_patterns(current_question, max_items // 3)
        context["relevant_patterns"] = applicable_patterns
        
        return context
    
    def get_calibrated_weights(self) -> Dict[str, float]:
        """Get calibrated weights based on user behavior."""
        # Start with default weights
        weights = {
            "info_gain_weight": 0.4,
            "coherence_weight": 0.3,
            "user_preference_weight": 0.2,
            "novelty_weight": 0.1
        }
        
        # Adjust based on user click patterns
        total_clicks = sum(self.user_click_patterns.values())
        if total_clicks > 10:  # Enough data for calibration
            # Boost weights for question types user clicks more
            for question_type, click_count in self.user_click_patterns.items():
                click_ratio = click_count / total_clicks
                
                if "define" in question_type.lower():
                    weights["info_gain_weight"] += click_ratio * 0.2
                elif "compare" in question_type.lower():
                    weights["coherence_weight"] += click_ratio * 0.2
                elif "simulate" in question_type.lower():
                    weights["novelty_weight"] += click_ratio * 0.2
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        return {
            "session_duration": (datetime.now() - self.session_start).total_seconds(),
            "questions_processed": len(self.short_term_questions),
            "threads_active": len([t for t in self.short_term_threads if t.status == "active"]),
            "responses_generated": len(self.current_session_responses),
            "user_interactions": sum(self.user_click_patterns.values()),
            "last_activity": self.last_activity.isoformat()
        }
    
    def consolidate_session(self) -> None:
        """Consolidate current session into long-term memory."""
        # Move significant questions to long-term
        for question in self.short_term_questions:
            if self._is_significant_question(question):
                self._archive_to_long_term(question)
        
        # Archive completed threads
        completed_threads = [t for t in self.short_term_threads if t.status == "ended"]
        for thread in completed_threads:
            self._archive_thread_to_long_term(thread)
        
        # Update calibrated weights
        self.calibrated_weights = self.get_calibrated_weights()
        
        # Clear short-term memory
        self.short_term_questions.clear()
        self.short_term_threads = [t for t in self.short_term_threads if t.status != "ended"]
        self.current_session_responses.clear()
    
    def _archive_to_long_term(self, question: Question) -> None:
        """Archive a question to long-term memory."""
        # Create a simplified representation
        archived_question = {
            "id": question.id,
            "text": question.text,
            "cognitive_move": question.cognitive_move.value,
            "level": question.level,
            "expected_info_gain": question.expected_info_gain,
            "confidence": question.confidence,
            "tags": question.tags,
            "archived_at": datetime.now().isoformat()
        }
        
        # Store in topic clusters
        primary_tag = question.tags[0] if question.tags else "general"
        if primary_tag not in self.topic_clusters:
            self.topic_clusters[primary_tag] = []
        self.topic_clusters[primary_tag].append(question.id)
        
        # Store question data (in a real system, this would go to persistent storage)
        # For now, just keep in memory
    
    def _archive_thread_to_long_term(self, thread: Thread) -> None:
        """Archive a thread to long-term memory."""
        # Create path signature
        path_signature = "_".join(thread.path)
        path_hash = str(hash(path_signature))
        
        self.resolved_paths[path_hash] = {
            "thread_id": thread.thread_id,
            "path": thread.path,
            "summary": thread.summary,
            "path_length": len(thread.path),
            "archived_at": datetime.now().isoformat()
        }
    
    def _extract_response_patterns(self, response: GodmodeResponse) -> None:
        """Extract patterns from a response for learning."""
        # Analyze which types of questions were generated
        question_types = []
        for scenario in response.graph_update.scenarios:
            for question in scenario.lane:
                question_types.append(question.cognitive_move.value)
        
        # Store pattern if significant
        if len(question_types) > 2:
            pattern = {
                "question_types": question_types,
                "scenario_count": len(response.graph_update.scenarios),
                "prior_count": len(response.graph_update.priors),
                "created_at": datetime.now().isoformat()
            }
            self.successful_patterns.append(pattern)
    
    def _update_preferences_from_click(self, question_type: str) -> None:
        """Update user preferences based on click."""
        if question_type not in self.user_preferences:
            self.user_preferences[question_type] = 0.5
        
        # Increase preference (with diminishing returns)
        current_pref = self.user_preferences[question_type]
        increment = 0.1 * (1.0 - current_pref)  # Diminishing returns
        self.user_preferences[question_type] = min(1.0, current_pref + increment)
    
    def _find_similar_questions(self, current_question: str, max_items: int) -> List[Dict[str, Any]]:
        """Find questions similar to current question."""
        # Simple keyword-based similarity for now
        current_words = set(current_question.lower().split())
        similar_questions = []
        
        for question in self.short_term_questions:
            question_words = set(question.text.lower().split())
            overlap = len(current_words & question_words)
            
            if overlap >= 2:  # At least 2 words in common
                similarity = overlap / len(current_words | question_words)
                similar_questions.append({
                    "question": question,
                    "similarity": similarity
                })
        
        # Sort by similarity and return top matches
        similar_questions.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_questions[:max_items]
    
    def _find_relevant_threads(self, current_question: str, max_items: int) -> List[Thread]:
        """Find threads relevant to current question."""
        # Simple relevance based on thread summary similarity
        current_words = set(current_question.lower().split())
        relevant_threads = []
        
        for thread in self.short_term_threads:
            summary_words = set(thread.summary.lower().split())
            overlap = len(current_words & summary_words)
            
            if overlap >= 1:
                relevance = overlap / len(current_words | summary_words)
                relevant_threads.append((thread, relevance))
        
        # Sort by relevance
        relevant_threads.sort(key=lambda x: x[1], reverse=True)
        return [thread for thread, _ in relevant_threads[:max_items]]
    
    def _find_applicable_patterns(self, current_question: str, max_items: int) -> List[Dict[str, Any]]:
        """Find patterns applicable to current question."""
        # Return most recent successful patterns
        return self.successful_patterns[-max_items:] if self.successful_patterns else []
    
    def _is_significant_question(self, question: Question) -> bool:
        """Determine if a question is significant enough to archive."""
        return (
            question.expected_info_gain > 0.5 or
            question.confidence > 0.7 or
            question.level >= 3 or
            len(question.builds_on) > 1
        )
    
    def _get_session_context(self) -> Dict[str, Any]:
        """Get current session context."""
        return {
            "session_duration": (datetime.now() - self.session_start).total_seconds(),
            "questions_count": len(self.short_term_questions),
            "active_threads": len([t for t in self.short_term_threads if t.status == "active"])
        }
    
    def _extract_successful_patterns(self, outcome: Dict[str, Any]) -> None:
        """Extract patterns from successful outcomes."""
        # Find the thread that led to this outcome
        thread_id = outcome["thread_id"]
        thread = next((t for t in self.short_term_threads if t.thread_id == thread_id), None)
        
        if thread and len(thread.path) > 1:
            # Extract pattern from successful path
            pattern = {
                "path_length": len(thread.path),
                "path_signature": "_".join(thread.path[:3]),  # First 3 steps
                "outcome_type": outcome["outcome_data"].get("type", "unknown"),
                "success_score": outcome["outcome_data"].get("score", 1.0),
                "extracted_at": datetime.now().isoformat()
            }
            self.successful_patterns.append(pattern)