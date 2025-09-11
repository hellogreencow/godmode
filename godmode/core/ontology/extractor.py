"""Entity and relation extraction for ontology building."""

import re
import asyncio
from typing import List, Dict, Any, Optional, Set
from ...models.core import Question, EntityType, Evidence


class EntityExtractor:
    """Extracts entities from text using rule-based and pattern matching approaches."""
    
    def __init__(self):
        # Entity type patterns
        self.entity_patterns = {
            EntityType.PERSON: [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # John Smith
                r'\b(CEO|CTO|manager|director|employee|user|customer|client)\b',
                r'\b(I|you|we|they|person|people|team|individual)\b'
            ],
            EntityType.ORG: [
                r'\b[A-Z][a-z]+ (Inc|LLC|Corp|Company|Organization)\b',
                r'\b(company|organization|business|startup|enterprise|firm)\b',
                r'\b(team|department|division|group)\b'
            ],
            EntityType.PRODUCT: [
                r'\b(product|service|software|application|app|tool|platform|system)\b',
                r'\b(website|portal|dashboard|interface)\b',
                r'\b[A-Z][a-z]+ (Pro|Plus|Premium|Enterprise)\b'
            ],
            EntityType.GOAL: [
                r'\b(goal|objective|target|aim|purpose|mission)\b',
                r'\b(success|achievement|outcome|result)\b',
                r'\b(increase|improve|optimize|maximize|minimize)\b'
            ],
            EntityType.METRIC: [
                r'\b(metric|measure|KPI|indicator|benchmark)\b',
                r'\b(revenue|profit|cost|price|budget|ROI)\b',
                r'\b(performance|efficiency|productivity|quality)\b',
                r'\b\d+%\b',  # Percentages
                r'\$\d+',    # Dollar amounts
            ],
            EntityType.PLACE: [
                r'\b[A-Z][a-z]+ (City|State|Country|Street|Avenue)\b',
                r'\b(office|location|site|facility|headquarters)\b',
                r'\b(online|remote|virtual|cloud)\b'
            ],
            EntityType.TIME: [
                r'\b(today|tomorrow|yesterday|week|month|year|quarter)\b',
                r'\b(deadline|timeline|schedule|duration)\b',
                r'\b\d{4}\b',  # Years
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b'
            ],
            EntityType.CONCEPT: [
                r'\b(strategy|approach|method|process|framework|methodology)\b',
                r'\b(technology|innovation|solution|opportunity|challenge)\b',
                r'\b(quality|value|benefit|advantage|risk|issue)\b'
            ]
        }
        
        # Common aliases and variations
        self.alias_patterns = {
            "user": ["customer", "client", "end-user"],
            "company": ["organization", "business", "firm"],
            "product": ["service", "solution", "offering"],
            "goal": ["objective", "target", "aim"],
            "metric": ["KPI", "measure", "indicator"],
        }
    
    async def extract_entities(
        self,
        current_question: str,
        context: Optional[str],
        questions: List[Question]
    ) -> List[Dict[str, Any]]:
        """Extract entities from question and context."""
        # Combine all text
        all_text = current_question
        if context:
            all_text += " " + context
        for question in questions:
            all_text += " " + question.text
        
        return await self.extract_entities_from_text(all_text)
    
    async def extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from raw text."""
        entities = []
        seen_entities = set()
        
        text_lower = text.lower()
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                
                for match in matches:
                    entity_name = match.group().strip()
                    
                    # Skip very short or common words
                    if len(entity_name) < 2 or entity_name in ['i', 'a', 'the', 'and', 'or']:
                        continue
                    
                    # Normalize entity name
                    entity_name = self._normalize_entity_name(entity_name)
                    
                    # Skip duplicates
                    entity_key = (entity_name, entity_type)
                    if entity_key in seen_entities:
                        continue
                    seen_entities.add(entity_key)
                    
                    # Get aliases
                    aliases = self._get_entity_aliases(entity_name)
                    
                    # Calculate confidence based on pattern strength and frequency
                    confidence = self._calculate_entity_confidence(entity_name, text, pattern)
                    
                    entity_data = {
                        "name": entity_name,
                        "type": entity_type,
                        "aliases": aliases,
                        "confidence": confidence
                    }
                    entities.append(entity_data)
        
        return entities
    
    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for consistency."""
        # Title case for proper nouns
        if re.match(r'^[A-Z]', name):
            return name.title()
        return name.lower()
    
    def _get_entity_aliases(self, entity_name: str) -> List[str]:
        """Get aliases for an entity."""
        name_lower = entity_name.lower()
        aliases = []
        
        for base_name, alias_list in self.alias_patterns.items():
            if base_name in name_lower or name_lower in base_name:
                aliases.extend(alias_list)
        
        # Remove the entity name itself from aliases
        aliases = [alias for alias in aliases if alias.lower() != name_lower]
        
        return list(set(aliases))  # Remove duplicates
    
    def _calculate_entity_confidence(self, entity_name: str, text: str, pattern: str) -> float:
        """Calculate confidence score for an entity."""
        base_confidence = 0.5
        
        # Boost confidence for proper nouns
        if re.match(r'^[A-Z]', entity_name):
            base_confidence += 0.2
        
        # Boost confidence for specific patterns
        if r'\b[A-Z]' in pattern:  # Capitalized patterns
            base_confidence += 0.1
        
        # Boost confidence for frequency
        frequency = text.lower().count(entity_name.lower())
        frequency_bonus = min(0.2, frequency * 0.05)
        base_confidence += frequency_bonus
        
        # Penalize very common words
        common_words = {'thing', 'stuff', 'something', 'anything', 'everything'}
        if entity_name.lower() in common_words:
            base_confidence -= 0.3
        
        return min(1.0, max(0.1, base_confidence))


class RelationExtractor:
    """Extracts relations between entities from text."""
    
    def __init__(self):
        # Relation patterns (subject, predicate, object)
        self.relation_patterns = [
            # Ownership/possession
            (r'(\w+)\s+(owns|has|possesses|contains)\s+(\w+)', 'owns'),
            (r'(\w+)\s+is\s+owned\s+by\s+(\w+)', 'owned_by'),
            
            # Hierarchical relationships
            (r'(\w+)\s+(manages|leads|supervises)\s+(\w+)', 'manages'),
            (r'(\w+)\s+(works\s+for|reports\s+to)\s+(\w+)', 'works_for'),
            (r'(\w+)\s+is\s+part\s+of\s+(\w+)', 'part_of'),
            
            # Causal relationships
            (r'(\w+)\s+(causes|leads\s+to|results\s+in)\s+(\w+)', 'causes'),
            (r'(\w+)\s+(affects|impacts|influences)\s+(\w+)', 'affects'),
            
            # Temporal relationships
            (r'(\w+)\s+(happens\s+before|precedes)\s+(\w+)', 'precedes'),
            (r'(\w+)\s+(happens\s+after|follows)\s+(\w+)', 'follows'),
            
            # Comparison relationships
            (r'(\w+)\s+is\s+(better|worse|faster|slower)\s+than\s+(\w+)', 'compared_to'),
            (r'(\w+)\s+(exceeds|surpasses|outperforms)\s+(\w+)', 'exceeds'),
            
            # Usage relationships
            (r'(\w+)\s+(uses|utilizes|employs)\s+(\w+)', 'uses'),
            (r'(\w+)\s+is\s+used\s+by\s+(\w+)', 'used_by'),
            
            # Dependency relationships
            (r'(\w+)\s+(depends\s+on|requires|needs)\s+(\w+)', 'depends_on'),
            (r'(\w+)\s+(provides|supplies|delivers)\s+(\w+)', 'provides'),
        ]
        
        # Predicate mappings for normalization
        self.predicate_mappings = {
            'owns': 'owns',
            'owned_by': 'owned_by',
            'manages': 'manages',
            'works_for': 'employed_by',
            'part_of': 'part_of',
            'causes': 'causes',
            'affects': 'affects',
            'precedes': 'precedes',
            'follows': 'follows',
            'compared_to': 'compared_to',
            'exceeds': 'exceeds',
            'uses': 'uses',
            'used_by': 'used_by',
            'depends_on': 'depends_on',
            'provides': 'provides',
        }
    
    async def extract_relations(
        self,
        current_question: str,
        context: Optional[str],
        questions: List[Question]
    ) -> List[Dict[str, Any]]:
        """Extract relations from question and context."""
        # Combine all text
        all_text = current_question
        if context:
            all_text += " " + context
        for question in questions:
            all_text += " " + question.text
        
        return await self.extract_relations_from_text(all_text)
    
    async def extract_relations_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract relations from raw text."""
        relations = []
        seen_relations = set()
        
        for pattern, predicate in self.relation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    subj = groups[0].strip()
                    obj = groups[-1].strip()  # Last group is object
                    
                    # Skip very short subjects/objects
                    if len(subj) < 2 or len(obj) < 2:
                        continue
                    
                    # Normalize predicate
                    normalized_pred = self.predicate_mappings.get(predicate, predicate)
                    
                    # Create relation key for deduplication
                    relation_key = (subj.lower(), normalized_pred, obj.lower())
                    if relation_key in seen_relations:
                        continue
                    seen_relations.add(relation_key)
                    
                    # Calculate confidence
                    confidence = self._calculate_relation_confidence(match.group(), text, pattern)
                    
                    # Create evidence
                    evidence = [Evidence(
                        source_type="user",
                        snippet=match.group(),
                        note=f"Extracted from pattern: {pattern}"
                    )]
                    
                    relation_data = {
                        "subj": subj,  # Will need to map to entity IDs later
                        "pred": normalized_pred,
                        "obj": obj,    # Will need to map to entity ID or keep as literal
                        "confidence": confidence,
                        "hypothesis": True,  # Mark as hypothesis since extracted
                        "evidence": evidence
                    }
                    relations.append(relation_data)
        
        return relations
    
    def _calculate_relation_confidence(self, match_text: str, full_text: str, pattern: str) -> float:
        """Calculate confidence score for a relation."""
        base_confidence = 0.6
        
        # Boost confidence for explicit relation words
        explicit_words = ['is', 'has', 'owns', 'manages', 'causes', 'affects']
        if any(word in match_text.lower() for word in explicit_words):
            base_confidence += 0.1
        
        # Boost confidence for longer matches (more context)
        if len(match_text.split()) > 3:
            base_confidence += 0.1
        
        # Boost confidence for frequency of similar patterns
        similar_matches = len(re.findall(pattern, full_text, re.IGNORECASE))
        if similar_matches > 1:
            base_confidence += min(0.2, similar_matches * 0.05)
        
        return min(1.0, max(0.3, base_confidence))