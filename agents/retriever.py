"""
Retrieval Agent - performs intelligent document chunk retrieval based on analysis.
"""

from typing import Dict, List, Any, Tuple
import numpy as np

from config import Config
from core.vector_store import VectorStore
from agents.query_analyzer import QueryAnalyzer
from utils.logger import get_logger

logger = get_logger(__name__)

class Retriever:
    """Agent responsible for intelligent document retrieval."""
    
    def __init__(self, config: Config):
        self.config = config
        self.vector_store = VectorStore(config)
    
    def retrieve_documents(self, query: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks based on query analysis.
        
        Args:
            query: Original user query
            analysis: Query analysis from QueryAnalyzer
            
        Returns:
            List of relevant document chunks with metadata
        """
        try:
            logger.info(f"Starting document retrieval for query: {query}")
            
            # Generate search queries from analysis
            analyzer = QueryAnalyzer(self.config)
            search_queries = analyzer.get_search_queries(analysis)
            
            all_results = []
            seen_chunks = set()  # Track unique chunks to avoid duplicates
            
            # Perform retrieval for each search query
            for search_query in search_queries:
                logger.info(f"Searching with query: {search_query}")
                
                results = self.vector_store.similarity_search(
                    search_query,
                    k=self.config.retrieval.top_k
                )
                
                # Add results, avoiding duplicates
                for result in results:
                    chunk_id = result.get('chunk_id', result.get('text', '')[:50])
                    if chunk_id not in seen_chunks:
                        seen_chunks.add(chunk_id)
                        all_results.append(result)
            
            # Rank and filter results
            filtered_results = self._rank_and_filter_results(
                all_results, 
                query, 
                analysis
            )
            
            logger.info(f"Retrieved {len(filtered_results)} relevant document chunks")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    def _rank_and_filter_results(
        self, 
        results: List[Dict[str, Any]], 
        query: str, 
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Rank and filter retrieval results based on relevance.
        
        Args:
            results: Raw retrieval results
            query: Original query
            analysis: Query analysis
            
        Returns:
            Ranked and filtered results
        """
        if not results:
            return []
        
        # Filter by similarity threshold
        filtered_results = [
            result for result in results 
            if result.get('similarity_score', 0) >= self.config.retrieval.similarity_threshold
        ]
        
        # If no results meet threshold, take top results anyway
        if not filtered_results and results:
            filtered_results = results[:self.config.retrieval.top_k]
        
        # Sort by similarity score (descending)
        filtered_results.sort(
            key=lambda x: x.get('similarity_score', 0), 
            reverse=True
        )
        
        # Apply additional ranking based on query analysis
        ranked_results = self._apply_intelligent_ranking(
            filtered_results, 
            analysis
        )
        
        # Limit final results
        max_results = min(self.config.retrieval.top_k * 2, 10)  # Allow up to 10 chunks
        return ranked_results[:max_results]
    
    def _apply_intelligent_ranking(
        self, 
        results: List[Dict[str, Any]], 
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply intelligent ranking based on query analysis.
        
        Args:
            results: Filtered results
            analysis: Query analysis
            
        Returns:
            Re-ranked results
        """
        key_entities = analysis.get('key_entities', [])
        main_topic = analysis.get('main_topic', '').lower()
        
        for result in results:
            text = result.get('text', '').lower()
            
            # Boost score based on entity presence
            entity_boost = 0
            for entity in key_entities:
                if entity.lower() in text:
                    entity_boost += 0.1
            
            # Boost score based on main topic presence
            topic_boost = 0.1 if main_topic in text else 0
            
            # Apply boosts
            original_score = result.get('similarity_score', 0)
            boosted_score = min(1.0, original_score + entity_boost + topic_boost)
            result['boosted_score'] = boosted_score
        
        # Sort by boosted score
        results.sort(key=lambda x: x.get('boosted_score', 0), reverse=True)
        
        return results
    
    def get_retrieval_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context for the synthesis agent.
        
        Args:
            retrieved_docs: List of retrieved document chunks
            
        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source_info = f"Source {i}"
            if doc.get('source'):
                source_info += f" ({doc['source']})"
            if doc.get('page'):
                source_info += f", Page {doc['page']}"
            
            context_part = f"[{source_info}]\n{doc.get('text', '')}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def get_source_citations(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract source citations from retrieved documents.
        
        Args:
            retrieved_docs: List of retrieved document chunks
            
        Returns:
            List of source citations
        """
        citations = []
        for doc in retrieved_docs:
            citation = {
                'file': doc.get('source', 'Unknown'),
                'page': doc.get('page'),
                'chunk_text': doc.get('text', '')[:200] + '...' if len(doc.get('text', '')) > 200 else doc.get('text', ''),
                'similarity_score': doc.get('similarity_score', 0)
            }
            citations.append(citation)
        
        return citations
