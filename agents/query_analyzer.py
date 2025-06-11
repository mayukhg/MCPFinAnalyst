"""
Query Analysis Agent - breaks down complex queries and identifies key entities.
"""

import json
from typing import Dict, List, Any
from openai import OpenAI

from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

class QueryAnalyzer:
    """Agent responsible for analyzing and breaking down user queries."""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.get_api_key("openai"))
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the user query to extract key information for retrieval.
        
        Args:
            query: The user's question or query
            
        Returns:
            Dict containing analysis results with query breakdown, entities, etc.
        """
        try:
            logger.info(f"Analyzing query: {query}")
            
            system_prompt = self._get_analysis_system_prompt()
            user_prompt = self._get_analysis_user_prompt(query)
            
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model=self.config.llm.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            analysis_result = json.loads(response.choices[0].message.content)
            
            logger.info(f"Query analysis completed: {analysis_result}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            # Return fallback analysis
            return self._get_fallback_analysis(query)
    
    def _get_analysis_system_prompt(self) -> str:
        """Get the system prompt for query analysis."""
        return """You are a query analysis expert. Your job is to analyze user queries and break them down for optimal document retrieval in a RAG system.

Analyze the given query and provide a JSON response with the following structure:
{
    "query_type": "simple|complex|multi_part",
    "main_topic": "primary subject of the query",
    "key_entities": ["list", "of", "important", "entities"],
    "sub_questions": ["list", "of", "sub-questions", "if", "complex"],
    "search_terms": ["optimized", "search", "terms", "for", "retrieval"],
    "intent": "what the user is trying to accomplish",
    "complexity_score": 1-5,
    "retrieval_strategy": "description of recommended retrieval approach"
}

Guidelines:
- Extract key entities like names, places, concepts, technical terms
- Break complex queries into simpler sub-questions
- Generate optimized search terms for vector similarity search
- Assess query complexity (1=simple factual, 5=complex multi-step reasoning)
- Suggest retrieval strategy based on query characteristics"""

    def _get_analysis_user_prompt(self, query: str) -> str:
        """Get the user prompt for query analysis."""
        return f"""Please analyze the following query:

Query: "{query}"

Provide a comprehensive analysis following the JSON structure specified in the system prompt."""

    def _get_fallback_analysis(self, query: str) -> Dict[str, Any]:
        """Provide fallback analysis if OpenAI call fails."""
        # Simple keyword extraction as fallback
        words = query.lower().split()
        key_terms = [word for word in words if len(word) > 3]
        
        return {
            "query_type": "simple",
            "main_topic": query[:50] + "..." if len(query) > 50 else query,
            "key_entities": key_terms[:5],
            "sub_questions": [query],
            "search_terms": key_terms,
            "intent": "information_seeking",
            "complexity_score": 2,
            "retrieval_strategy": "basic_similarity_search"
        }
    
    def get_search_queries(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate optimized search queries based on analysis.
        
        Args:
            analysis: The query analysis result
            
        Returns:
            List of search queries optimized for vector retrieval
        """
        search_queries = []
        
        # Primary search using original query
        search_queries.append(" ".join(analysis.get("search_terms", [])))
        
        # Additional searches for complex queries
        if analysis.get("query_type") == "complex" and analysis.get("sub_questions"):
            for sub_q in analysis["sub_questions"][:3]:  # Limit to 3 sub-questions
                search_queries.append(sub_q)
        
        # Entity-based search
        if analysis.get("key_entities"):
            entity_search = " ".join(analysis["key_entities"][:3])
            if entity_search not in search_queries:
                search_queries.append(entity_search)
        
        return search_queries[:5]  # Limit total searches
