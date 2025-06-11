"""
Defines the QueryAnalyzer, Retriever, and Synthesizer agents for the RAG workflow.
"""

import json
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
import numpy as np

from components.vector_stores import VectorStoreManager
from utils.logger import get_logger

logger = get_logger(__name__)

class QueryAnalyzer:
    """Agent responsible for analyzing and breaking down user queries."""
    
    def __init__(self, config):
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


class Retriever:
    """Agent responsible for intelligent document retrieval."""
    
    def __init__(self, config):
        self.config = config
        self.vector_store = VectorStoreManager(config)
    
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


class Synthesizer:
    """Agent responsible for synthesizing answers from retrieved context."""
    
    def __init__(self, config):
        self.config = config
        self.client = OpenAI(api_key=config.get_api_key("openai"))
    
    def synthesize_answer(
        self, 
        query: str, 
        context: str, 
        analysis: Dict[str, Any],
        retrieved_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive answer based on retrieved context.
        
        Args:
            query: Original user query
            context: Retrieved document context
            analysis: Query analysis from QueryAnalyzer
            retrieved_docs: List of retrieved documents for citation
            
        Returns:
            Dict containing synthesized answer and metadata
        """
        try:
            logger.info(f"Synthesizing answer for query: {query}")
            
            system_prompt = self._get_synthesis_system_prompt()
            user_prompt = self._get_synthesis_user_prompt(query, context, analysis)
            
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model=self.config.llm.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens
            )
            
            answer = response.choices[0].message.content
            
            # Extract source information for citations
            sources = self._extract_sources_from_context(retrieved_docs)
            
            result = {
                "answer": answer,
                "sources": sources,
                "confidence": self._estimate_confidence(answer, context),
                "query_complexity": analysis.get("complexity_score", 1)
            }
            
            logger.info("Answer synthesis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            return self._get_fallback_answer(query)
    
    def _get_synthesis_system_prompt(self) -> str:
        """Get the system prompt for answer synthesis."""
        return """You are an expert research assistant specializing in synthesizing comprehensive answers from multiple sources. Your role is to:

1. ANALYZE the provided context documents carefully
2. SYNTHESIZE a clear, accurate answer that directly addresses the user's question
3. CITE sources appropriately using the format [Source X] where X is the source number
4. ENSURE all claims in your answer are supported by the provided context
5. ACKNOWLEDGE limitations when the context doesn't fully address the question

Guidelines for your response:
- Begin with a direct answer to the question
- Use evidence from the provided sources to support your points
- Include specific details and examples from the sources when relevant
- Cite sources throughout your answer using [Source X] format
- If information is incomplete or unclear, explicitly state this
- Maintain objectivity and avoid speculation beyond what sources support
- Structure your answer logically with clear reasoning
- End with a summary if the question was complex

CRITICAL: Only use information that is explicitly present in the provided context. Do not add external knowledge or make unsupported claims."""

    def _get_synthesis_user_prompt(
        self, 
        query: str, 
        context: str, 
        analysis: Dict[str, Any]
    ) -> str:
        """Get the user prompt for answer synthesis."""
        complexity_note = ""
        if analysis.get("complexity_score", 1) > 3:
            complexity_note = "\nNote: This is a complex query that may require multi-step reasoning. Please structure your answer to address each aspect systematically."
        
        return f"""Please provide a comprehensive answer to the following question based on the provided context documents.

QUESTION: {query}

CONTEXT DOCUMENTS:
{context}

Query Analysis Summary:
- Main Topic: {analysis.get('main_topic', 'Not specified')}
- Key Entities: {', '.join(analysis.get('key_entities', []))}
- Complexity: {analysis.get('complexity_score', 1)}/5{complexity_note}

Please synthesize a clear, well-cited answer that directly addresses the question using only the information provided in the context documents."""

    def _extract_sources_from_context(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information for citations."""
        sources = []
        for i, doc in enumerate(retrieved_docs, 1):
            source = {
                'id': i,
                'file': doc.get('source', 'Unknown'),
                'page': doc.get('page'),
                'chunk_text': doc.get('text', ''),
                'similarity_score': doc.get('similarity_score', 0)
            }
            sources.append(source)
        
        return sources
    
    def _estimate_confidence(self, answer: str, context: str) -> float:
        """
        Estimate confidence in the generated answer.
        
        Args:
            answer: Generated answer
            context: Source context
            
        Returns:
            Confidence score between 0 and 1
        """
        # Simple heuristic-based confidence estimation
        confidence = 0.5  # Base confidence
        
        # Boost confidence if answer contains citations
        citation_count = answer.count('[Source')
        if citation_count > 0:
            confidence += min(0.3, citation_count * 0.1)
        
        # Boost confidence based on answer length (within reason)
        answer_length = len(answer.split())
        if 50 <= answer_length <= 300:
            confidence += 0.1
        
        # Reduce confidence if answer is very short or very long
        if answer_length < 20:
            confidence -= 0.2
        elif answer_length > 500:
            confidence -= 0.1
        
        # Boost confidence if context is substantial
        context_length = len(context.split())
        if context_length > 200:
            confidence += 0.1
        
        return max(0.1, min(1.0, confidence))
    
    def _get_fallback_answer(self, query: str) -> Dict[str, Any]:
        """Provide fallback answer if synthesis fails."""
        return {
            "answer": f"I apologize, but I encountered an error while processing your query: '{query}'. Please try again or rephrase your question.",
            "sources": [],
            "confidence": 0.1,
            "query_complexity": 1
        }