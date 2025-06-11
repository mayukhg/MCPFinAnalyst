"""
Synthesis Agent - generates coherent, source-cited answers from retrieved context.
"""

import json
from typing import Dict, List, Any
from openai import OpenAI

from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

class Synthesizer:
    """Agent responsible for synthesizing answers from retrieved context."""
    
    def __init__(self, config: Config):
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
    
    def validate_answer_quality(self, answer: str, query: str, context: str) -> Dict[str, Any]:
        """
        Validate the quality of the generated answer.
        
        Args:
            answer: Generated answer
            query: Original query
            context: Source context
            
        Returns:
            Quality assessment dict
        """
        quality_metrics = {
            "has_citations": "[Source" in answer,
            "appropriate_length": 20 <= len(answer.split()) <= 500,
            "addresses_query": len(set(query.lower().split()) & set(answer.lower().split())) > 0,
            "uses_context": any(chunk in answer.lower() for chunk in context.lower().split()[:10]),
        }
        
        quality_score = sum(quality_metrics.values()) / len(quality_metrics)
        
        return {
            "quality_score": quality_score,
            "metrics": quality_metrics,
            "recommendations": self._get_quality_recommendations(quality_metrics)
        }
    
    def _get_quality_recommendations(self, metrics: Dict[str, bool]) -> List[str]:
        """Get recommendations for improving answer quality."""
        recommendations = []
        
        if not metrics["has_citations"]:
            recommendations.append("Answer should include source citations")
        
        if not metrics["appropriate_length"]:
            recommendations.append("Answer length should be between 20-500 words")
        
        if not metrics["addresses_query"]:
            recommendations.append("Answer should more directly address the query")
        
        if not metrics["uses_context"]:
            recommendations.append("Answer should make better use of provided context")
        
        return recommendations
