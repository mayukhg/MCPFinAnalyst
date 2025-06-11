"""
Workflow Orchestrator - manages the multi-agent RAG workflow execution.
"""

import time
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agents import QueryAnalyzer, Retriever, Synthesizer
from utils.config_manager import ConfigManager
from utils.logger import get_logger

logger = get_logger(__name__)

class WorkflowOrchestrator:
    """Orchestrates the multi-agent RAG workflow."""
    
    def __init__(self, config):
        self.config = config
        self.query_analyzer = QueryAnalyzer(config)
        self.retriever = Retriever(config)
        self.synthesizer = Synthesizer(config)
        self.workflow_steps = []
    
    def execute_workflow(self, query: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Execute the complete agentic RAG workflow.
        
        Args:
            query: User query
            verbose: Whether to include detailed workflow steps in output
            
        Returns:
            Dict containing the final answer, sources, and workflow metadata
        """
        try:
            logger.info(f"Starting workflow execution for query: {query}")
            start_time = time.time()
            
            # Initialize workflow tracking
            self.workflow_steps = []
            
            # Step 1: Query Analysis
            logger.info("Step 1: Analyzing query")
            analysis_start = time.time()
            
            analysis = self.query_analyzer.analyze_query(query)
            
            analysis_time = time.time() - analysis_start
            self._add_workflow_step(
                agent="Query Analyzer",
                description="Analyzed query structure and extracted key entities",
                details=f"Query type: {analysis.get('query_type', 'unknown')}, "
                       f"Complexity: {analysis.get('complexity_score', 1)}/5, "
                       f"Key entities: {', '.join(analysis.get('key_entities', [])[:3])}",
                duration=analysis_time
            )
            
            # Step 2: Document Retrieval
            logger.info("Step 2: Retrieving relevant documents")
            retrieval_start = time.time()
            
            retrieved_docs = self.retriever.retrieve_documents(query, analysis)
            
            if not retrieved_docs:
                logger.warning("No relevant documents found")
                return self._handle_no_documents_found(query)
            
            retrieval_time = time.time() - retrieval_start
            self._add_workflow_step(
                agent="Retriever",
                description=f"Retrieved {len(retrieved_docs)} relevant document chunks",
                details=f"Sources: {', '.join(set(doc.get('source', 'Unknown') for doc in retrieved_docs[:3]))}",
                duration=retrieval_time
            )
            
            # Step 3: Context Preparation
            context = self.retriever.get_retrieval_context(retrieved_docs)
            citations = self.retriever.get_source_citations(retrieved_docs)
            
            # Step 4: Answer Synthesis
            logger.info("Step 3: Synthesizing answer")
            synthesis_start = time.time()
            
            synthesis_result = self.synthesizer.synthesize_answer(
                query=query,
                context=context,
                analysis=analysis,
                retrieved_docs=retrieved_docs
            )
            
            synthesis_time = time.time() - synthesis_start
            self._add_workflow_step(
                agent="Synthesizer",
                description="Generated comprehensive answer with source citations",
                details=f"Confidence: {synthesis_result.get('confidence', 0):.2f}, "
                       f"Sources cited: {len(citations)}",
                duration=synthesis_time
            )
            
            # Prepare final result
            total_time = time.time() - start_time
            
            result = {
                "answer": synthesis_result["answer"],
                "sources": citations,
                "confidence": synthesis_result.get("confidence", 0.5),
                "query_analysis": analysis,
                "processing_time": total_time,
                "workflow_steps": self.workflow_steps if verbose else None
            }
            
            logger.info(f"Workflow completed successfully in {total_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return self._handle_workflow_error(query, str(e))
    
    def _add_workflow_step(
        self, 
        agent: str, 
        description: str, 
        details: str = None, 
        duration: float = None
    ):
        """Add a step to the workflow tracking."""
        step = {
            "agent": agent,
            "description": description,
            "timestamp": time.time()
        }
        
        if details:
            step["details"] = details
        if duration:
            step["duration"] = round(duration, 3)
        
        self.workflow_steps.append(step)
    
    def _handle_no_documents_found(self, query: str) -> Dict[str, Any]:
        """Handle case when no relevant documents are found."""
        self._add_workflow_step(
            agent="System",
            description="No relevant documents found",
            details="The query did not match any documents in the knowledge base"
        )
        
        return {
            "answer": f"I couldn't find any relevant documents in the knowledge base to answer your query: '{query}'. "
                     f"This could mean the information is not available in the indexed documents, or you might want to "
                     f"try rephrasing your question with different keywords.",
            "sources": [],
            "confidence": 0.1,
            "query_analysis": {"query_type": "no_match"},
            "processing_time": 0,
            "workflow_steps": self.workflow_steps
        }
    
    def _handle_workflow_error(self, query: str, error_message: str) -> Dict[str, Any]:
        """Handle workflow execution errors."""
        self._add_workflow_step(
            agent="System",
            description="Workflow execution failed",
            details=f"Error: {error_message}"
        )
        
        return {
            "answer": f"I encountered an error while processing your query: '{query}'. "
                     f"Error details: {error_message}. Please try again or contact support if the issue persists.",
            "sources": [],
            "confidence": 0.0,
            "query_analysis": {"query_type": "error"},
            "processing_time": 0,
            "workflow_steps": self.workflow_steps
        }
    
    def validate_workflow_config(self) -> Dict[str, Any]:
        """
        Validate that all workflow components are properly configured.
        
        Returns:
            Validation results
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Test OpenAI API key
            api_key = self.config.get_api_key("openai")
            if not api_key or api_key == "default_key":
                validation_results["errors"].append("OpenAI API key not configured")
                validation_results["valid"] = False
            
            # Test vector store connection
            try:
                doc_count = self.retriever.vector_store.get_document_count()
                if doc_count == 0:
                    validation_results["warnings"].append("No documents indexed in vector store")
            except Exception as e:
                validation_results["errors"].append(f"Vector store connection failed: {e}")
                validation_results["valid"] = False
            
            # Validate configuration parameters
            if self.config.retrieval.top_k <= 0:
                validation_results["errors"].append("Invalid retrieval top_k value")
                validation_results["valid"] = False
            
            if self.config.chunking.chunk_size <= 0:
                validation_results["errors"].append("Invalid chunking chunk_size value")
                validation_results["valid"] = False
            
        except Exception as e:
            validation_results["errors"].append(f"Configuration validation failed: {e}")
            validation_results["valid"] = False
        
        return validation_results