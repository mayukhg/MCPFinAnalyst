#!/usr/bin/env python3
"""
Agentic RAG Workflow Engine CLI
Main entry point for the command-line interface.
"""

import click
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
import time

from config import Config
from core.document_processor import DocumentProcessor
from core.vector_store import VectorStore
from core.workflow_orchestrator import WorkflowOrchestrator
from utils.logger import setup_logger

console = Console()
logger = setup_logger(__name__)

@click.group()
@click.option('--config', '-c', default='config.yaml', help='Configuration file path')
@click.pass_context
def cli(ctx, config):
    """Agentic RAG Workflow Engine - Intelligent document retrieval and synthesis."""
    ctx.ensure_object(dict)
    try:
        ctx.obj['config'] = Config.load_config(config)
        logger.info(f"Loaded configuration from {config}")
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        sys.exit(1)

@cli.command()
@click.argument('documents_path', type=click.Path(exists=True))
@click.option('--recursive', '-r', is_flag=True, help='Recursively process subdirectories')
@click.pass_context
def ingest(ctx, documents_path, recursive):
    """Ingest documents from a directory into the vector database."""
    config = ctx.obj['config']
    
    console.print(Panel.fit(
        "[bold cyan]Document Ingestion Process[/bold cyan]",
        border_style="cyan"
    ))
    
    try:
        # Initialize components
        doc_processor = DocumentProcessor(config)
        vector_store = VectorStore(config)
        
        documents_path = Path(documents_path)
        
        # Find all supported documents
        supported_extensions = {'.pdf', '.md', '.txt'}
        documents = []
        
        if documents_path.is_file():
            if documents_path.suffix.lower() in supported_extensions:
                documents = [documents_path]
        else:
            pattern = "**/*" if recursive else "*"
            for ext in supported_extensions:
                documents.extend(documents_path.glob(f"{pattern}{ext}"))
        
        if not documents:
            console.print("[yellow]No supported documents found.[/yellow]")
            return
        
        console.print(f"Found {len(documents)} documents to process")
        
        # Process documents with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Processing documents...", total=len(documents))
            processed_count = 0
            
            for doc_path in documents:
                try:
                    progress.update(task, description=f"Processing {doc_path.name}...")
                    
                    # Process document
                    chunks = doc_processor.process_document(doc_path)
                    
                    # Store in vector database
                    vector_store.add_documents(chunks, str(doc_path))
                    
                    processed_count += 1
                    progress.update(task, advance=1)
                    
                except Exception as e:
                    logger.error(f"Error processing {doc_path}: {e}")
                    console.print(f"[red]Error processing {doc_path.name}: {e}[/red]")
        
        console.print(f"[green]Successfully processed {processed_count}/{len(documents)} documents[/green]")
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        console.print(f"[red]Ingestion failed: {e}[/red]")
        sys.exit(1)

@cli.command()
@click.argument('query', type=str)
@click.option('--verbose', '-v', is_flag=True, help='Show detailed workflow steps')
@click.pass_context
def query(ctx, query, verbose):
    """Query the RAG system with an intelligent multi-agent workflow."""
    config = ctx.obj['config']
    
    console.print(Panel.fit(
        f"[bold cyan]Query:[/bold cyan] {query}",
        border_style="cyan"
    ))
    
    try:
        # Initialize orchestrator
        orchestrator = WorkflowOrchestrator(config)
        
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Processing query...", total=None)
            
            # Execute workflow
            result = orchestrator.execute_workflow(query, verbose=verbose)
            
            progress.update(task, description="Query completed!")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Display results
        console.print("\n" + "="*60)
        console.print(Panel.fit(
            "[bold green]Generated Answer[/bold green]",
            border_style="green"
        ))
        
        console.print(result['answer'])
        
        # Display sources
        if result.get('sources'):
            console.print("\n" + Panel.fit(
                "[bold blue]Sources[/bold blue]",
                border_style="blue"
            ))
            
            for i, source in enumerate(result['sources'], 1):
                console.print(f"{i}. [cyan]{source['file']}[/cyan]")
                if source.get('page'):
                    console.print(f"   Page: {source['page']}")
                if source.get('chunk_text'):
                    # Show first 100 chars of the chunk
                    preview = source['chunk_text'][:100] + "..." if len(source['chunk_text']) > 100 else source['chunk_text']
                    console.print(f"   Preview: [dim]{preview}[/dim]")
                console.print()
        
        # Show workflow steps if verbose
        if verbose and result.get('workflow_steps'):
            console.print(Panel.fit(
                "[bold yellow]Workflow Steps[/bold yellow]",
                border_style="yellow"
            ))
            
            for step in result['workflow_steps']:
                console.print(f"[bold]{step['agent']}:[/bold] {step['description']}")
                if step.get('details'):
                    console.print(f"  [dim]{step['details']}[/dim]")
                console.print()
        
        console.print(f"[dim]Processing time: {processing_time:.2f}s[/dim]")
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        console.print(f"[red]Query failed: {e}[/red]")
        sys.exit(1)

@cli.command()
@click.pass_context
def status(ctx):
    """Show system status and configuration."""
    config = ctx.obj['config']
    
    console.print(Panel.fit(
        "[bold cyan]System Status[/bold cyan]",
        border_style="cyan"
    ))
    
    try:
        # Check vector store
        vector_store = VectorStore(config)
        doc_count = vector_store.get_document_count()
        
        console.print(f"Vector Database: [green]Connected[/green]")
        console.print(f"Documents indexed: [cyan]{doc_count}[/cyan]")
        console.print(f"LLM Model: [cyan]{config.llm.model}[/cyan]")
        console.print(f"Embedding Model: [cyan]{config.embeddings.model}[/cyan]")
        console.print(f"Vector Database: [cyan]{config.vector_store.type}[/cyan]")
        
        # Check OpenAI API
        if os.getenv("OPENAI_API_KEY"):
            console.print("OpenAI API Key: [green]Configured[/green]")
        else:
            console.print("OpenAI API Key: [red]Not configured[/red]")
            
    except Exception as e:
        console.print(f"[red]Status check failed: {e}[/red]")

if __name__ == '__main__':
    cli()
