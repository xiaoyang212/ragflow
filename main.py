#!/usr/bin/env python
#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
Dual Index Main Script for Concept Graph and Summary Tree Retrieval
This script provides indexing and retrieval functionality for both GraphRAG and RAPTOR.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from common import settings
from common.constants import LLMType
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.document_service import DocumentService
from api.db.services.llm_service import LLMBundle
from api.db.services.user_service import TenantService
from graphrag.general.index import run_graphrag_for_kb
from rag.svr.task_executor import run_raptor_for_kb
from graphrag.search import KGSearch
from rag.nlp import search


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: dict):
    """Validate configuration parameters."""
    required_fields = ['knowledgebase', 'llm']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required configuration field: {field}")
    
    kb_config = config['knowledgebase']
    if 'tenant_id' not in kb_config or 'kb_id' not in kb_config:
        raise ValueError("knowledgebase configuration must include tenant_id and kb_id")
    
    if kb_config['tenant_id'] == 'your_tenant_id' or kb_config['kb_id'] == 'your_kb_id':
        raise ValueError("Please update tenant_id and kb_id in the configuration file")


async def run_graphrag_indexing(config: dict, kb: dict, tenant: dict, chat_mdl, embed_mdl):
    """Run GraphRAG indexing to generate concept graph."""
    logger.info("Starting GraphRAG indexing...")
    
    graphrag_config = config.get('graphrag', {})
    if not graphrag_config.get('enabled', True):
        logger.info("GraphRAG is disabled in configuration")
        return
    
    indexing_config = config.get('indexing', {})
    doc_ids = indexing_config.get('doc_ids', [])
    
    # Prepare parser config for GraphRAG
    kb_parser_config = {
        "graphrag": {
            "method": graphrag_config.get('method', 'light'),
            "entity_types": graphrag_config.get('entity_types', []),
        }
    }
    
    # Create a row dict simulating task structure
    row = {
        "id": "dual_index_graphrag_task",
        "tenant_id": kb.tenant_id,
        "kb_id": kb.id,
        "name": kb.name,
        "kb_parser_config": kb_parser_config,
        "pagerank": 0,
    }
    
    def callback(msg):
        logger.info(f"[GraphRAG] {msg}")
    
    try:
        result = await run_graphrag_for_kb(
            row=row,
            doc_ids=doc_ids,
            language=kb.language or "English",
            kb_parser_config=kb_parser_config,
            chat_model=chat_mdl,
            embedding_model=embed_mdl,
            callback=callback,
            with_resolution=graphrag_config.get('with_resolution', True),
            with_community=graphrag_config.get('with_community', True),
            max_parallel_docs=indexing_config.get('max_parallel_docs', 4),
        )
        
        logger.info(f"GraphRAG indexing completed: {result}")
        return result
    except Exception as e:
        logger.error(f"GraphRAG indexing failed: {e}", exc_info=True)
        raise


async def run_raptor_indexing(config: dict, kb: dict, tenant: dict, chat_mdl, embed_mdl):
    """Run RAPTOR indexing to generate summary tree."""
    logger.info("Starting RAPTOR indexing...")
    
    raptor_config = config.get('raptor', {})
    if not raptor_config.get('enabled', True):
        logger.info("RAPTOR is disabled in configuration")
        return
    
    indexing_config = config.get('indexing', {})
    doc_ids = indexing_config.get('doc_ids', [])
    
    # Prepare parser config for RAPTOR
    kb_parser_config = {
        "raptor": {
            "use_raptor": True,
            "max_cluster": raptor_config.get('max_cluster', 64),
            "max_token": raptor_config.get('max_token', 512),
            "threshold": raptor_config.get('threshold', 0.1),
            "random_seed": raptor_config.get('random_seed', 42),
            "scope": raptor_config.get('scope', 'kb'),
            "prompt": raptor_config.get('prompt', 
                "Write a comprehensive summary of the following:\n{cluster_content}\n"
                "The summary should cover all the key points and main ideas presented.\nSUMMARY:")
        }
    }
    
    # Create a row dict simulating task structure
    row = {
        "id": "dual_index_raptor_task",
        "tenant_id": kb.tenant_id,
        "kb_id": kb.id,
        "name": kb.name,
        "pagerank": 0,
    }
    
    def callback(msg):
        logger.info(f"[RAPTOR] {msg}")
    
    try:
        result = await run_raptor_for_kb(
            row=row,
            kb_parser_config=kb_parser_config,
            chat_mdl=chat_mdl,
            embd_mdl=embed_mdl,
            vector_size=len(embed_mdl.encode(["test"])[0][0]),
            callback=callback,
            doc_ids=doc_ids,
        )
        
        logger.info(f"RAPTOR indexing completed: {len(result)} chunks generated")
        return result
    except Exception as e:
        logger.error(f"RAPTOR indexing failed: {e}", exc_info=True)
        raise


def perform_dual_retrieval(config: dict, query: str, kb: dict, tenant: dict, embed_mdl, chat_mdl):
    """Perform dual retrieval using both concept graph and summary tree."""
    logger.info(f"Starting dual retrieval for query: {query}")
    
    retrieval_config = config.get('retrieval', {})
    graphrag_retrieval_config = retrieval_config.get('graphrag', {})
    raptor_retrieval_config = retrieval_config.get('raptor', {})
    
    results = {
        'query': query,
        'graphrag_chunks': [],
        'raptor_chunks': [],
        'standard_chunks': [],
        'combined_chunks': []
    }
    
    # 1. Standard chunk retrieval
    logger.info("Performing standard chunk retrieval...")
    try:
        standard_results = settings.retriever.retrieval(
            query,
            embed_mdl,
            [kb.tenant_id],
            [kb.id],
            1,
            retrieval_config.get('top_n', 6),
            retrieval_config.get('similarity_threshold', 0.3),
            1 - retrieval_config.get('keywords_similarity_weight', 0.5),
        )
        results['standard_chunks'] = standard_results.get('chunks', [])
        logger.info(f"Retrieved {len(results['standard_chunks'])} standard chunks")
    except Exception as e:
        logger.error(f"Standard retrieval failed: {e}")
    
    # 2. GraphRAG (Concept Graph) retrieval
    if graphrag_retrieval_config.get('enabled', True):
        logger.info("Performing GraphRAG (concept graph) retrieval...")
        try:
            kg_search = KGSearch(settings.docStoreConn)
            graphrag_result = kg_search.retrieval(
                question=query,
                tenant_ids=[kb.tenant_id],
                kb_ids=[kb.id],
                emb_mdl=embed_mdl,
                llm=chat_mdl,
                ent_topn=graphrag_retrieval_config.get('ent_topn', 6),
                rel_topn=graphrag_retrieval_config.get('rel_topn', 6),
                comm_topn=graphrag_retrieval_config.get('comm_topn', 1),
                ent_sim_threshold=graphrag_retrieval_config.get('ent_sim_threshold', 0.3),
                rel_sim_threshold=graphrag_retrieval_config.get('rel_sim_threshold', 0.3),
            )
            if graphrag_result and graphrag_result.get('content_with_weight'):
                results['graphrag_chunks'].append(graphrag_result)
                logger.info("GraphRAG retrieval successful")
            else:
                logger.warning("GraphRAG retrieval returned no results")
        except Exception as e:
            logger.error(f"GraphRAG retrieval failed: {e}")
    
    # 3. RAPTOR (Summary Tree) retrieval
    if raptor_retrieval_config.get('enabled', True):
        logger.info("Performing RAPTOR (summary tree) retrieval...")
        try:
            # Use direct search with raptor_kwd filter
            req = {
                "kb_ids": [kb.id],
                "question": query,
                "page": 1,
                "size": raptor_retrieval_config.get('top_n', 3),
                "similarity": raptor_retrieval_config.get('similarity_threshold', 0.2),
                "raptor_kwd": "raptor",  # Filter for RAPTOR chunks
                "available_int": 1,
            }
            
            sres = settings.retriever.search(
                req,
                [search.index_name(kb.tenant_id)],
                [kb.id],
                embed_mdl
            )
            
            # Convert search results to chunks format
            raptor_chunks = []
            for chunk_id in sres.ids:
                chunk = sres.field[chunk_id]
                raptor_chunks.append({
                    "chunk_id": chunk_id,
                    "content_with_weight": chunk.get("content_with_weight", ""),
                    "doc_id": chunk.get("doc_id", ""),
                    "docnm_kwd": chunk.get("docnm_kwd", ""),
                    "kb_id": chunk.get("kb_id", ""),
                    "similarity": chunk.get("_score", 0.0),
                })
            
            results['raptor_chunks'] = raptor_chunks
            logger.info(f"Retrieved {len(results['raptor_chunks'])} RAPTOR chunks")
        except Exception as e:
            logger.error(f"RAPTOR retrieval failed: {e}")
    
    # 4. Combine results
    combination_strategy = retrieval_config.get('combination', 'hybrid')
    max_total_chunks = retrieval_config.get('max_total_chunks', 10)
    
    if combination_strategy == 'hybrid':
        # Prioritize: GraphRAG > RAPTOR > Standard
        combined = []
        if results['graphrag_chunks']:
            combined.extend(results['graphrag_chunks'])
        if results['raptor_chunks']:
            combined.extend(results['raptor_chunks'][:max(1, max_total_chunks // 3)])
        if results['standard_chunks']:
            remaining = max_total_chunks - len(combined)
            combined.extend(results['standard_chunks'][:remaining])
    elif combination_strategy == 'sequential':
        # Use only the first non-empty result
        if results['graphrag_chunks']:
            combined = results['graphrag_chunks']
        elif results['raptor_chunks']:
            combined = results['raptor_chunks'][:max_total_chunks]
        else:
            combined = results['standard_chunks'][:max_total_chunks]
    else:  # parallel
        # Mix all results evenly
        combined = []
        max_per_type = max_total_chunks // 3
        if results['graphrag_chunks']:
            combined.extend(results['graphrag_chunks'][:max_per_type])
        if results['raptor_chunks']:
            combined.extend(results['raptor_chunks'][:max_per_type])
        if results['standard_chunks']:
            combined.extend(results['standard_chunks'][:max_per_type])
    
    results['combined_chunks'] = combined[:max_total_chunks]
    logger.info(f"Combined {len(results['combined_chunks'])} chunks using {combination_strategy} strategy")
    
    return results


def print_retrieval_results(results: dict):
    """Print retrieval results in a readable format."""
    print("\n" + "="*80)
    print(f"DUAL RETRIEVAL RESULTS FOR QUERY: {results['query']}")
    print("="*80)
    
    print(f"\n[GraphRAG] Retrieved {len(results['graphrag_chunks'])} chunks")
    for i, chunk in enumerate(results['graphrag_chunks'], 1):
        print(f"\n--- GraphRAG Chunk {i} ---")
        content = chunk.get('content_with_weight', '')[:500]
        print(f"Content: {content}...")
    
    print(f"\n[RAPTOR] Retrieved {len(results['raptor_chunks'])} chunks")
    for i, chunk in enumerate(results['raptor_chunks'], 1):
        print(f"\n--- RAPTOR Chunk {i} ---")
        content = chunk.get('content_with_weight', '')[:500]
        print(f"Content: {content}...")
    
    print(f"\n[Standard] Retrieved {len(results['standard_chunks'])} chunks")
    
    print(f"\n[Combined] Total {len(results['combined_chunks'])} chunks")
    print("="*80 + "\n")


async def main():
    parser = argparse.ArgumentParser(
        description='Dual Index - Concept Graph and Summary Tree Retrieval'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['index', 'retrieve', 'both'],
        default='both',
        help='Operation mode: index, retrieve, or both'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Query string for retrieval (required for retrieve/both modes)'
    )
    
    args = parser.parse_args()
    
    # Initialize settings
    settings.init_settings()
    
    # Load and validate configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    validate_config(config)
    
    # Get knowledge base info
    kb_config = config['knowledgebase']
    tenant_id = kb_config['tenant_id']
    kb_id = kb_config['kb_id']
    
    # Load tenant and KB
    _, tenant = TenantService.get_by_id(tenant_id)
    if not tenant:
        raise ValueError(f"Tenant {tenant_id} not found")
    
    _, kb = KnowledgebaseService.get_by_id(kb_id)
    if not kb:
        raise ValueError(f"Knowledge base {kb_id} not found")
    
    logger.info(f"Tenant: {tenant.name}, KB: {kb.name}")
    
    # Initialize LLM models
    llm_config = config['llm']
    chat_mdl = LLMBundle(tenant_id, LLMType.CHAT, tenant.llm_id)
    embed_mdl = LLMBundle(tenant_id, LLMType.EMBEDDING, kb.embd_id)
    
    # Run indexing if needed
    if args.mode in ['index', 'both']:
        logger.info("Starting indexing phase...")
        
        # Run GraphRAG indexing
        if config.get('graphrag', {}).get('enabled', True):
            await run_graphrag_indexing(config, kb, tenant, chat_mdl, embed_mdl)
        
        # Run RAPTOR indexing
        if config.get('raptor', {}).get('enabled', True):
            await run_raptor_indexing(config, kb, tenant, chat_mdl, embed_mdl)
        
        logger.info("Indexing phase completed")
    
    # Run retrieval if needed
    if args.mode in ['retrieve', 'both']:
        if not args.query:
            raise ValueError("--query is required for retrieve/both modes")
        
        logger.info("Starting retrieval phase...")
        results = perform_dual_retrieval(config, args.query, kb, tenant, embed_mdl, chat_mdl)
        print_retrieval_results(results)
        logger.info("Retrieval phase completed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
