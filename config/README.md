# Dual Index Configuration Guide

## Overview

The Dual Index system enables simultaneous use of two advanced RAG techniques:
1. **GraphRAG (Concept Graph)**: Extracts entities and relationships to build a knowledge graph
2. **RAPTOR (Summary Tree)**: Creates hierarchical summaries using recursive clustering

## Setup

### 1. Install Dependencies

```bash
# Install Python dependencies using uv
uv sync --python 3.12 --all-extras
uv run download_deps.py

# OR use pip
pip install -r requirements.txt
```

### 2. Start Backend Services

```bash
# Start required services (MySQL, Elasticsearch/Infinity, Redis, MinIO)
docker compose -f docker/docker-compose-base.yml up -d
```

### 3. Configure the System

Edit `config/dual_index_config.yaml` and update:
- `knowledgebase.tenant_id`: Your tenant ID
- `knowledgebase.kb_id`: Your knowledge base ID
- `llm.chat_model`: Your chat model (if different from default)
- `llm.embedding_model`: Your embedding model (if different from default)

## Usage

### Quick Start

For a quick overview of usage examples, run:
```bash
./config/example_usage.sh
```

### Indexing Only

Generate both concept graph and summary tree indices:

```bash
python main.py --config ./config/dual_index_config.yaml --mode index
```

### Retrieval Only

Perform dual retrieval on existing indices:

```bash
python main.py --config ./config/dual_index_config.yaml --mode retrieve --query "Your question here"
```

### Both Indexing and Retrieval

```bash
python main.py --config ./config/dual_index_config.yaml --mode both --query "Your question here"
```

## Configuration Options

### GraphRAG Settings

```yaml
graphrag:
  enabled: true
  method: "light"  # or "general" for more detailed extraction
  entity_types: []  # Leave empty for auto-detection
  with_resolution: true  # Enable entity resolution
  with_community: true   # Enable community detection
```

### RAPTOR Settings

```yaml
raptor:
  enabled: true
  max_cluster: 64  # Maximum clusters per layer
  max_token: 512   # Maximum tokens in summary
  threshold: 0.1   # Cluster assignment threshold
  random_seed: 42
  scope: "kb"      # "kb" for knowledge base level, "file" for document level
```

### Retrieval Settings

```yaml
retrieval:
  top_n: 6  # Number of standard chunks
  similarity_threshold: 0.3
  
  graphrag:
    enabled: true
    ent_topn: 6  # Top entities
    rel_topn: 6  # Top relationships
    
  raptor:
    enabled: true
    top_n: 3  # Number of RAPTOR summaries
    
  combination: "hybrid"  # How to combine results
  max_total_chunks: 10   # Maximum chunks in final result
```

## Combination Strategies

### Hybrid (Recommended)
Prioritizes GraphRAG, then RAPTOR, then standard chunks:
```yaml
combination: "hybrid"
```

### Sequential
Uses only the first non-empty result source:
```yaml
combination: "sequential"
```

### Parallel
Mixes all sources evenly:
```yaml
combination: "parallel"
```

## Troubleshooting

### Concept Graph Not Generated

**Issue**: GraphRAG indexing completes but no graph is visible in retrieval.

**Solutions**:
1. Check that `graphrag.enabled` is `true` in config
2. Verify documents are properly chunked in the knowledge base
3. Check logs for entity extraction errors
4. Try using `method: "general"` for more robust extraction

### RAPTOR Indexing Fails

**Issue**: RAPTOR indexing throws timeout or clustering errors.

**Solutions**:
1. Reduce `max_cluster` value (try 32 or 16)
2. Increase timeout by setting environment variable:
   ```bash
   export ENABLE_TIMEOUT_ASSERTION=0
   ```
3. Check that documents have sufficient content for clustering
4. Verify LLM model is accessible and responding

### Retrieval Returns Empty Results

**Issue**: Retrieval completes but returns no chunks.

**Solutions**:
1. Lower `similarity_threshold` values in retrieval config
2. Verify indices were created successfully (check previous logs)
3. Check that tenant_id and kb_id are correct
4. Try retrieval with each source separately:
   - Disable GraphRAG: `retrieval.graphrag.enabled: false`
   - Disable RAPTOR: `retrieval.raptor.enabled: false`

### Import Errors

**Issue**: `ModuleNotFoundError` when running main.py

**Solutions**:
1. Ensure you're in the project root directory
2. Install dependencies: `uv sync` or `pip install -r requirements.txt`
3. Activate virtual environment if using one:
   ```bash
   source .venv/bin/activate  # Linux/Mac
   # OR
   .venv\Scripts\activate  # Windows
   ```

## Advanced Usage

### Processing Specific Documents

To index only specific documents, add their IDs to the config:

```yaml
indexing:
  doc_ids: ["doc_id_1", "doc_id_2"]
```

### Parallel Processing

Adjust parallel processing for large knowledge bases:

```yaml
indexing:
  max_parallel_docs: 8  # Increase for faster processing (requires more memory)
```

### Custom Prompts

Customize the RAPTOR summarization prompt:

```yaml
raptor:
  prompt: |
    Summarize the following content focusing on key technical details:
    {cluster_content}
    
    Provide a concise summary that captures:
    1. Main concepts
    2. Important relationships
    3. Critical details
    
    SUMMARY:
```

## Examples

### Academic Papers

```yaml
graphrag:
  method: "general"
  entity_types: ["researcher", "methodology", "finding", "dataset"]
  
raptor:
  max_cluster: 32
  max_token: 1024
```

### Technical Documentation

```yaml
graphrag:
  method: "light"
  
raptor:
  max_cluster: 64
  max_token: 512
  scope: "file"
```

### Legal Documents

```yaml
graphrag:
  method: "general"
  entity_types: ["law", "case", "statute", "precedent"]
  with_community: false  # Focus on direct relationships
  
raptor:
  max_cluster: 48
  threshold: 0.15
```
