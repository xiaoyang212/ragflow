#!/bin/bash
#
# Example usage script for dual index system
# This script demonstrates how to use the dual retrieval functionality

set -e

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Dual Index System - Example Usage ===${NC}\n"

# Check if configuration is set up
if ! grep -q "your_tenant_id" config/dual_index_config.yaml 2>/dev/null; then
    echo -e "${GREEN}✓ Configuration appears to be customized${NC}"
else
    echo -e "${YELLOW}⚠ Warning: Configuration still contains placeholder values${NC}"
    echo "Please edit config/dual_index_config.yaml or set environment variables:"
    echo "  export TENANT_ID=your_actual_tenant_id"
    echo "  export KB_ID=your_actual_kb_id"
    echo ""
fi

# Example 1: Index only
echo -e "${GREEN}Example 1: Generate indices for concept graph and summary tree${NC}"
echo "Command:"
echo "  python main.py --config ./config/dual_index_config.yaml --mode index"
echo ""

# Example 2: Retrieve only
echo -e "${GREEN}Example 2: Perform dual retrieval${NC}"
echo "Command:"
echo "  python main.py --config ./config/dual_index_config.yaml --mode retrieve \\"
echo '    --query "What are the main concepts in this knowledge base?"'
echo ""

# Example 3: Both index and retrieve
echo -e "${GREEN}Example 3: Index and retrieve in one operation${NC}"
echo "Command:"
echo "  python main.py --config ./config/dual_index_config.yaml --mode both \\"
echo '    --query "Explain the key relationships between entities"'
echo ""

# Example 4: Using environment variables
echo -e "${GREEN}Example 4: Using environment variables${NC}"
echo "Commands:"
echo "  export TENANT_ID=abc123"
echo "  export KB_ID=xyz789"
echo "  python main.py --config ./config/dual_index_config.yaml --mode both \\"
echo '    --query "Summarize the main topics"'
echo ""

# Example 5: Custom retrieval strategies
echo -e "${GREEN}Example 5: Different combination strategies${NC}"
echo "Edit config/dual_index_config.yaml and change:"
echo "  retrieval:"
echo "    combination: 'hybrid'     # Prioritize GraphRAG > RAPTOR > Standard"
echo "    # OR"
echo "    combination: 'sequential'  # Use first non-empty result"
echo "    # OR"
echo "    combination: 'parallel'    # Mix all sources evenly"
echo ""

echo -e "${GREEN}For more information, see:${NC}"
echo "  - config/README.md - Complete documentation"
echo "  - config/dual_index_config.yaml - Configuration options"
echo ""

echo -e "${GREEN}Quick Start:${NC}"
echo "1. Update config/dual_index_config.yaml with your tenant_id and kb_id"
echo "2. Ensure backend services are running (MySQL, ES/Infinity, Redis, MinIO)"
echo "3. Run indexing: python main.py --config ./config/dual_index_config.yaml --mode index"
echo "4. Run retrieval: python main.py --config ./config/dual_index_config.yaml --mode retrieve --query 'your question'"
