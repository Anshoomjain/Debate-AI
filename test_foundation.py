"""
Test script to verify foundation is working
Run this after saving all base files
"""

import sys
import yaml
from pathlib import Path

print("=" * 60)
print("DEBATEAI - Foundation Test")
print("=" * 60)

# Test 1: Import core interfaces
print("\n[Test 1] Importing core interfaces...")
try:
    from core.interfaces import (
        BaseAgent, BaseRetriever, BaseScorer, BaseOrchestrator,
        Document, DebateState
    )
    print("✓ All base classes imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Load agent config
print("\n[Test 2] Loading agent configuration...")
try:
    with open('config/agents_config.yaml', 'r') as f:
        agent_config = yaml.safe_load(f)
    
    active_agents = agent_config['active_agents']
    print(f"✓ Config loaded: {len(active_agents)} agents defined")
    
    for agent in active_agents:
        if agent['enabled']:
            print(f"  - {agent['name']}: {agent['model']}")
except Exception as e:
    print(f"✗ Config load failed: {e}")
    sys.exit(1)

# Test 3: Load retrieval config
print("\n[Test 3] Loading retrieval configuration...")
try:
    with open('config/retrieval_config.yaml', 'r') as f:
        retrieval_config = yaml.safe_load(f)
    
    retrievers = retrieval_config['retrievers']
    enabled = [r for r in retrievers if r['enabled']]
    print(f"✓ Config loaded: {len(enabled)} retrievers enabled")
    
    for ret in enabled:
        print(f"  - {ret['name']}: weight={ret['weight']}")
except Exception as e:
    print(f"✗ Config load failed: {e}")
    sys.exit(1)

# Test 4: Check prompts exist
print("\n[Test 4] Checking prompt templates...")
prompt_dir = Path('config/prompts')
expected_prompts = ['pro.txt', 'con.txt', 'judge.txt']

for prompt_file in expected_prompts:
    path = prompt_dir / prompt_file
    if path.exists():
        content = path.read_text()
        print(f"✓ {prompt_file}: {len(content)} characters")
    else:
        print(f"✗ {prompt_file}: NOT FOUND")
        sys.exit(1)

# Test 5: Create test objects
print("\n[Test 5] Creating test objects...")
try:
    # Test Document creation
    doc = Document(
        text="Test document content",
        source="test.pdf",
        chunk_id="test_001",
        metadata={'date': '2026-01-01'},
        score=0.95
    )
    print(f"✓ Document created: {doc.chunk_id}")
    
    # Test DebateState creation
    state = DebateState(query="Should I invest in copper?")
    state.add_round("pro", "Bullish argument here")
    state.add_round("con", "Bearish argument here")
    print(f"✓ DebateState created: {len(state.rounds)} rounds")
    
except Exception as e:
    print(f"✗ Object creation failed: {e}")
    sys.exit(1)

# Test 6: Verify folder structure
print("\n[Test 6] Verifying folder structure...")
required_folders = [
    'core', 'agents', 'retrievers', 'scorers', 
    'interfaces', 'config', 'data', 'tests'
]

for folder in required_folders:
    path = Path(folder)
    if path.exists() and path.is_dir():
        print(f"✓ {folder}/")
    else:
        print(f"✗ {folder}/ NOT FOUND")
        sys.exit(1)

print("\n" + "=" * 60)
print("SUCCESS: Foundation is ready! 🎉")
print("=" * 60)
print("\nNext steps:")
print("1. Implement FAISSRetriever in retrievers/faiss_retriever.py")
print("2. Implement BM25Retriever in retrievers/bm25_retriever.py")
print("3. Implement ProAgent in agents/pro.py")
print("=" * 60)
