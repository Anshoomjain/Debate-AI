# DEBATEAI - Integration & Testing Guide
## Next Steps After Adding Agents

---

## 📋 WHAT YOU'VE COMPLETED
✅ Core interfaces (BaseAgent, BaseRetriever, Document, DebateState)
✅ All 5 agents (Pro, Con, Judge, Reporter, FactChecker)
✅ Configuration files (agents_config.yaml, prompts)
✅ Orchestrator logic
✅ CLI interface

---

## 🎯 WHAT TO DO NOW

### PHASE 1: Create Missing Components (30 minutes)

#### Step 1.1: Copy Files to Your Project
Copy these files I just created to your project:

```bash
# From Claude's workspace to your D:\FSP folder
cp document_processor.py D:/FSP/core/
cp faiss_retriever.py D:/FSP/retrievers/
```

#### Step 1.2: Verify File Structure
Your D:\FSP should have:
```
D:\FSP\
├── agents\
│   ├── __init__.py
│   ├── pro.py
│   ├── con.py
│   ├── judge.py
│   ├── reporter.py
│   └── fact_checker.py
├── core\
│   ├── __init__.py
│   ├── interfaces.py
│   ├── orchestrator.py
│   ├── hybrid_retriever.py
│   └── document_processor.py  ← NEW
├── retrievers\
│   ├── __init__.py
│   ├── faiss_retriever.py  ← NEW
│   └── bm25_retriever.py
├── config\
│   ├── agents_config.yaml
│   └── prompts\
│       ├── pro.txt
│       ├── con.txt
│       ├── judge.txt
│       └── reporter.txt
├── data\
│   ├── raw\          ← Put your PDFs/CSVs here
│   └── processed\
├── outputs\          ← Reports will be saved here
├── cli.py
└── requirements.txt
```

---

### PHASE 2: Install Dependencies (10 minutes)

#### Step 2.1: Check Ollama is Running
```bash
# Open a terminal and run:
ollama serve

# In another terminal, verify models:
ollama list
```

You should see:
- llama3.1:8b
- mistral:7b
- phi3:medium
- phi3:mini

If missing, download them:
```bash
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull phi3:medium
ollama pull phi3:mini
```

#### Step 2.2: Install Python Libraries
```bash
cd D:\FSP
pip install -r requirements.txt
```

This installs:
- llama-index (RAG framework)
- faiss-cpu or faiss-gpu (vector search)
- rank-bm25 (keyword search)
- sentence-transformers (embeddings)
- ollama (LLM client)
- pandas, numpy, pypdf (data processing)
- pyyaml (config)

---

### PHASE 3: Test Individual Components (20 minutes)

#### Test 3.1: Document Processor
```bash
cd D:\FSP
python core/document_processor.py
```

Expected output:
```
Testing Document Processor
==========================
✓ DocumentProcessor initialized
✓ Loaded X chunks
First chunk: ...
```

#### Test 3.2: FAISS Retriever
```bash
python retrievers/faiss_retriever.py
```

Expected output:
```
Testing FAISS Retriever
=======================
Loading embedding model: all-MiniLM-L6-v2...
✓ FAISSRetriever initialized
Indexing 3 documents...
✓ Indexed 3 documents
Query: 'How much copper do electric vehicles need?'
1. Score: 0.856
   Text: Electric vehicles require 83kg...
```

#### Test 3.3: BM25 Retriever
```bash
python retrievers/bm25_retriever.py
```

#### Test 3.4: Hybrid Retriever
```bash
python core/hybrid_retriever.py
```

#### Test 3.5: Pro Agent
```bash
python agents/pro.py
```

#### Test 3.6: Con Agent
```bash
python agents/con.py
```

#### Test 3.7: Judge Agent
```bash
python agents/judge.py
```

---

### PHASE 4: Test Full System (15 minutes)

#### Test 4.1: Orchestrator Demo
```bash
python core/orchestrator.py
```

This runs a demo debate with sample data.

Expected output:
```
DEBATEAI Orchestrator
=====================
Initializing agents...
✓ ProAgent initialized
✓ ConAgent initialized
✓ JudgeAgent initialized

DEBATE: Should I invest in copper?
===================================
[Retrieval] Searching...
✓ Retrieved 5 documents

ROUND 1: Initial Arguments
✓ Pro arguments complete
✓ Con arguments complete

ROUND 2: Rebuttals
✓ Pro rebuttal complete
✓ Con rebuttal complete

ROUND 3: Judge Synthesis
✓ Verdict delivered

Verdict: FAVORABLE
Trust Score: 78.5%
```

---

### PHASE 5: Add Real Data (20 minutes)

#### Step 5.1: Collect Documents
Put 10-20 documents in `D:\FSP\data\raw\`:
- PDFs (research reports, news articles)
- CSVs (price data, statistics)
- TXT files (news, analysis)

Example structure:
```
data/raw/
├── pdfs/
│   ├── IEA_EV_Report_2024.pdf
│   ├── USGS_Copper_2024.pdf
│   └── Market_Analysis.pdf
├── csvs/
│   ├── copper_prices_2024.csv
│   └── ev_sales_data.csv
└── news/
    ├── reuters_copper_jan2025.txt
    └── bloomberg_ev_trends.txt
```

#### Step 5.2: Process Documents
Create a script `process_data.py`:

```python
from core.document_processor import DocumentProcessor
from pathlib import Path

# Initialize processor
processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)

# Load all documents from data/raw
documents = processor.load_documents("data/raw")

# Save processed chunks
Path("data/processed").mkdir(exist_ok=True)
processor.save_chunks(documents, "data/processed/chunks.json")

print(f"✓ Processed {len(documents)} chunks!")
```

Run it:
```bash
python process_data.py
```

---

### PHASE 6: Run Your First Real Debate (10 minutes)

#### Option A: Using CLI
```bash
python cli.py
```

Then:
1. Choose "1" to index documents
2. Choose "2" to run debate
3. Enter your question: "Should I invest in copper given current EV trends?"

#### Option B: Using Python Script
Create `run_debate.py`:

```python
from core.orchestrator import DebateOrchestrator
from core.document_processor import DocumentProcessor

# Load processed documents
processor = DocumentProcessor()
documents = processor.load_chunks("data/processed/chunks.json")

# Initialize orchestrator
orchestrator = DebateOrchestrator()
orchestrator.index_documents(documents)

# Run debate
query = "Should I invest in copper given current EV trends?"
debate_state = orchestrator.run_debate(query, top_k=5)

# Display report
report = orchestrator.format_debate_report(debate_state)
print(report)

# Save report
orchestrator.save_debate(debate_state, "outputs/my_first_debate.txt")
```

Run it:
```bash
python run_debate.py
```

---

## 🐛 TROUBLESHOOTING

### Problem: "ModuleNotFoundError: No module named 'core'"
**Solution:**
```bash
# Make sure __init__.py exists in all folders
touch core/__init__.py
touch agents/__init__.py
touch retrievers/__init__.py

# Or run from project root:
cd D:\FSP
python -m cli
```

### Problem: "Ollama connection failed"
**Solution:**
```bash
# Start Ollama server in separate terminal
ollama serve

# Verify it's running
curl http://localhost:11434
```

### Problem: "CUDA out of memory"
**Solution:**
Use CPU mode or smaller models:
```python
# In agents/__init__.py or config
# Change all models to phi3:mini
```

### Problem: "No documents indexed"
**Solution:**
```bash
# Check data folder has files
ls data/raw/

# Run document processor first
python core/document_processor.py

# Then run orchestrator
python core/orchestrator.py
```

---

## 📊 EXPECTED TIMELINE

- ✅ **You are here**: Agents created
- ⏱️ **Phase 1** (30 min): File setup
- ⏱️ **Phase 2** (10 min): Dependencies
- ⏱️ **Phase 3** (20 min): Component tests
- ⏱️ **Phase 4** (15 min): System test
- ⏱️ **Phase 5** (20 min): Real data
- ⏱️ **Phase 6** (10 min): First debate

**Total:** ~2 hours to fully working system!

---

## 🎯 SUCCESS CRITERIA

You'll know it's working when:
1. ✅ All component tests pass
2. ✅ Orchestrator demo runs without errors
3. ✅ You get a debate report with:
   - Pro arguments with citations
   - Con arguments with citations
   - Judge's verdict
   - Trust score (50-100%)
4. ✅ Report saved to `outputs/` folder

---

## 📝 QUICK COMMAND REFERENCE

```bash
# Start Ollama
ollama serve

# Test individual components
python core/document_processor.py
python retrievers/faiss_retriever.py
python agents/pro.py

# Run full system
python core/orchestrator.py

# Run CLI
python cli.py

# Process your data
python process_data.py

# Run custom debate
python run_debate.py
```

---

## 🚀 NEXT STEPS AFTER FIRST DEBATE

Once you have a working debate:

1. **Week 3 Tasks**:
   - Add Fact-Checker agent (already in code)
   - Add Reporter agent (already in code)
   - Expand to 5-round debates

2. **Week 4 Tasks**:
   - Build Streamlit web UI
   - Add visualizations (trust score gauge, charts)
   - Run evaluation metrics

3. **Documentation**:
   - Write academic paper
   - Create presentation
   - Record demo video

---

## 🆘 GETTING HELP

If stuck:
1. Check error messages carefully
2. Verify Ollama is running
3. Check file paths are correct
4. Test components individually
5. Ask me specific questions!

---

**Ready to proceed? Start with Phase 1!**
