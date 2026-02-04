"""
Enhanced Debate Orchestrator - 5-Agent System
==============================================
Coordinates 5-round debate with Fact-Checker and Reporter.

Author: [Your Name]
Date: January 2026
"""

from typing import List, Dict
from datetime import datetime
import yaml
from pathlib import Path

from core.interfaces import BaseOrchestrator, DebateState, Document
from core.hybrid_retriever import HybridRetriever
from retrievers.faiss_retriever import FAISSRetriever
from retrievers.bm25_retriever import BM25Retriever
from agents.pro import ProAgent
from agents.con import ConAgent
from agents.judge import JudgeAgent
from agents.fact_checker import FactCheckerAgent
from agents.reporter import ReporterAgent


class EnhancedDebateOrchestrator(BaseOrchestrator):
    """
    Orchestrates complete 5-round debate:
    - Round 1: Pro and Con present initial arguments
    - Round 2: Pro and Con respond to each other (rebuttals)
    - Round 3: Fact-Checker validates all claims
    - Round 4: Judge synthesizes with fact-check results
    - Round 5: Reporter creates executive summary
    """
    
    def __init__(self, config_path: str = "config/agents_config.yaml"):
        super().__init__()
        
        print("\n" + "=" * 60)
        print("Initializing Enhanced DEBATEAI Orchestrator (5 Agents)")
        print("=" * 60)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize retrievers
        print("\n[1] Setting up retrievers...")
        self._setup_retrievers()
        
        # Initialize all 5 agents
        print("\n[2] Loading all debate agents...")
        self._setup_agents()
        
        print("\n" + "=" * 60)
        print("✓ Enhanced Orchestrator ready!")
        print(f"  - Agents: {list(self.agents.keys())}")
        print(f"  - Retrievers: {list(self.retrievers.keys())}")
        print("=" * 60)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠ Config not found, using defaults")
            return {'debate_settings': {'max_rounds': 5}}
    
    def _setup_retrievers(self):
        """Initialize retrieval system"""
        faiss = FAISSRetriever(device="cpu")  # Change to "cuda" if GPU available
        bm25 = BM25Retriever()
        
        self.hybrid_retriever = HybridRetriever(
            retrievers=[faiss, bm25],
            weights={'faiss': 0.6, 'bm25': 0.4}
        )
        
        self.add_retriever(faiss)
        self.add_retriever(bm25)
    
    def _setup_agents(self):
        """Initialize all 5 debate agents"""
        # Create agents
        pro = ProAgent()
        con = ConAgent()
        fact_checker = FactCheckerAgent()
        judge = JudgeAgent()
        reporter = ReporterAgent()
        
        # Add to orchestrator
        self.add_agent(pro)
        self.add_agent(con)
        self.add_agent(fact_checker)
        self.add_agent(judge)
        self.add_agent(reporter)
    
    def index_documents(self, documents: List[Document]):
        """Index documents for retrieval"""
        print(f"\nIndexing {len(documents)} documents...")
        self.hybrid_retriever.add_documents(documents)
        print("✓ Documents indexed and ready for retrieval")
    
    def run_debate(self, query: str, top_k: int = 5) -> DebateState:
        """
        Run complete 5-round debate.
        
        Args:
            query: User's question
            top_k: Number of documents to retrieve
            
        Returns:
            DebateState with complete debate and final report
        """
        print("\n" + "=" * 60)
        print(f"ENHANCED DEBATE: {query}")
        print("=" * 60)
        
        # Initialize debate state
        debate_state = DebateState(query=query)
        
        # Retrieve relevant documents
        print("\n[Retrieval] Searching knowledge base...")
        context_docs = self.hybrid_retriever.retrieve(query, top_k=top_k)
        debate_state.context_docs = context_docs
        
        print(f"✓ Retrieved {len(context_docs)} relevant documents")
        for i, doc in enumerate(context_docs, 1):
            print(f"  {i}. {doc.source} (score: {doc.score:.3f})")
        
        # ROUND 1: Initial Arguments
        print("\n" + "=" * 60)
        print("ROUND 1: Initial Arguments")
        print("=" * 60)
        
        print("\n[Pro Agent] Generating bullish arguments...")
        pro_args = self.agents['pro'].generate(query, context_docs, debate_state)
        debate_state.add_round('pro', pro_args)
        print("✓ Pro arguments complete")
        self._free_gpu_memory()
        
        print("\n[Con Agent] Generating bearish arguments...")
        con_args = self.agents['con'].generate(query, context_docs, debate_state)
        debate_state.add_round('con', con_args)
        print("✓ Con arguments complete")
        self._free_gpu_memory()
        
        # ROUND 2: Rebuttals
        debate_state.next_round()
        print("\n" + "=" * 60)
        print("ROUND 2: Rebuttals")
        print("=" * 60)
        
        print("\n[Pro Agent] Responding to Con's concerns...")
        pro_rebuttal = self.agents['pro'].generate(query, context_docs, debate_state)
        debate_state.add_round('pro', pro_rebuttal)
        print("✓ Pro rebuttal complete")
        self._free_gpu_memory()
        
        print("\n[Con Agent] Countering Pro's optimism...")
        con_rebuttal = self.agents['con'].generate(query, context_docs, debate_state)
        debate_state.add_round('con', con_rebuttal)
        print("✓ Con rebuttal complete")
        self._free_gpu_memory()
        
        # ROUND 3: Fact-Checking
        debate_state.next_round()
        print("\n" + "=" * 60)
        print("ROUND 3: Fact-Checking")
        print("=" * 60)
        
        print("\n[Fact-Checker Agent] Validating claims...")
        fact_check = self.agents['fact_checker'].generate(query, context_docs, debate_state)
        debate_state.add_round('fact_checker', fact_check)
        print("✓ Fact-check complete")
        self._free_gpu_memory()
        
        # ROUND 4: Judge Synthesis
        debate_state.next_round()
        print("\n" + "=" * 60)
        print("ROUND 4: Judge Synthesis")
        print("=" * 60)
        
        print("\n[Judge Agent] Synthesizing arguments...")
        verdict = self.agents['judge'].generate(query, context_docs, debate_state)
        debate_state.add_round('judge', verdict)
        print("✓ Verdict delivered")
        self._free_gpu_memory()
        
        # ROUND 5: Final Report
        debate_state.next_round()
        print("\n" + "=" * 60)
        print("ROUND 5: Executive Summary")
        print("=" * 60)
        
        print("\n[Reporter Agent] Creating final report...")
        final_report = self.agents['reporter'].generate(query, context_docs, debate_state)
        debate_state.add_round('reporter', final_report)
        print("✓ Report complete")
        
        print("\n" + "=" * 60)
        print("5-ROUND DEBATE COMPLETE")
        print("=" * 60)
        print(f"Verdict: {debate_state.verdict}")
        print(f"Trust Score: {debate_state.trust_score:.1f}%")
        print("=" * 60)
        
        return debate_state
    
    def format_debate_report(self, debate_state: DebateState) -> str:
        """Format complete debate into readable report"""
        report = []
        report.append("=" * 70)
        report.append("     DEBATEAI - ENHANCED 5-AGENT DEBATE REPORT")
        report.append("=" * 70)
        report.append(f"\nQuery: {debate_state.query}")
        report.append(f"Date: {debate_state.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nFinal Verdict: {debate_state.verdict}")
        report.append(f"Trust Score: {debate_state.trust_score:.1f}%")
        
        # Round 1: Initial Arguments
        if len(debate_state.rounds) > 0:
            round1 = debate_state.rounds[0]
            report.append("\n" + "=" * 70)
            report.append("ROUND 1: Initial Arguments")
            report.append("=" * 70)
            
            if 'pro' in round1:
                report.append("\n✅ PRO ARGUMENTS:")
                report.append("-" * 70)
                report.append(round1['pro']['content'])
            
            if 'con' in round1:
                report.append("\n\n❌ CON ARGUMENTS:")
                report.append("-" * 70)
                report.append(round1['con']['content'])
        
        # Round 2: Rebuttals
        if len(debate_state.rounds) > 1:
            round2 = debate_state.rounds[1]
            report.append("\n\n" + "=" * 70)
            report.append("ROUND 2: Rebuttals")
            report.append("=" * 70)
            
            if 'pro' in round2:
                report.append("\n✅ PRO REBUTTAL:")
                report.append("-" * 70)
                report.append(round2['pro']['content'])
            
            if 'con' in round2:
                report.append("\n\n❌ CON REBUTTAL:")
                report.append("-" * 70)
                report.append(round2['con']['content'])
        
        # Round 3: Fact-Check
        if len(debate_state.rounds) > 2:
            round3 = debate_state.rounds[2]
            report.append("\n\n" + "=" * 70)
            report.append("ROUND 3: Fact-Check Verification")
            report.append("=" * 70)
            
            if 'fact_checker' in round3:
                report.append("\n🔍 FACT-CHECKER REPORT:")
                report.append("-" * 70)
                report.append(round3['fact_checker']['content'])
        
        # Round 4: Judge
        if len(debate_state.rounds) > 3:
            round4 = debate_state.rounds[3]
            report.append("\n\n" + "=" * 70)
            report.append("ROUND 4: Judge's Analysis")
            report.append("=" * 70)
            
            if 'judge' in round4:
                report.append("\n⚖️ JUDGE'S VERDICT:")
                report.append("-" * 70)
                report.append(round4['judge']['content'])
        
        # Round 5: Final Report
        if len(debate_state.rounds) > 4:
            round5 = debate_state.rounds[4]
            report.append("\n\n" + "=" * 70)
            report.append("ROUND 5: Executive Summary")
            report.append("=" * 70)
            
            if 'reporter' in round5:
                report.append("\n📊 FINAL REPORT:")
                report.append("-" * 70)
                report.append(round5['reporter']['content'])
        
        # Sources
        report.append("\n\n" + "=" * 70)
        report.append("SOURCES CONSULTED")
        report.append("=" * 70)
        
        for i, doc in enumerate(debate_state.context_docs, 1):
            report.append(f"\n[{i}] {doc.source}")
            report.append(f"    Relevance: {doc.score:.3f}")
            report.append(f"    Preview: {doc.text[:100]}...")
        
        report.append("\n" + "=" * 70)
        report.append("End of Enhanced Debate Report")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def save_debate(self, debate_state: DebateState, output_path: str):
        """Save debate report to file"""
        report = self.format_debate_report(debate_state)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ Debate saved to {output_path}")
    
    def _free_gpu_memory(self):
        """Free GPU memory between agent calls"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass


# Demo function
def demo_enhanced_debate():
    """Run enhanced 5-agent debate demo"""
    print("\n" + "=" * 60)
    print("DEBATEAI - Enhanced 5-Agent System Demo")
    print("=" * 60)
    
    # Create sample documents
    sample_docs = [
        Document(
            text="Copper prices increased 15% in 2024 due to strong electric vehicle demand. EVs require 83kg of copper per car, which is 4 times more than traditional vehicles. Global EV sales grew 23% year-over-year.",
            source="IEA_Global_EV_Outlook_2024.pdf",
            chunk_id="iea_001",
            metadata={'date': '2024-05-15'},
            score=0.95
        ),
        Document(
            text="Global copper production declined 2% in 2024 due to aging mines in Chile and Peru. USGS projects a supply deficit of 400,000 tonnes by 2026.",
            source="USGS_Copper_2024.pdf",
            chunk_id="usgs_001",
            metadata={'date': '2024-01-20'},
            score=0.92
        ),
        Document(
            text="Copper prices fluctuated between $3.80 and $4.70 per pound in Q4 2024, a 23% range indicating market volatility.",
            source="Yahoo_Finance_Copper.csv",
            chunk_id="yf_001",
            metadata={'date': '2024-12-31'},
            score=0.88
        ),
    ]
    
    # Initialize orchestrator
    orchestrator = EnhancedDebateOrchestrator()
    
    # Index documents
    orchestrator.index_documents(sample_docs)
    
    # Run debate
    query = "Should I invest in copper given current EV market trends?"
    debate_state = orchestrator.run_debate(query, top_k=3)
    
    # Display report
    print("\n\n")
    report = orchestrator.format_debate_report(debate_state)
    print(report)
    
    # Save to file
    output_path = "outputs/enhanced_debate_report.txt"
    Path("outputs").mkdir(exist_ok=True)
    orchestrator.save_debate(debate_state, output_path)


if __name__ == "__main__":
    demo_enhanced_debate()
