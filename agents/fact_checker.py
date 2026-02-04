"""
Fact-Checker Agent - Validates Claims Against Sources
======================================================
Verifies all arguments made by Pro and Con agents.

Author: [Your Name]
Date: January 2026
"""

from typing import List, Optional, Dict
import ollama
import re
import core



from core.interfaces import BaseAgent, Document, DebateState


class FactCheckerAgent(BaseAgent):
    """
    Fact-Checker Agent verifies claims against source documents.
    Ensures all arguments are grounded in evidence.
    """
    
    def __init__(self):
        super().__init__(
            name="fact_checker",
            model="phi3:mini",  # Fast and accurate for verification
            role="Claim validator and fact-checker"
        )
        
        print(f"✓ FactCheckerAgent initialized:")
        print(f"  - Model: {self.model}")
        print(f"  - Role: {self.role}")
    
    def generate(self, query: str, context: List[Document], 
                 debate_state: Optional[DebateState] = None) -> str:
        """
        Verify claims from Pro and Con agents.
        
        Args:
            query: User's question
            context: Retrieved documents
            debate_state: Debate state with Pro/Con arguments
            
        Returns:
            Fact-check report
        """
        if not debate_state or not debate_state.rounds:
            return "[Error: No arguments to fact-check]"
        
        # Get Pro and Con arguments
        pro_args = self._get_arguments(debate_state, 'pro')
        con_args = self._get_arguments(debate_state, 'con')
        
        # Extract claims
        all_claims = self._extract_claims(pro_args, con_args)
        
        if not all_claims:
            return "No specific claims found to verify."
        
        # Format context
        context_text = self.format_context(context)
        
        # Build verification prompt
        prompt = self._build_prompt(all_claims, context_text)
        
        # Generate fact-check report
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        'temperature': 0.1,  # Very low for factual verification
                        'num_predict': 600,
                    }
                )
                
                fact_check_report = response['response'].strip()
                
                # Calculate pass rate
                pass_rate = self._calculate_pass_rate(fact_check_report)
                
                # Add summary
                summary = f"\n\n{'='*60}\nFACT-CHECK SUMMARY\n{'='*60}\n"
                summary += f"Claims Verified: {len(all_claims)}\n"
                summary += f"Pass Rate: {pass_rate:.1f}%\n"
                summary += "='*60"
                
                return fact_check_report + summary
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠ Attempt {attempt + 1} failed, retrying...")
                    import time
                    time.sleep(2)
                else:
                    return f"[Error: Failed after {max_retries} attempts: {e}]"
    
    def _build_prompt(self, claims: List[str], context: str) -> str:
        """Build fact-checking prompt"""
        claims_text = "\n".join([f"{i+1}. {claim}" for i, claim in enumerate(claims)])
        
        return f"""You are a fact-checker verifying claims against source documents.

CONTEXT DOCUMENTS:
{context}

CLAIMS TO VERIFY:
{claims_text}

YOUR TASK: For EACH claim, verify if it is supported by the context documents.

RESPONSE FORMAT (for each claim):
Claim X: [restate claim]
Status: ✓ VERIFIED / ✗ UNVERIFIED / ⚠ PARTIALLY VERIFIED
Evidence: [cite specific source and quote if verified]
Notes: [any caveats or context]

RESPONSE:"""
    
    def _get_arguments(self, debate_state: DebateState, agent_name: str) -> str:
        """Extract specific agent's arguments"""
        all_args = []
        for round_data in debate_state.rounds:
            if agent_name in round_data:
                content = round_data[agent_name]['content']
                if not content.startswith("[Error:"):
                    all_args.append(content)
        return "\n\n".join(all_args) if all_args else ""
    
    def _extract_claims(self, pro_args: str, con_args: str) -> List[str]:
        """
        Extract specific factual claims from arguments.
        Looks for statements with citations.
        """
        combined = pro_args + "\n\n" + con_args
        claims = []
        
        # Find sentences with citations [Source: ...]
        pattern = r'([^.!?]+\[Source:[^\]]+\])'
        matches = re.findall(pattern, combined)
        
        for match in matches:
            claim = match.strip()
            if len(claim) > 20:  # Filter out very short claims
                claims.append(claim)
        
        # Limit to top 10 most important claims
        return claims[:10]
    
    def _calculate_pass_rate(self, report: str) -> float:
        """Calculate percentage of verified claims"""
        verified = report.count('✓ VERIFIED')
        unverified = report.count('✗ UNVERIFIED')
        partial = report.count('⚠ PARTIALLY VERIFIED')
        
        total = verified + unverified + partial
        
        if total == 0:
            return 50.0  # Default if no clear verification markers
        
        # Partial counts as 0.5
        score = (verified + (partial * 0.5)) / total
        return score * 100


# Test function
def test_fact_checker():
    """Test Fact-Checker Agent"""
    print("\n" + "=" * 60)
    print("Testing Fact-Checker Agent")
    print("=" * 60)
    
    # Create sample documents
    sample_docs = [
        Document(
            text="Copper prices increased 15% in 2024 due to strong electric vehicle demand.",
            source="IEA_Report_2024.pdf",
            chunk_id="iea_001",
            metadata={},
            score=0.9
        ),
        Document(
            text="China consumes 54% of global copper supply.",
            source="Reuters_2025.txt",
            chunk_id="reuters_001",
            metadata={},
            score=0.85
        ),
    ]
    
    # Create debate state with mock arguments
    debate_state = DebateState(query="Test query")
    debate_state.add_round('pro', """
1. STRONG DEMAND
   "Copper prices increased 15% in 2024" [Source: IEA Report 2024]
   
2. EV GROWTH  
   "EVs require 83kg of copper per car" [Source: Test Data]
""")
    
    debate_state.add_round('con', """
1. CHINA RISK
   "China consumes 54% of global copper" [Source: Reuters 2025]
   
2. VOLATILITY
   "Prices fluctuated 25% in Q4" [Source: Unknown]
""")
    
    # Initialize agent
    agent = FactCheckerAgent()
    
    print(f"\nVerifying claims...\n")
    
    # Generate fact-check
    report = agent.generate(
        query="Test",
        context=sample_docs,
        debate_state=debate_state
    )
    
    print("=" * 60)
    print("FACT-CHECK REPORT:")
    print("=" * 60)
    print(report)
    print("=" * 60)


if __name__ == "__main__":
    test_fact_checker()
