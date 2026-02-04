"""
Judge Agent - Balanced Synthesizer
===================================
Creates final verdict and calculates trust score.

Author: [Your Name]
Date: January 2026
"""

from typing import List, Optional
import ollama
import re

from core.interfaces import BaseAgent, Document, DebateState


class JudgeAgent(BaseAgent):
    """
    Judge Agent synthesizes Pro and Con arguments.
    Provides balanced verdict and trust score.
    """
    
    def __init__(self):
        super().__init__(
            name="judge",
            model="llama3.1:8b",
            role="Balanced synthesizer and judge"
        )
        
        print(f"✓ JudgeAgent initialized:")
        print(f"  - Model: {self.model}")
        print(f"  - Role: {self.role}")
    
    def generate(self, query: str, context: List[Document], 
                 debate_state: Optional[DebateState] = None) -> str:
        """
        Generate final verdict synthesizing both sides.
        
        Args:
            query: User's question
            context: Retrieved documents
            debate_state: Debate state with Pro/Con arguments
            
        Returns:
            Judge's verdict and recommendation
        """
        if not debate_state or not debate_state.rounds:
            return "[Error: No debate to judge]"
        
        # Get Pro and Con arguments
        pro_args = self._get_arguments(debate_state, 'pro')
        con_args = self._get_arguments(debate_state, 'con')
        
        # Format context
        context_text = self.format_context(context)
        
        # Build prompt
        prompt = self._build_prompt(query, context_text, pro_args, con_args)
        
        # Generate response
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        'temperature': 0.2,  # Very low for balanced judgment
                        'num_predict': 800,  # Longer response
                    }
                )
                
                verdict_text = response['response'].strip()
                
                # Calculate trust score
                trust_score = self._calculate_trust_score(
                    pro_args, con_args, verdict_text, context
                )
                
                # Update debate state
                if debate_state:
                    debate_state.trust_score = trust_score
                    debate_state.verdict = self._extract_verdict(verdict_text)
                
                # Add trust score to output
                final_output = f"{verdict_text}\n\n{'='*60}\nTRUST SCORE: {trust_score:.1f}%\n{'='*60}"
                
                return final_output
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠ Attempt {attempt + 1} failed, retrying...")
                    # Wait a bit before retry
                    import time
                    time.sleep(2)
                    # Clear GPU memory
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except:
                        pass
                else:
                    return f"[Error: Failed after {max_retries} attempts: {e}]"
    
    def _build_prompt(self, query: str, context: str, 
                     pro_args: str, con_args: str) -> str:
        """Build judgment prompt"""
        return f"""{self.system_prompt}

CONTEXT DOCUMENTS:
{context}

USER QUESTION: {query}

PRO AGENT'S ARGUMENTS (Bullish):
{pro_args}

CON AGENT'S ARGUMENTS (Bearish):
{con_args}

YOUR TASK: Provide a balanced final verdict considering BOTH perspectives.

RESPONSE:"""
    
    def _get_arguments(self, debate_state: DebateState, agent_name: str) -> str:
        """Extract specific agent's arguments from all rounds"""
        all_args = []
        
        for round_data in debate_state.rounds:
            if agent_name in round_data:
                all_args.append(round_data[agent_name]['content'])
        
        return "\n\n".join(all_args) if all_args else "No arguments"
    
    def _extract_verdict(self, text: str) -> str:
        """Extract verdict from judge's response"""
        text_upper = text.upper()
        
        if 'VERDICT:' in text_upper:
            # Try to extract from "VERDICT: XXX" pattern
            match = re.search(r'VERDICT:\s*(FAVORABLE|UNFAVORABLE|UNCERTAIN)', text_upper)
            if match:
                return match.group(1)
        
        # Fallback: look for keywords
        if 'FAVORABLE' in text_upper and 'UNFAVORABLE' not in text_upper:
            return 'FAVORABLE'
        elif 'UNFAVORABLE' in text_upper:
            return 'UNFAVORABLE'
        else:
            return 'UNCERTAIN'
    
    def _calculate_trust_score(self, pro_args: str, con_args: str, 
                               verdict: str, context: List[Document]) -> float:
        """
        Calculate trust score based on multiple factors.
        
        Components:
        - Citation rate (50%): How many claims have sources
        - Argument balance (30%): Are both sides represented
        - Data recency (20%): How fresh is the data
        """
        # Component 1: Citation rate
        citation_rate = self._calculate_citation_rate(pro_args, con_args)
        
        # Component 2: Argument balance
        balance_score = self._calculate_balance(pro_args, con_args)
        
        # Component 3: Data recency
        recency_score = self._calculate_recency(context)
        
        # Weighted sum
        trust_score = (
            citation_rate * 0.5 +
            balance_score * 0.3 +
            recency_score * 0.2
        )
        
        return min(100.0, trust_score * 100)  # Convert to percentage
    
    def _calculate_citation_rate(self, pro_args: str, con_args: str) -> float:
        """Calculate percentage of claims with citations"""
        combined = pro_args + " " + con_args
        
        # Count citations [Source: ...]
        citations = len(re.findall(r'\[Source:', combined, re.IGNORECASE))
        
        # Count sentences (rough estimate of claims)
        sentences = len(re.findall(r'[.!?]+', combined))
        
        if sentences == 0:
            return 0.0
        
        # Ratio of citations to sentences
        rate = min(1.0, citations / sentences)
        return rate
    
    def _calculate_balance(self, pro_args: str, con_args: str) -> float:
        """Calculate how balanced the arguments are"""
        pro_words = len(pro_args.split())
        con_words = len(con_args.split())
        
        if pro_words + con_words == 0:
            return 0.0
        
        # Ideal is 50/50 split
        pro_ratio = pro_words / (pro_words + con_words)
        
        # Score is higher when closer to 0.5
        balance = 1.0 - abs(pro_ratio - 0.5) * 2
        return balance
    
    def _calculate_recency(self, context: List[Document]) -> float:
        """Calculate data recency score"""
        if not context:
            return 0.5  # Neutral if no context
        
        # For now, return moderate score
        # In production, would parse dates from metadata
        return 0.8  # Assume relatively recent data


# Test function
def test_judge_agent():
    """Test Judge Agent"""
    print("\n" + "=" * 60)
    print("Testing Judge Agent")
    print("=" * 60)
    
    # Create sample documents
    sample_docs = [
        Document(
            text="Copper demand growing due to EV adoption.",
            source="test.pdf",
            chunk_id="test_1",
            metadata={},
            score=0.9
        ),
    ]
    
    # Create debate state with mock arguments
    debate_state = DebateState(query="Should I invest in copper?")
    debate_state.add_round('pro', """
1. STRONG EV DEMAND
   "EV adoption increased 23% in 2024" [Source: IEA Report 2024]
   
2. SUPPLY DEFICIT
   "400,000 tonne deficit by 2026" [Source: USGS 2024]
""")
    
    debate_state.add_round('con', """
1. PRICE VOLATILITY
   "Copper fluctuated $3.80-$4.70 in Q4" [Source: Yahoo Finance]
   
2. CHINA RISK
   "China controls 54% of demand" [Source: Reuters 2025]
""")
    
    # Initialize agent
    agent = JudgeAgent()
    
    print(f"\nSynthesizing debate...\n")
    
    # Generate verdict
    verdict = agent.generate(
        query="Should I invest in copper?",
        context=sample_docs,
        debate_state=debate_state
    )
    
    print("=" * 60)
    print("JUDGE'S VERDICT:")
    print("=" * 60)
    print(verdict)
    print("\n" + "=" * 60)
    print(f"Extracted Verdict: {debate_state.verdict}")
    print(f"Trust Score: {debate_state.trust_score:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    test_judge_agent()
