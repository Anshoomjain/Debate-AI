"""
Con Agent - Skeptical Risk Analyst
===================================
Identifies risks and bearish factors with evidence.

Author: [Your Name]
Date: January 2026
"""

from typing import List, Optional
import ollama

from core.interfaces import BaseAgent, Document, DebateState


class ConAgent(BaseAgent):
    """
    Con Agent argues AGAINST the investment/decision.
    Focuses on risks, uncertainties, and potential downsides.
    """
    
    def __init__(self):
        super().__init__(
            name="con",
            model="mistral:7b",
            role="Skeptical risk analyst"
        )
        
        print(f"✓ ConAgent initialized:")
        print(f"  - Model: {self.model}")
        print(f"  - Role: {self.role}")
    
    def generate(self, query: str, context: List[Document], 
                 debate_state: Optional[DebateState] = None) -> str:
        """
        Generate bearish arguments.
        
        Args:
            query: User's question
            context: Retrieved documents
            debate_state: Current debate state (for rebuttals)
            
        Returns:
            Con agent's arguments as string
        """
        # Format context documents
        context_text = self.format_context(context)
        
        # Check if this is a rebuttal (Round 2)
        is_rebuttal = debate_state and debate_state.current_round > 0
        
        if is_rebuttal:
            # Get Pro agent's arguments for rebuttal
            pro_args = self._get_pro_arguments(debate_state)
            prompt = self._build_rebuttal_prompt(query, context_text, pro_args)
        else:
            # Initial arguments (Round 1)
            prompt = self._build_initial_prompt(query, context_text)
        
        # Generate response using Ollama
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        'temperature': 0.4,  # Slightly higher for critical thinking
                        'num_predict': 500,
                    }
                )
                
                return response['response'].strip()
                
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
    
    def _build_initial_prompt(self, query: str, context: str) -> str:
        """Build prompt for initial arguments"""
        return f"""{self.system_prompt}

CONTEXT DOCUMENTS:
{context}

USER QUESTION: {query}

YOUR TASK: Present 3-5 BEARISH arguments highlighting risks and concerns.

RESPONSE:"""
    
    def _build_rebuttal_prompt(self, query: str, context: str, pro_args: str) -> str:
        """Build prompt for rebuttal"""
        return f"""{self.system_prompt}

CONTEXT DOCUMENTS:
{context}

USER QUESTION: {query}

PRO AGENT'S ARGUMENTS:
{pro_args}

YOUR TASK: Counter the Pro Agent's optimism with evidence-based concerns.
Identify overlooked risks and challenge overly bullish assumptions.

RESPONSE:"""
    
    def _get_pro_arguments(self, debate_state: DebateState) -> str:
        """Extract Pro agent's arguments from previous round"""
        if debate_state.rounds and len(debate_state.rounds) > 0:
            last_round = debate_state.rounds[-1]
            return last_round.get('pro', {}).get('content', 'No arguments yet')
        return 'No arguments yet'


# Test function
def test_con_agent():
    """Test Con Agent"""
    print("\n" + "=" * 60)
    print("Testing Con Agent")
    print("=" * 60)
    
    # Create sample documents
    sample_docs = [
        Document(
            text="Copper prices fluctuated between $3.80-$4.70 per pound in Q4 2024, a 23% range indicating high market volatility and uncertainty.",
            source="Yahoo_Finance_Copper.csv",
            chunk_id="yfinance_chunk_5",
            metadata={'date': '2024-12-20'},
            score=0.92
        ),
        Document(
            text="China consumes 54% of global copper supply. Recent government stockpiling and import restrictions create significant geopolitical risk.",
            source="Reuters_Commodities_Jan2025.txt",
            chunk_id="reuters_chunk_2",
            metadata={'date': '2025-01-18'},
            score=0.87
        ),
    ]
    
    # Initialize agent
    agent = ConAgent()
    
    # Test query
    query = "Should I invest in copper given current market conditions?"
    
    print(f"\nQuery: '{query}'")
    print("\nGenerating Con arguments...\n")
    
    # Generate response
    response = agent.generate(query, sample_docs)
    
    print("=" * 60)
    print("CON AGENT RESPONSE:")
    print("=" * 60)
    print(response)
    print("=" * 60)


if __name__ == "__main__":
    test_con_agent()
