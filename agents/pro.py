"""
Pro Agent - Optimistic Financial Analyst
=========================================
Finds and presents bullish arguments with evidence.

Author: [Your Name]
Date: January 2026
"""

from typing import List, Optional
import ollama

from core.interfaces import BaseAgent, Document, DebateState


class ProAgent(BaseAgent):
    """
    Pro Agent argues FOR the investment/decision.
    Focuses on opportunities, growth drivers, and positive trends.
    """
    
    def __init__(self):
        super().__init__(
            name="pro",
            model="llama3.1:8b",
            role="Optimistic financial analyst"
        )
        
        print(f"✓ ProAgent initialized:")
        print(f"  - Model: {self.model}")
        print(f"  - Role: {self.role}")
    
    def generate(self, query: str, context: List[Document], 
                 debate_state: Optional[DebateState] = None) -> str:
        """
        Generate bullish arguments.
        
        Args:
            query: User's question
            context: Retrieved documents
            debate_state: Current debate state (for rebuttals)
            
        Returns:
            Pro agent's arguments as string
        """
        # Format context documents
        context_text = self.format_context(context)
        
        # Check if this is a rebuttal (Round 2)
        is_rebuttal = debate_state and debate_state.current_round > 0
        
        if is_rebuttal:
            # Get Con agent's arguments for rebuttal
            con_args = self._get_con_arguments(debate_state)
            prompt = self._build_rebuttal_prompt(query, context_text, con_args)
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
                        'temperature': 0.3,  # Low for factual
                        'num_predict': 500,  # Max tokens
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

YOUR TASK: Present 3-5 BULLISH arguments supporting this investment/decision.

RESPONSE:"""
    
    def _build_rebuttal_prompt(self, query: str, context: str, con_args: str) -> str:
        """Build prompt for rebuttal"""
        return f"""{self.system_prompt}

CONTEXT DOCUMENTS:
{context}

USER QUESTION: {query}

CON AGENT'S ARGUMENTS:
{con_args}

YOUR TASK: Respond to the Con Agent's concerns while reinforcing your bullish case.
Address their specific points with evidence from the context.

RESPONSE:"""
    
    def _get_con_arguments(self, debate_state: DebateState) -> str:
        """Extract Con agent's arguments from previous round"""
        if debate_state.rounds and len(debate_state.rounds) > 0:
            last_round = debate_state.rounds[-1]
            return last_round.get('con', {}).get('content', 'No arguments yet')
        return 'No arguments yet'


# Test function
def test_pro_agent():
    """Test Pro Agent"""
    print("\n" + "=" * 60)
    print("Testing Pro Agent")
    print("=" * 60)
    
    # Create sample documents
    sample_docs = [
        Document(
            text="Copper prices increased 15% in 2024 due to strong EV demand. Electric vehicles require 83kg of copper per car, which is 4x more than traditional vehicles.",
            source="IEA_Report_2024.pdf",
            chunk_id="iea_chunk_1",
            metadata={'date': '2024-05-01'},
            score=0.95
        ),
        Document(
            text="Global copper production declined 2% in 2024 due to aging mines in Chile and Peru, creating a projected supply deficit of 400,000 tonnes by 2026.",
            source="USGS_Copper_Summary_2024.pdf",
            chunk_id="usgs_chunk_3",
            metadata={'date': '2024-01-15'},
            score=0.88
        ),
    ]
    
    # Initialize agent
    agent = ProAgent()
    
    # Test query
    query = "Should I invest in copper given current market conditions?"
    
    print(f"\nQuery: '{query}'")
    print("\nGenerating Pro arguments...\n")
    
    # Generate response
    response = agent.generate(query, sample_docs)
    
    print("=" * 60)
    print("PRO AGENT RESPONSE:")
    print("=" * 60)
    print(response)
    print("=" * 60)


if __name__ == "__main__":
    test_pro_agent()
