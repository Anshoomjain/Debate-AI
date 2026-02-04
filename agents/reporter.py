"""
Reporter Agent - Formats Final Debate Report
=============================================
Creates polished, professional output from debate results.

Author: [Your Name]
Date: January 2026
"""

from typing import List, Optional
import ollama

from core.interfaces import BaseAgent, Document, DebateState


class ReporterAgent(BaseAgent):
    """
    Reporter Agent formats the final debate output.
    Creates professional, structured reports.
    """
    
    def __init__(self):
        super().__init__(
            name="reporter",
            model="phi3:mini",  # Fast for formatting tasks
            role="Report formatter and presenter"
        )
        
        print(f"✓ ReporterAgent initialized:")
        print(f"  - Model: {self.model}")
        print(f"  - Role: {self.role}")
    
    def generate(self, query: str, context: List[Document], 
                 debate_state: Optional[DebateState] = None) -> str:
        """
        Format final debate report.
        
        Args:
            query: User's question
            context: Retrieved documents
            debate_state: Complete debate state
            
        Returns:
            Formatted final report
        """
        if not debate_state:
            return "[Error: No debate state to format]"
        
        # Get all components
        pro_args = self._get_arguments(debate_state, 'pro')
        con_args = self._get_arguments(debate_state, 'con')
        fact_check = self._get_arguments(debate_state, 'fact_checker')
        verdict = self._get_arguments(debate_state, 'judge')
        
        # Build formatting prompt
        prompt = self._build_prompt(
            query, pro_args, con_args, fact_check, verdict,
            debate_state.trust_score, debate_state.verdict
        )
        
        # Generate formatted report
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        'temperature': 0.3,
                        'num_predict': 1000,  # Longer for full report
                    }
                )
                
                return response['response'].strip()
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠ Attempt {attempt + 1} failed, retrying...")
                    import time
                    time.sleep(2)
                else:
                    return self._create_fallback_report(debate_state)
    
    def _build_prompt(self, query: str, pro_args: str, con_args: str,
                     fact_check: str, verdict: str, trust_score: float,
                     verdict_type: str) -> str:
        """Build report formatting prompt"""
        return f"""You are a professional report writer. Create a clear, executive summary of this debate.

QUERY: {query}

PRO ARGUMENTS:
{pro_args[:500]}...

CON ARGUMENTS:
{con_args[:500]}...

FACT-CHECK RESULTS:
{fact_check[:300] if fact_check else 'Not available'}...

JUDGE'S VERDICT: {verdict_type}
TRUST SCORE: {trust_score:.1f}%

YOUR TASK: Create a 1-page executive summary with these sections:

1. VERDICT SUMMARY (2-3 sentences)
2. KEY FINDINGS (3-4 bullet points)
3. RECOMMENDATION (specific action items)
4. RISK FACTORS (2-3 key risks)
5. CONFIDENCE ASSESSMENT (explain trust score)

Keep it concise, professional, and actionable.

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
    
    def _create_fallback_report(self, debate_state: DebateState) -> str:
        """Create basic report if LLM fails"""
        report = []
        report.append("=" * 70)
        report.append("EXECUTIVE SUMMARY")
        report.append("=" * 70)
        report.append(f"\nQuery: {debate_state.query}")
        report.append(f"Verdict: {debate_state.verdict}")
        report.append(f"Trust Score: {debate_state.trust_score:.1f}%")
        report.append("\nNote: Full report generation unavailable.")
        report.append("Please review detailed debate transcript.")
        report.append("=" * 70)
        
        return "\n".join(report)


# Test function
def test_reporter():
    """Test Reporter Agent"""
    print("\n" + "=" * 60)
    print("Testing Reporter Agent")
    print("=" * 60)
    
    # Create mock debate state
    debate_state = DebateState(query="Should I invest in copper?")
    debate_state.verdict = "FAVORABLE"
    debate_state.trust_score = 75.5
    
    debate_state.add_round('pro', "Strong EV demand driving prices up.")
    debate_state.add_round('con', "China risk and price volatility concerning.")
    debate_state.add_round('fact_checker', "✓ Most claims verified")
    debate_state.add_round('judge', "Moderately favorable with caveats.")
    
    # Initialize agent
    agent = ReporterAgent()
    
    print(f"\nGenerating executive summary...\n")
    
    # Generate report
    report = agent.generate(
        query="Should I invest in copper?",
        context=[],
        debate_state=debate_state
    )
    
    print("=" * 60)
    print("EXECUTIVE SUMMARY:")
    print("=" * 60)
    print(report)
    print("=" * 60)


if __name__ == "__main__":
    test_reporter()
