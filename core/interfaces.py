"""
Base Interfaces for DEBATEAI
============================
Abstract classes that define the contract for all system components.
This ensures extensibility - new components just inherit from these.

Author: [Your Name]
Date: January 2026
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Document:
    """Standard document format across the system"""
    text: str
    source: str
    chunk_id: str
    metadata: Dict[str, Any]
    score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'text': self.text,
            'source': self.source,
            'chunk_id': self.chunk_id,
            'metadata': self.metadata,
            'score': self.score
        }


@dataclass
class DebateState:
    """Tracks the entire debate conversation"""
    query: str
    context_docs: List[Document]
    rounds: List[Dict[str, Any]]
    current_round: int
    trust_score: float
    verdict: str
    timestamp: datetime
    
    def __init__(self, query: str):
        self.query = query
        self.context_docs = []
        self.rounds = []
        self.current_round = 0
        self.trust_score = 0.0
        self.verdict = ""
        self.timestamp = datetime.now()
    
    def add_round(self, agent_name: str, content: str):
        """Add agent response to current round"""
        if self.current_round >= len(self.rounds):
            self.rounds.append({})
        
        self.rounds[self.current_round][agent_name] = {
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
    
    def next_round(self):
        """Move to next debate round"""
        self.current_round += 1


class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval methods.
    
    Any new retriever (FAISS, BM25, Elasticsearch, etc.) 
    must inherit from this and implement retrieve().
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User's question
            top_k: Number of documents to return
            
        Returns:
            List of Document objects with scores
        """
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Document]):
        """Add new documents to the index"""
        pass
    
    def get_name(self) -> str:
        """Return retriever name"""
        return self.name


class BaseAgent(ABC):
    """
    Abstract base class for all debate agents.
    
    Any new agent (Pro, Con, Fact-Checker, Custom) 
    must inherit from this and implement generate().
    """
    
    def __init__(self, name: str, model: str, role: str):
        self.name = name
        self.model = model
        self.role = role
        self.system_prompt = self._load_system_prompt()
    
    def _load_system_prompt(self) -> str:
        """
        Load system prompt from config file.
        Allows changing behavior without code changes.
        """
        try:
            with open(f'config/prompts/{self.name}.txt', 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            # Return default prompt if file doesn't exist
            return f"You are a {self.role} in a structured debate."
    
    @abstractmethod
    def generate(self, query: str, context: List[Document], 
                 debate_state: Optional[DebateState] = None) -> str:
        """
        Generate agent's response.
        
        Args:
            query: User's question
            context: Retrieved documents
            debate_state: Current debate state (for rebuttals)
            
        Returns:
            Agent's response as string
        """
        pass
    
    def format_context(self, context: List[Document]) -> str:
        """Format documents into context string"""
        formatted = []
        for i, doc in enumerate(context, 1):
            formatted.append(
                f"[Document {i} - Source: {doc.source}]\n{doc.text}\n"
            )
        return "\n".join(formatted)
    
    def get_info(self) -> Dict[str, str]:
        """Return agent information"""
        return {
            'name': self.name,
            'model': self.model,
            'role': self.role
        }


class BaseScorer(ABC):
    """
    Abstract base class for all scoring metrics.
    
    Any new scorer (TrustScore, SentimentScore, etc.) 
    must inherit from this and implement calculate().
    """
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
    
    @abstractmethod
    def calculate(self, debate_state: DebateState) -> float:
        """
        Calculate score based on debate state.
        
        Args:
            debate_state: Current debate state
            
        Returns:
            Score between 0.0 and 1.0
        """
        pass
    
    def get_weighted_score(self, debate_state: DebateState) -> float:
        """Calculate and apply weight"""
        return self.calculate(debate_state) * self.weight


class BaseOrchestrator(ABC):
    """
    Abstract base class for debate orchestration.
    
    Manages the flow of debate between agents.
    """
    
    def __init__(self):
        self.agents = {}
        self.retrievers = {}
        self.scorers = {}
    
    @abstractmethod
    def run_debate(self, query: str) -> DebateState:
        """
        Run complete debate and return final state.
        
        Args:
            query: User's question
            
        Returns:
            Final DebateState with verdict and trust score
        """
        pass
    
    def add_agent(self, agent: BaseAgent):
        """Plugin new agent at runtime"""
        self.agents[agent.name] = agent
    
    def add_retriever(self, retriever: BaseRetriever):
        """Plugin new retriever at runtime"""
        self.retrievers[retriever.name] = retriever
    
    def add_scorer(self, scorer: BaseScorer):
        """Plugin new scorer at runtime"""
        self.scorers[scorer.name] = scorer
    
    def remove_agent(self, agent_name: str):
        """Remove agent from debate"""
        if agent_name in self.agents:
            del self.agents[agent_name]
    
    def list_components(self) -> Dict[str, List[str]]:
        """List all active components"""
        return {
            'agents': list(self.agents.keys()),
            'retrievers': list(self.retrievers.keys()),
            'scorers': list(self.scorers.keys())
        }


# Utility function for consistent error handling
class DebateAIException(Exception):
    """Base exception for all DEBATEAI errors"""
    pass


class RetrievalException(DebateAIException):
    """Raised when retrieval fails"""
    pass


class AgentException(DebateAIException):
    """Raised when agent generation fails"""
    pass


class ConfigurationException(DebateAIException):
    """Raised when configuration is invalid"""
    pass


# Version info
__version__ = "1.0.0"
__author__ = "[Your Name]"
__date__ = "January 2026"
