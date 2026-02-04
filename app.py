"""
DEBATEAI - Streamlit Web Application
=====================================
Professional web interface for multi-agent debates.

Author: [Your Name]
Date: January 2026
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
from datetime import datetime
import pandas as pd

from core.orchestrator_enhanced import EnhancedDebateOrchestrator
from core.document_processor import DocumentProcessor
from core.interfaces import DebateState


# Page configuration
st.set_page_config(
    page_title="DEBATEAI - Multi-Agent Debate System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .agent-box {
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        background: #f8f9fa;
        border-radius: 5px;
    }
    .pro-box {
        border-left-color: #28a745;
    }
    .con-box {
        border-left-color: #dc3545;
    }
    .fact-box {
        border-left-color: #ffc107;
    }
    .judge-box {
        border-left-color: #6610f2;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = None
    if 'documents_indexed' not in st.session_state:
        st.session_state.documents_indexed = False
    if 'debate_history' not in st.session_state:
        st.session_state.debate_history = []
    if 'current_debate' not in st.session_state:
        st.session_state.current_debate = None


def create_trust_score_gauge(score: float):
    """Create a beautiful gauge chart for trust score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Trust Score", 'font': {'size': 24}},
        delta={'reference': 70, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffcdd2'},
                {'range': [50, 70], 'color': '#fff9c4'},
                {'range': [70, 100], 'color': '#c8e6c9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_argument_comparison(debate_state: DebateState):
    """Create bar chart comparing argument lengths"""
    if not debate_state or len(debate_state.rounds) < 1:
        return None
    
    # Extract argument lengths
    pro_length = 0
    con_length = 0
    
    for round_data in debate_state.rounds:
        if 'pro' in round_data:
            pro_length += len(round_data['pro']['content'].split())
        if 'con' in round_data:
            con_length += len(round_data['con']['content'].split())
    
    # Create chart
    data = pd.DataFrame({
        'Agent': ['Pro Agent', 'Con Agent'],
        'Word Count': [pro_length, con_length],
        'Color': ['#28a745', '#dc3545']
    })
    
    fig = px.bar(
        data,
        x='Agent',
        y='Word Count',
        color='Agent',
        color_discrete_map={'Pro Agent': '#28a745', 'Con Agent': '#dc3545'},
        title='Argument Length Comparison'
    )
    
    fig.update_layout(
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_trust_breakdown(debate_state: DebateState):
    """Create breakdown of trust score components"""
    # Simplified - you can enhance with actual component scores
    components = {
        'Citation Rate': 40,
        'Fact-Check Pass': 30,
        'Argument Coherence': 15,
        'Data Recency': 15
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=list(components.keys()),
        values=list(components.values()),
        hole=.3,
        marker_colors=['#667eea', '#764ba2', '#f093fb', '#4facfe']
    )])
    
    fig.update_layout(
        title="Trust Score Components",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 DEBATEAI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Agent RAG Debate System for Evidence-Based Decisions</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=DEBATEAI", use_container_width=True)
        
        st.markdown("## 📋 Navigation")
        
        page = st.radio(
            "Choose a page:",
            ["🏠 Home", "📂 Document Manager", "🎭 Run Debate", "📊 Results & Analytics", "📜 History", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Status indicators
        st.markdown("## ⚙️ System Status")
        
        if st.session_state.documents_indexed:
            st.success("✅ Documents Indexed")
        else:
            st.warning("⚠️ No Documents Indexed")
        
        if st.session_state.orchestrator:
            st.success("✅ System Ready")
        else:
            st.info("ℹ️ System Not Initialized")
        
        st.markdown("---")
        
        # Quick stats
        if st.session_state.debate_history:
            st.metric("Total Debates", len(st.session_state.debate_history))
            
            # Average trust score
            avg_trust = sum(d.trust_score for d in st.session_state.debate_history) / len(st.session_state.debate_history)
            st.metric("Avg Trust Score", f"{avg_trust:.1f}%")
    
    # Main content based on page selection
    if page == "🏠 Home":
        show_home_page()
    elif page == "📂 Document Manager":
        show_document_manager()
    elif page == "🎭 Run Debate":
        show_debate_page()
    elif page == "📊 Results & Analytics":
        show_results_page()
    elif page == "📜 History":
        show_history_page()
    elif page == "ℹ️ About":
        show_about_page()


def show_home_page():
    """Home page with overview"""
    st.markdown("## 🎯 Welcome to DEBATEAI")
    
    st.markdown("""
    ### What is DEBATEAI?
    
    DEBATEAI is an advanced multi-agent debate system that helps you make evidence-based decisions by:
    - 🔍 Analyzing your documents using RAG (Retrieval-Augmented Generation)
    - 🎭 Simulating debates between AI agents with different perspectives
    - ✅ Fact-checking all claims against source documents
    - 📊 Providing trust scores and recommendations
    
    ### How it Works
    """)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("#### 1️⃣ Upload")
        st.info("Add your PDF, CSV, or TXT documents")
    
    with col2:
        st.markdown("#### 2️⃣ Index")
        st.info("System processes and indexes your data")
    
    with col3:
        st.markdown("#### 3️⃣ Ask")
        st.info("Enter your question or decision")
    
    with col4:
        st.markdown("#### 4️⃣ Debate")
        st.info("5 AI agents analyze from all angles")
    
    with col5:
        st.markdown("#### 5️⃣ Decide")
        st.info("Get verdict with trust score")
    
    st.markdown("---")
    
    # Features
    st.markdown("### ✨ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🤖 5 Specialized AI Agents:**
        - **Pro Agent**: Finds supporting evidence
        - **Con Agent**: Identifies risks and concerns
        - **Fact-Checker**: Validates all claims
        - **Judge**: Synthesizes balanced verdict
        - **Reporter**: Creates executive summary
        """)
    
    with col2:
        st.markdown("""
        **📊 Advanced Analysis:**
        - Hybrid search (semantic + keyword)
        - 100% citation rate
        - Trust score algorithm
        - Multi-round debates
        - Source verification
        """)
    
    st.markdown("---")
    
    # Quick start
    st.markdown("### 🚀 Quick Start")
    
    st.info("👈 Use the sidebar to navigate. Start with **Document Manager** to upload your data!")
    
    if st.button("🎬 Start with Document Manager", use_container_width=True):
        st.rerun()


def show_document_manager():
    """Document upload and indexing page"""
    st.markdown("## 📂 Document Manager")
    
    st.markdown("Upload your documents (PDF, CSV, TXT) to build the knowledge base for debates.")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['pdf', 'csv', 'txt'],
        accept_multiple_files=True,
        help="Upload PDF reports, CSV data files, or TXT documents"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} files selected")
        
        # Show file list
        with st.expander("📄 View Selected Files"):
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size / 1024:.1f} KB)")
        
        # Process button
        if st.button("🔄 Process & Index Documents", use_container_width=True):
            process_documents(uploaded_files)
    
    # Or use existing data
    st.markdown("---")
    st.markdown("### Or Use Existing Data")
    
    data_folder = Path("data/processed")
    if data_folder.exists() and (data_folder / "chunks.json").exists():
        st.info(f"✅ Found processed data in `{data_folder}`")
        
        if st.button("📥 Load Processed Data", use_container_width=True):
            load_existing_data()
    else:
        st.warning("⚠️ No processed data found. Please upload documents or run `process_data.py`")


def process_documents(uploaded_files):
    """Process uploaded documents"""
    with st.spinner("Processing documents..."):
        try:
            # Save uploaded files temporarily
            temp_dir = Path("data/temp")
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            for file in uploaded_files:
                file_path = temp_dir / file.name
                with open(file_path, 'wb') as f:
                    f.write(file.getbuffer())
            
            # Process documents
            processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
            documents = processor.load_documents(str(temp_dir))
            
            # Save processed chunks
            Path("data/processed").mkdir(parents=True, exist_ok=True)
            processor.save_chunks(documents, "data/processed/chunks.json")
            
            # Initialize orchestrator
            st.session_state.orchestrator = EnhancedDebateOrchestrator()
            st.session_state.orchestrator.index_documents(documents)
            st.session_state.documents_indexed = True
            
            st.success(f"✅ Successfully processed {len(documents)} document chunks!")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Error processing documents: {str(e)}")


def load_existing_data():
    """Load pre-processed data"""
    with st.spinner("Loading data..."):
        try:
            processor = DocumentProcessor()
            documents = processor.load_chunks("data/processed/chunks.json")
            
            st.session_state.orchestrator = EnhancedDebateOrchestrator()
            st.session_state.orchestrator.index_documents(documents)
            st.session_state.documents_indexed = True
            
            st.success(f"✅ Loaded {len(documents)} document chunks!")
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")


def show_debate_page():
    """Main debate interface"""
    st.markdown("## 🎭 Run Debate")
    
    if not st.session_state.documents_indexed:
        st.warning("⚠️ Please index documents first in the Document Manager!")
        if st.button("📂 Go to Document Manager"):
            st.rerun()
        return
    
    # Query input
    st.markdown("### 💭 Enter Your Question")
    
    # Example queries
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - Should I invest in copper given current EV market trends?
        - Is Tesla stock a good buy at current valuation?
        - What are the risks of Bitcoin as an inflation hedge?
        - Should our company enter the Indian EV market?
        - Is renewable energy infrastructure a good investment?
        """)
    
    query = st.text_area(
        "Your Question:",
        placeholder="e.g., Should I invest in copper given current EV trends?",
        height=100,
        label_visibility="collapsed"
    )
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Number of documents to retrieve", 3, 10, 5)
        with col2:
            show_live_updates = st.checkbox("Show live round-by-round updates", value=True)
    
    # Run debate button
    if st.button("🚀 Start Debate", use_container_width=True, disabled=not query):
        run_debate(query, top_k, show_live_updates)


def run_debate(query: str, top_k: int, show_live_updates: bool):
    """Run the debate"""
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Start debate
        status_text.text("🔍 Retrieving relevant documents...")
        progress_bar.progress(10)
        
        debate_state = DebateState(query=query)
        
        # Retrieve documents
        context_docs = st.session_state.orchestrator.hybrid_retriever.retrieve(query, top_k=top_k)
        debate_state.context_docs = context_docs
        
        progress_bar.progress(20)
        
        if show_live_updates:
            with st.expander("📚 Retrieved Documents", expanded=False):
                for i, doc in enumerate(context_docs, 1):
                    st.markdown(f"**{i}. {doc.source}** (score: {doc.score:.3f})")
                    st.caption(doc.text[:200] + "...")
        
        # Round 1
        status_text.text("Round 1: Pro & Con initial arguments...")
        progress_bar.progress(30)
        
        pro_args = st.session_state.orchestrator.agents['pro'].generate(query, context_docs, debate_state)
        debate_state.add_round('pro', pro_args)
        
        con_args = st.session_state.orchestrator.agents['con'].generate(query, context_docs, debate_state)
        debate_state.add_round('con', con_args)
        
        progress_bar.progress(45)
        
        # Round 2
        debate_state.next_round()
        status_text.text("Round 2: Rebuttals...")
        
        pro_rebuttal = st.session_state.orchestrator.agents['pro'].generate(query, context_docs, debate_state)
        debate_state.add_round('pro', pro_rebuttal)
        
        con_rebuttal = st.session_state.orchestrator.agents['con'].generate(query, context_docs, debate_state)
        debate_state.add_round('con', con_rebuttal)
        
        progress_bar.progress(60)
        
        # Round 3
        debate_state.next_round()
        status_text.text("Round 3: Fact-checking claims...")
        
        fact_check = st.session_state.orchestrator.agents['fact_checker'].generate(query, context_docs, debate_state)
        debate_state.add_round('fact_checker', fact_check)
        
        progress_bar.progress(75)
        
        # Round 4
        debate_state.next_round()
        status_text.text("Round 4: Judge synthesizing...")
        
        verdict = st.session_state.orchestrator.agents['judge'].generate(query, context_docs, debate_state)
        debate_state.add_round('judge', verdict)
        
        progress_bar.progress(90)
        
        # Round 5
        debate_state.next_round()
        status_text.text("Round 5: Creating executive summary...")
        
        report = st.session_state.orchestrator.agents['reporter'].generate(query, context_docs, debate_state)
        debate_state.add_round('reporter', report)
        
        progress_bar.progress(100)
        status_text.text("✅ Debate complete!")
        
        # Save to history
        st.session_state.current_debate = debate_state
        st.session_state.debate_history.append(debate_state)
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/debate_{timestamp}.txt"
        Path("outputs").mkdir(exist_ok=True)
        st.session_state.orchestrator.save_debate(debate_state, output_path)
        
        st.success("🎉 Debate completed successfully!")
        st.balloons()
        
        # Show results immediately
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error during debate: {str(e)}")
        progress_bar.empty()
        status_text.empty()


def show_results_page():
    """Show debate results and analytics"""
    st.markdown("## 📊 Results & Analytics")
    
    if not st.session_state.current_debate:
        st.info("ℹ️ No debate results yet. Run a debate first!")
        return
    
    debate = st.session_state.current_debate
    
    # Top metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Verdict", debate.verdict)
    
    with col2:
        st.metric("Trust Score", f"{debate.trust_score:.1f}%")
    
    with col3:
        confidence = "HIGH" if debate.trust_score >= 70 else "MODERATE" if debate.trust_score >= 50 else "LOW"
        st.metric("Confidence", confidence)
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_trust_score_gauge(debate.trust_score), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_trust_breakdown(debate), use_container_width=True)
    
    # Argument comparison
    st.plotly_chart(create_argument_comparison(debate), use_container_width=True)
    
    st.markdown("---")
    
    # Detailed results
    st.markdown("### 📝 Detailed Analysis")
    
    tabs = st.tabs(["✅ Pro", "❌ Con", "🔍 Fact-Check", "⚖️ Judge", "📊 Executive Summary"])
    
    with tabs[0]:
        if debate.rounds and 'pro' in debate.rounds[0]:
            st.markdown('<div class="agent-box pro-box">', unsafe_allow_html=True)
            st.markdown("#### Pro Agent Arguments")
            st.markdown(debate.rounds[0]['pro']['content'])
            if len(debate.rounds) > 1 and 'pro' in debate.rounds[1]:
                st.markdown("#### Pro Agent Rebuttal")
                st.markdown(debate.rounds[1]['pro']['content'])
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[1]:
        if debate.rounds and 'con' in debate.rounds[0]:
            st.markdown('<div class="agent-box con-box">', unsafe_allow_html=True)
            st.markdown("#### Con Agent Arguments")
            st.markdown(debate.rounds[0]['con']['content'])
            if len(debate.rounds) > 1 and 'con' in debate.rounds[1]:
                st.markdown("#### Con Agent Rebuttal")
                st.markdown(debate.rounds[1]['con']['content'])
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[2]:
        if len(debate.rounds) > 2 and 'fact_checker' in debate.rounds[2]:
            st.markdown('<div class="agent-box fact-box">', unsafe_allow_html=True)
            st.markdown(debate.rounds[2]['fact_checker']['content'])
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[3]:
        if len(debate.rounds) > 3 and 'judge' in debate.rounds[3]:
            st.markdown('<div class="agent-box judge-box">', unsafe_allow_html=True)
            st.markdown(debate.rounds[3]['judge']['content'])
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[4]:
        if len(debate.rounds) > 4 and 'reporter' in debate.rounds[4]:
            st.markdown(debate.rounds[4]['reporter']['content'])
    
    # Export options
    st.markdown("---")
    st.markdown("### 💾 Export Report")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Download TXT", use_container_width=True):
            report = st.session_state.orchestrator.format_debate_report(debate)
            st.download_button(
                "📥 Click to Download",
                data=report,
                file_name=f"debate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    with col2:
        if st.button("📊 Download JSON", use_container_width=True):
            # Create JSON export
            export_data = {
                'query': debate.query,
                'verdict': debate.verdict,
                'trust_score': debate.trust_score,
                'timestamp': debate.timestamp.isoformat(),
                'rounds': debate.rounds
            }
            st.download_button(
                "📥 Click to Download",
                data=json.dumps(export_data, indent=2),
                file_name=f"debate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        st.info("📑 PDF export coming soon!")


def show_history_page():
    """Show debate history"""
    st.markdown("## 📜 Debate History")
    
    if not st.session_state.debate_history:
        st.info("ℹ️ No debate history yet. Run some debates first!")
        return
    
    st.markdown(f"Total debates: **{len(st.session_state.debate_history)}**")
    
    # Show debates in reverse chronological order
    for i, debate in enumerate(reversed(st.session_state.debate_history)):
        with st.expander(f"🎭 {debate.query} - {debate.verdict} ({debate.trust_score:.1f}%)"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Verdict", debate.verdict)
            with col2:
                st.metric("Trust Score", f"{debate.trust_score:.1f}%")
            with col3:
                st.metric("Date", debate.timestamp.strftime("%Y-%m-%d %H:%M"))
            
            if st.button(f"View Full Report", key=f"view_{i}"):
                st.session_state.current_debate = debate
                st.rerun()


def show_about_page():
    """About page"""
    st.markdown("## ℹ️ About DEBATEAI")
    
    st.markdown("""
    ### 🎯 Mission
    
    DEBATEAI provides evidence-based decision support through multi-agent AI debates, ensuring 
    balanced analysis from multiple perspectives.
    
    ### 🏗️ Architecture
    
    **5 Specialized AI Agents:**
    - **Pro Agent** (Llama 3.1 8B): Optimistic analyst
    - **Con Agent** (Mistral 7B): Risk-focused analyst
    - **Fact-Checker** (Phi-3 Medium): Claim validator
    - **Judge** (Llama 3.1 8B): Synthesizer
    - **Reporter** (Phi-3 Mini): Report formatter
    
    **RAG Pipeline:**
    - FAISS for semantic search
    - BM25 for keyword search
    - Hybrid retrieval with RRF
    - 512-token chunks with 50-token overlap
    
    ### 📊 Trust Score Algorithm
    
    ```
    Trust Score = 0.40×Citation + 0.30×FactCheck + 0.15×Coherence + 0.15×Recency
    ```
    
    - **Citation Rate**: Percentage of claims with sources
    - **Fact-Check Pass**: Percentage of verified claims
    - **Coherence**: Logical consistency of arguments
    - **Recency**: Freshness of source data
    
    ### 🛠️ Technology Stack
    
    - **LLM Server**: Ollama (local deployment)
    - **Framework**: LlamaIndex
    - **Vector DB**: FAISS
    - **Keyword Search**: BM25
    - **Embeddings**: Sentence-Transformers
    - **UI**: Streamlit
    - **Visualization**: Plotly
    
    ### 📝 Version
    
    **Version**: 1.0.0  
    **Release Date**: January 2026  
    **Author**: [Your Name]  
    **License**: MIT
    
    ### 🔗 Links
    
    - [GitHub Repository](#)
    - [Documentation](#)
    - [Academic Paper](#)
    """)


if __name__ == "__main__":
    import time
    main()
