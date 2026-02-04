"""
DEBATEAI - Command Line Interface
==================================
Interactive CLI for running debates.

Author: [Your Name]
Date: January 2026

"""

import sys
import os
from pathlib import Path
from datetime import datetime

from core.orchestrator_enhanced import EnhancedDebateOrchestrator
# from core.orchestrator import DebateOrchestrator
from core.document_processor import DocumentProcessor
from core.interfaces import Document


class CLI:
    """Interactive command-line interface for DEBATEAI"""
    
    def __init__(self):
        self.orchestrator = None
        self.documents_indexed = False
        
        # Ensure outputs directory exists
        Path("outputs").mkdir(exist_ok=True)
    
    def print_header(self):
        """Print application header"""
        print("\n" + "=" * 70)
        print(" " * 20 + "🤖 DEBATEAI 🤖")
        print(" " * 10 + "Multi-Agent RAG Debate System")
        print("=" * 70)
    
    def print_menu(self):
        """Print main menu"""
        print("\n📋 MAIN MENU:")
        print("-" * 70)
        print("  1. 📂 Index Documents (Load data for debates)")
        print("  2. 🎭 Run New Debate")
        print("  3. 📊 View Last Report")
        print("  4. ⚙️  System Status")
        print("  5. 💡 Example Queries")
        print("  6. ❌ Exit")
        print("-" * 70)
    
    def index_documents_menu(self):
        """Index documents from data folder"""
        print("\n" + "=" * 70)
        print("📂 DOCUMENT INDEXING")
        print("=" * 70)
        
        # Check if data folder exists and has files
        data_folder = Path("data/raw")
        if not data_folder.exists():
            print("\n⚠ Error: data/raw folder not found!")
            print("Please create it and add your documents.")
            input("\nPress Enter to continue...")
            return
        
        # Count files
        pdf_files = list(data_folder.rglob("*.pdf"))
        csv_files = list(data_folder.rglob("*.csv"))
        txt_files = list(data_folder.rglob("*.txt"))
        total_files = len(pdf_files) + len(csv_files) + len(txt_files)
        
        if total_files == 0:
            print("\n⚠ No documents found in data/raw/")
            print("Supported formats: PDF, CSV, TXT")
            print("\nPlease add documents to data/raw/ folder:")
            print("  - data/raw/pdfs/")
            print("  - data/raw/csvs/")
            print("  - data/raw/news/")
            input("\nPress Enter to continue...")
            return
        
        print(f"\n📊 Found {total_files} documents:")
        print(f"  - PDFs: {len(pdf_files)}")
        print(f"  - CSVs: {len(csv_files)}")
        print(f"  - TXTs: {len(txt_files)}")
        
        confirm = input("\n⚠ Index these documents? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Indexing cancelled.")
            return
        
        try:
            # Process documents
            print("\n[1/3] Processing documents...")
            processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
            documents = processor.load_documents("data/raw")
            
            if not documents:
                print("⚠ No documents could be processed!")
                input("\nPress Enter to continue...")
                return
            
            # Save processed chunks
            processor.save_chunks(documents, "data/processed/chunks.json")
            
            # Initialize orchestrator if not already done
            if self.orchestrator is None:
                print("\n[2/3] Initializing debate system...")
                self.orchestrator = EnhancedDebateOrchestrator()
            
            # Index documents
            print("\n[3/3] Building search indices...")
            self.orchestrator.index_documents(documents)
            
            self.documents_indexed = True
            
            print("\n" + "=" * 70)
            print("✅ SUCCESS! Documents indexed and ready for debates.")
            print("=" * 70)
            
        except Exception as e:
            print(f"\n❌ Error during indexing: {e}")
            print("Please check your documents and try again.")
        
        input("\nPress Enter to continue...")
    
    def run_debate_menu(self):
        """Run a new debate"""
        if not self.documents_indexed:
            print("\n⚠ Please index documents first (Option 1)")
            input("\nPress Enter to continue...")
            return
        
        print("\n" + "=" * 70)
        print("🎭 NEW DEBATE")
        print("=" * 70)
        
        # Get query from user
        print("\nEnter your question (or 'back' to return):")
        print("Example: Should I invest in copper given current EV trends?")
        print("-" * 70)
        query = input("➤ ").strip()
        
        if query.lower() == 'back' or not query:
            return
        
        try:
            # Run debate
            print("\n🚀 Starting debate...")
            print("This may take 1-2 minutes...\n")
            
            debate_state = self.orchestrator.run_debate(query, top_k=5)
            
            # Display results
            print("\n" + "=" * 70)
            print("📊 DEBATE RESULTS")
            print("=" * 70)
            print(f"\n✅ Verdict: {debate_state.verdict}")
            print(f"🎯 Trust Score: {debate_state.trust_score:.1f}%")
            
            # Interpret trust score
            if debate_state.trust_score >= 70:
                confidence = "HIGH"
                emoji = "🟢"
            elif debate_state.trust_score >= 50:
                confidence = "MODERATE"
                emoji = "🟡"
            else:
                confidence = "LOW"
                emoji = "🔴"
            
            print(f"{emoji} Confidence Level: {confidence}")
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"outputs/debate_{timestamp}.txt"
            self.orchestrator.save_debate(debate_state, output_path)
            
            print(f"\n💾 Full report saved to: {output_path}")
            
            # Ask if user wants to see full report
            view = input("\n📄 View full report now? (yes/no): ").strip().lower()
            if view in ['yes', 'y']:
                self.view_report(output_path)
            
        except Exception as e:
            print(f"\n❌ Error during debate: {e}")
        
        input("\nPress Enter to continue...")
    
    def view_last_report(self):
        """View the most recent debate report"""
        print("\n" + "=" * 70)
        print("📊 LAST DEBATE REPORT")
        print("=" * 70)
        
        # Find most recent report
        outputs_dir = Path("outputs")
        reports = list(outputs_dir.glob("debate_*.txt"))
        
        if not reports:
            print("\n⚠ No reports found. Run a debate first!")
            input("\nPress Enter to continue...")
            return
        
        # Get most recent
        latest_report = max(reports, key=lambda p: p.stat().st_mtime)
        self.view_report(str(latest_report))
    
    def view_report(self, filepath: str):
        """Display a report file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print("\n" + content)
            
        except Exception as e:
            print(f"\n❌ Error reading report: {e}")
        
        input("\nPress Enter to continue...")
    
    def system_status(self):
        """Show system status"""
        print("\n" + "=" * 70)
        print("⚙️  SYSTEM STATUS")
        print("=" * 70)
        
        # Check documents
        print("\n📂 Document Status:")
        if self.documents_indexed:
            print("  ✅ Documents indexed and ready")
            
            # Count chunks
            chunks_file = Path("data/processed/chunks.json")
            if chunks_file.exists():
                import json
                with open(chunks_file, 'r') as f:
                    chunks = json.load(f)
                print(f"  📊 Total chunks: {len(chunks)}")
        else:
            print("  ⚠️  No documents indexed (use Option 1)")
        
        # Check agents
        print("\n🤖 Agents Status:")
        if self.orchestrator:
            for agent_name in self.orchestrator.agents.keys():
                print(f"  ✅ {agent_name.capitalize()} Agent: Ready")
        else:
            print("  ⚠️  Agents not initialized")
        
        # Check Ollama
        print("\n🔧 Ollama Status:")
        try:
            import ollama
            models = ollama.list()
            print(f"  ✅ Connected")
            print(f"  📦 Available models: {len(models.get('models', []))}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        # Check GPU
        print("\n🎮 GPU Status:")
        try:
            import torch
            if torch.cuda.is_available():
                print(f"  ✅ CUDA Available")
                print(f"  💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            else:
                print(f"  ⚠️  CPU Mode (slower)")
        except Exception:
            print(f"  ⚠️  Could not detect GPU")
        
        input("\nPress Enter to continue...")
    
    def show_examples(self):
        """Show example queries"""
        print("\n" + "=" * 70)
        print("💡 EXAMPLE QUERIES")
        print("=" * 70)
        
        examples = [
            ("Finance", [
                "Should I invest in copper given current EV market trends?",
                "Is now a good time to buy Tesla stock?",
                "What are the risks of investing in cryptocurrency?",
            ]),
            ("Business", [
                "Should our company enter the Indian EV market?",
                "Is this the right time to launch our AI product?",
                "Should we invest in renewable energy infrastructure?",
            ]),
            ("Academic", [
                "Should I pursue a PhD in artificial intelligence?",
                "What are the main open problems in quantum computing?",
                "Is transformer architecture still the best for NLP?",
            ]),
        ]
        
        for category, queries in examples:
            print(f"\n📁 {category}:")
            for i, query in enumerate(queries, 1):
                print(f"  {i}. {query}")
        
        print("\n💡 Tips:")
        print("  - Be specific about your situation")
        print("  - Include relevant context (e.g., 'given current trends')")
        print("  - Ask about trade-offs and risks")
        print("  - Questions work better than statements")
        
        input("\nPress Enter to continue...")
    
    def run(self):
        """Main application loop"""
        self.print_header()
        
        while True:
            self.print_menu()
            
            choice = input("\n➤ Enter your choice (1-6): ").strip()
            
            if choice == '1':
                self.index_documents_menu()
            elif choice == '2':
                self.run_debate_menu()
            elif choice == '3':
                self.view_last_report()
            elif choice == '4':
                self.system_status()
            elif choice == '5':
                self.show_examples()
            elif choice == '6':
                print("\n👋 Thanks for using DEBATEAI!")
                print("=" * 70)
                sys.exit(0)
            else:
                print("\n⚠️  Invalid choice. Please enter 1-6.")
                input("\nPress Enter to continue...")


def main():
    """Entry point"""
    cli = CLI()
    
    try:
        cli.run()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
