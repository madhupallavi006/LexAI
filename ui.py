# ui.py
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from rag_pipeline import LegalRAGPipeline

class LexAIUI:
    """Simple terminal-based UI for VS Code."""

    def __init__(self, rag: LegalRAGPipeline):
        self.rag             = rag
        self.current_doc_id  = None
        self.current_doc_name = ""
        self.current_lang    = "English"

    def render(self):
        print("\n" + "="*50)
        print("  ⚖  LexAI — Legal Document Assistant")
        print("="*50)
        print(f"  📚 {self.rag.cuad_kb_status()}")
        print("="*50)
        self._main_loop()

    def _main_loop(self):
        while True:
            print("\nWhat do you want to do?")
            print("  1 → Load a contract (paste text)")
            print("  2 → Summary")
            print("  3 → Risk Analysis")
            print("  4 → Key Terms")
            print("  5 → Ask a question")
            print("  6 → Exit")

            choice = input("\nEnter number: ").strip()

            if choice == "1":
                self._load_document()
            elif choice == "2":
                self._summary()
            elif choice == "3":
                self._risks()
            elif choice == "4":
                self._terms()
            elif choice == "5":
                self._ask()
            elif choice == "6":
                print("\n👋 Goodbye!")
                break
            else:
                print("⚠ Please enter a number between 1 and 6")

    def _load_document(self):
        print("\nPaste your contract text below.")
        print("When done, type END on a new line and press Enter:")
        print("-" * 40)
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            except UnicodeDecodeError:
                print("⚠ Encoding issue on this line, skipping...")
                continue
        text = "\n".join(lines)
        if not text.strip():
            print("⚠ No text entered.")
            return
        doc_id, n, lang = self.rag.ingest(text, doc_name="contract.txt")
        self.current_doc_id   = doc_id
        self.current_doc_name = "contract.txt"
        self.current_lang     = lang
        print(f"\n✅ Document indexed!")
        print(f"   Chunks  : {n}")
        print(f"   Language: {lang}")

    def _require_doc(self):
        if not self.current_doc_id:
            print("⚠ Please load a document first (option 1)")
            return False
        return True

    def _summary(self):
        if not self._require_doc(): return
        print("\n⏳ Retrieving relevant clauses...")
        chunks = self.rag.multi_retrieve([
            "parties involved purpose of contract",
            "main obligations responsibilities",
            "payment salary compensation amount",
            "duration term expiry date"
        ], doc_id=self.current_doc_id, top_k=3)
        context = self.rag.build_context(chunks[:8])
        print("\n" + "="*50)
        print("📝 SUMMARY — Retrieved Clauses")
        print("="*50)
        print(context)
        print(f"\n📊 {len(chunks)} clauses retrieved")

    def _risks(self):
        if not self._require_doc(): return
        print("\n⏳ Scanning for risky clauses...")
        rqs = [
            "termination penalty damages",
            "non-compete restriction",
            "indemnification hold harmless",
            "arbitration waiver",
            "intellectual property assignment",
            "automatic renewal"
        ]
        seen, results = set(), []
        for q in rqs:
            for r in self.rag.retrieve(q, doc_id=self.current_doc_id, top_k=2):
                if r.chunk.id not in seen:
                    results.append(r)
                    seen.add(r.chunk.id)
        print("\n" + "="*50)
        print("⚠  RISK ANALYSIS — Flagged Clauses")
        print("="*50)
        for i, r in enumerate(results[:6], 1):
            print(f"\n[{i}] Clause {r.chunk.metadata['clause_ref']} "
                  f"(score: {r.score})")
            print(f"    {r.chunk.text[:200]}...")
        print(f"\n📊 {len(results)} risky clauses found")

    def _terms(self):
        if not self._require_doc(): return
        print("\n⏳ Extracting key terms...")
        tqs = {
            "Parties"      : "parties names company individual",
            "Dates"        : "effective date start signing",
            "Duration"     : "term duration period expires",
            "Payment"      : "payment amount fee salary",
            "Termination"  : "termination cancel notice",
            "Governing Law": "governing law jurisdiction"
        }
        print("\n" + "="*50)
        print("📋 KEY TERMS")
        print("="*50)
        for label, query in tqs.items():
            res = self.rag.retrieve(query, doc_id=self.current_doc_id, top_k=1)
            if res:
                print(f"\n{label}:")
                print(f"  {res[0].chunk.text[:150]}...")

    def _ask(self):
        if not self._require_doc(): return
        question = input("\nYour question: ").strip()
        if not question:
            return
        print("\n⏳ Searching contract...")
        results  = self.rag.retrieve(question, doc_id=self.current_doc_id, top_k=5)
        context  = self.rag.build_context(results)
        print("\n" + "="*50)
        print("❓ ANSWER — Most Relevant Clauses")
        print("="*50)
        print(context)
        print(f"\n📊 {len(results)} clauses retrieved")