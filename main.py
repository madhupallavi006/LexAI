

from rag_pipeline import LegalRAGPipeline
from ui import LexAIUI

# Start the RAG engine
rag = LegalRAGPipeline()

# Launch the UI
ui = LexAIUI(rag)
ui.render()