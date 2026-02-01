import os
import json
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("⚠️ Set GEMINI_API_KEY environment variable")
    # For testing only
    api_key = "YOUR_KEY_HERE"

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Using Flash for speed/cost, Pro for complex reasoning
MODEL_NAME = "gemini-2.5-flash"

# ============================================================================
# PYDANTIC MODELS - Dynamic Structure
# ============================================================================

class TableData(BaseModel):
    """Represents a single table extracted from the text"""
    title: str = Field(description="A descriptive title for the table")
    headers: List[str] = Field(description="Column headers")
    rows: List[Dict[str, str]] = Field(description="List of rows, where keys match headers")

class DocumentSection(BaseModel):
    """A logical section detected by the LLM"""
    section_title: str = Field(description="Title of the section (e.g., 'Background', 'Financials')")
    summary_content: str = Field(description="Condensed text summary of the section (40-60% of original).")
    key_points: List[str] = Field(default=[], description="Bullet points of critical info (dates, amounts, risks)")
    tables: List[TableData] = Field(default=[], description="Any tables found in this section")
    
class CondensedDocument(BaseModel):
    """The complete analyzed document"""
    document_title: str = Field(description="Main title of the document")
    document_date: Optional[str] = Field(description="Date of the document")
    executive_summary: str = Field(description="2-paragraph executive summary for top management")
    sections: List[DocumentSection] = Field(description="List of all logical sections found in the doc")
    critical_action_items: List[str] = Field(description="List of required approvals or actions")

# ============================================================================
# DOCUMENT PROCESSOR
# ============================================================================

class BankingNoteCondenser:
    def __init__(self, client: OpenAI, model: str = MODEL_NAME):
        self.client = client
        self.model = model
    
    def process_document(self, markdown_file_path: str) -> CondensedDocument:
        print("\n" + "="*70)
        print("BANKING NOTE CONDENSER - Auto-Detection Mode")
        print("="*70)
        
        # 1. Read File
        try:
            with open(markdown_file_path, 'r', encoding='utf-8') as f:
                full_doc = f.read()
        except FileNotFoundError:
            print(f"❌ Error: Input file '{markdown_file_path}' not found.")
            return None

        print(f"📄 Input Size: {len(full_doc)} chars")

        # 2. Define System Prompt
        system_prompt = """You are an expert Banking Analyst. 
        Your task is to summarize a Board Note or Financial Document.
        
        1. **Analyze Structure:** Auto-detect the logical sections (e.g., Background, Analysis, Financials, Recommendations). Do not force a format if it doesn't fit.
        2. **Summarize:** Condense text to 50% length but PRESERVE all specific numbers, dates, circular refs, and compliance issues.
        3. **Extract Tables:** If you see tabular data (or data that looks like a table), extract it structured. 
           - Consolidate similar small tables if they belong together.
           - Ensure 'headers' list matches the keys in 'rows'.
        4. **Action Items:** Explicitly list what is being asked (Approvals, Budget, Policy changes).
        """

        # 3. Call OpenAI-compatible API with Structured Output
        print("🤖 Analyzing document (this may take a moment)...")
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"--- BEGIN DOCUMENT ---\n{full_doc}\n--- END DOCUMENT ---"}
                ],
                response_format=CondensedDocument,
            )
            
            result = completion.choices[0].message.parsed
            print("✅ Analysis Complete!")
            return result

        except Exception as e:
            print(f"❌ API Error: {e}")
            return None

# ============================================================================
# MARKDOWN GENERATOR
# ============================================================================

def generate_output_markdown(condensed: CondensedDocument, output_path: str):
    """Generate clean markdown output from structured object"""
    if not condensed:
        return

    md = []
    
    # 1. Header
    md.append(f"# {condensed.document_title}")
    if condensed.document_date:
        md.append(f"**Date:** {condensed.document_date}")
    md.append("---")
    
    # 2. Executive Summary
    md.append("## 📌 Executive Summary")
    md.append(condensed.executive_summary)
    md.append("")
    
    # 3. Action Items
    if condensed.critical_action_items:
        md.append("## ⚡ Critical Action Items")
        for item in condensed.critical_action_items:
            md.append(f"- [ ] {item}")
        md.append("")

    # 4. Sections & Tables
    md.append("---")
    for sec in condensed.sections:
        md.append(f"## {sec.section_title}")
        md.append(sec.summary_content)
        md.append("")
        
        # Key Points
        if sec.key_points:
            for kp in sec.key_points:
                md.append(f"- {kp}")
            md.append("")

        # Tables
        if sec.tables:
            for tbl in sec.tables:
                md.append(f"### *{tbl.title}*")
                
                # Check if headers exist
                if not tbl.headers:
                    # Try to infer headers from first row if missing
                    if tbl.rows:
                        tbl.headers = list(tbl.rows[0].keys())
                    else:
                        continue # Skip empty table

                # Header Row
                header_row = "| " + " | ".join(tbl.headers) + " |"
                separator_row = "|" + "|".join(["---"] * len(tbl.headers)) + "|"
                
                md.append(header_row)
                md.append(separator_row)
                
                # Data Rows
                for row in tbl.rows:
                    # Map row values to headers order
                    row_values = []
                    for h in tbl.headers:
                        val = row.get(h, "") # Get value or empty string
                        row_values.append(str(val))
                    
                    md.append("| " + " | ".join(row_values) + " |")
                md.append("") # Spacing after table
        
        md.append("") # Spacing after section

    # Footer
    md.append("---")
    md.append("*Generated by AI Banking Analyst*")

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print(f"✅ Markdown saved to: {output_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Define paths
    INPUT_FILE = "Data_extraction/ApprovalNoteOriginal.md"
    OUTPUT_FILE = "Data_extraction/ApprovalNote_Condensed.md"
    
    # Create directory if missing
    os.makedirs(os.path.dirname(INPUT_FILE), exist_ok=True)


    # Run Process
    condenser = BankingNoteCondenser(client)
    result = condenser.process_document(INPUT_FILE)
    
    if result:
        generate_output_markdown(result, OUTPUT_FILE)
