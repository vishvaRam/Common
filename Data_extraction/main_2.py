import os
import json
from typing import List, Optional, Dict, Any
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

MODEL_NAME = "gemini-2.5-flash"

# ============================================================================
# PYDANTIC MODELS - Banking Standard Format
# ============================================================================

class HeaderDetails(BaseModel):
    """Specific header fields found in banking notes"""
    department: str = Field(description="Department Name (e.g., Information Technology Department)")
    date: str = Field(description="Date of the note")
    committee_name: Optional[str] = Field(description="Name of the committee (e.g., ITSC, Audit Committee)")
    agenda_no: Optional[str] = Field(description="Agenda Number if present")
    critical_theme: Optional[str] = Field(description="Theme (e.g., Compliance, Strategy)")
    purpose: Optional[str] = Field(description="Purpose (e.g., For Permission, For Information)")

class TableData(BaseModel):
    """Represents a table"""
    title: Optional[str] = Field(description="Title or caption of the table")
    headers: List[str] = Field(description="Column headers")
    rows: List[Dict[str, str]] = Field(description="Rows as key-value pairs matching headers")

class BodySection(BaseModel):
    """A logical section of the note"""
    title: str = Field(description="Section Title (e.g., Present Requirement, Prayer)")
    content: str = Field(description="The textual content of the section")
    tables: List[TableData] = Field(default=[], description="Tables embedded in this section")

class Signatory(BaseModel):
    """Signatory details"""
    designation: str = Field(description="Designation (e.g., DVP, VP, EVP)")
    department: Optional[str] = Field(description="Department of the signatory")

class BankingNote(BaseModel):
    """The complete structure of the Banking Approval Note"""
    header: HeaderDetails
    main_subject: str = Field(description="The main subject line of the note")
    
    # Standard sections usually found
    background_section: Optional[BodySection] = Field(description="Context/Background section")
    requirement_section: Optional[BodySection] = Field(description="Present Requirement/Proposal section")
    prayer_section: Optional[BodySection] = Field(description="Prayer/Recommendation section")
    
    # Generic list for any other sections detected
    other_sections: List[BodySection] = Field(default=[], description="Any other sections found")
    
    signatories: List[Signatory] = Field(default=[], description="List of officials signing the document")

# ============================================================================
# DOCUMENT PROCESSOR
# ============================================================================

class BankingNoteProcessor:
    def __init__(self, client: OpenAI, model: str = MODEL_NAME):
        self.client = client
        self.model = model
    
    def process_document(self, input_text: str) -> BankingNote:
        print("\n" + "="*70)
        print("🏦 BANKING NOTE PROCESSOR - Strict Standard Format")
        print("="*70)
        print(f"📄 Input Size: {len(input_text)} chars")

        system_prompt = """You are a Banking Compliance Officer. 
        Convert the raw input text into a structured 'Board Approval Note' strictly following this format:
        
        1. **Header:** Extract Dept, Date, Committee Name, Agenda No, Theme, Purpose.
        2. **Subject:** The main title of the note.
        3. **Sections:**
           - **Background:** Historical context, previous approvals.
           - **Present Requirement:** RBI circulars, audit observations, new needs.
           - **Prayer:** The specific approval request.
        4. **Tables:** Preserve all tables exactly as they appear (e.g., List of CII items).
        5. **Signatories:** Extract the list of officials at the bottom.

        If specific headers are missing, infer them from context or leave empty.
        """

        try:
            print("🤖 analyzing...")
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"--- DOCUMENT START ---\n{input_text}\n--- DOCUMENT END ---"}
                ],
                response_format=BankingNote,
            )
            return completion.choices[0].message.parsed

        except Exception as e:
            print(f"❌ Analysis Failed: {e}")
            return None

# ============================================================================
# MARKDOWN GENERATOR (Visual Replica)
# ============================================================================

def generate_formatted_markdown(note: BankingNote, output_path: str):
    if not note: return

    md = []
    
    # 1. HEADER TABLE (Mimicking the visual layout)
    md.append("| | **Managing Director & CEO** | |")
    md.append("| :--- | :---: | :--- |")
    md.append(f"| **{note.header.department}** | | **Date:** {note.header.date} |")
    md.append(f"| **Committee:** {note.header.committee_name or '-'} | | **Agenda No:** {note.header.agenda_no or '-'} |")
    md.append(f"| **Theme:** {note.header.critical_theme or '-'} | | **Purpose:** {note.header.purpose or '-'} |")
    md.append("")
    md.append("---")
    md.append("")

    # 2. MAIN SUBJECT
    md.append(f"### **{note.main_subject}**")
    md.append("")
    
    # 3. HELPER: Render Section
    def render_section(sec: Optional[BodySection]):
        if not sec: return
        # Title usually not repeated if it's implicit, but let's add it if bold
        if sec.title and sec.title.lower() not in ["background", "introduction"]: 
             md.append(f"**{sec.title}:**")
        
        md.append(sec.content)
        md.append("")
        
        for tbl in sec.tables:
            if tbl.title: md.append(f"*{tbl.title}*")
            if tbl.headers:
                md.append("| " + " | ".join(tbl.headers) + " |")
                md.append("|" + "|".join(["---"] * len(tbl.headers)) + "|")
                for row in tbl.rows:
                    vals = [str(row.get(h, "")) for h in tbl.headers]
                    md.append("| " + " | ".join(vals) + " |")
            md.append("")

    # 4. SECTIONS
    render_section(note.background_section)
    render_section(note.requirement_section)
    
    # Other sections detected dynamically
    for sec in note.other_sections:
        render_section(sec)

    render_section(note.prayer_section)

    # 5. SIGNATORIES (Footer Table)
    if note.signatories:
        md.append("---")
        md.append("")
        md.append("**Submitted By:**")
        md.append("")
        # Create a simple list or table for signatories
        md.append("| Designation | Department |")
        md.append("| :--- | :--- |")
        for sig in note.signatories:
            md.append(f"| {sig.designation} | {sig.department or ''} |")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print(f"✅ Formatted Note Saved: {output_path}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    INPUT_FILE = "Data_extraction/ApprovalNoteOriginal.md"
    OUTPUT_FILE = "Data_extraction/ApprovalNote_Formatted.md"
    
    # Ensure dir exists
    os.makedirs(os.path.dirname(INPUT_FILE), exist_ok=True)
    
    # Read Input
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            
        processor = BankingNoteProcessor(client)
        result = processor.process_document(content)
        generate_formatted_markdown(result, OUTPUT_FILE)
        
    except FileNotFoundError:
        print(f"❌ Input file not found: {INPUT_FILE}")
