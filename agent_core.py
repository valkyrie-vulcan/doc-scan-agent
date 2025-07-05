# agent_core.py

import os

import io
from datetime import datetime
from typing import TypedDict, Dict, Any, Optional

# --- Environment and Third-Party Imports ---
import pymongo
from rapidocr_onnxruntime import RapidOCR
import cv2
import numpy as np
from PIL import Image

# --- LangChain/LangGraph Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END

# ==============================================================================
# 1. INITIALIZATION & CONFIGURATION
# ==============================================================================

# --- Initialize Global Clients (for performance) ---
try:
    print("Initializing clients...")
    # Initialize the OCR engine. This can take a moment on first run.
    ocr_engine = RapidOCR()
    
    # Initialize MongoDB client and select database/collection
    mongo_client = pymongo.MongoClient(os.getenv("MONGO_URI"))
    db = mongo_client[os.getenv("MONGO_DB_NAME")]
    collection = db[os.getenv("MONGO_COLLECTION_NAME")]
    # Test the connection
    mongo_client.server_info() 
    print("MongoDB connection successful.")
    
    print("Clients initialized successfully.")
except Exception as e:
    print(f"FATAL: Failed to initialize clients. Error: {e}")
    # In a real application, you might want to exit or have a retry mechanism
    # For now, we'll just print the error. The app will likely fail later.

# ==============================================================================
# 2. DEFINE THE AGENT'S STATE
# ==============================================================================

class AgentState(TypedDict):
    """
    This TypedDict defines the structure of the "state" that is passed between nodes.
    It's the agent's memory or workspace for a single run.
    """
    image_bytes: bytes
    source_filename: str
    external_transaction_id: Optional[str]

    # Intermediate results
    raw_ocr_text: Optional[str]
    structured_data: Optional[Dict[str, Any]]

    # Final outcome
    status: str # e.g., "PROCESSING", "SUCCESS", "ERROR_OCR", "ERROR_EXTRACTION"
    error_message: Optional[str]
    db_document_id: Optional[str]

# ==============================================================================
# 3. DEFINE THE GRAPH NODES (AGENT'S CAPABILITIES)
# ==============================================================================

def ocr_node(state: AgentState) -> Dict[str, Any]:
    """
    Performs OCR on the image to extract raw text.
    Handles potential errors during the OCR process.
    """
    print("---NODE: Performing OCR---")
    try:
        image_bytes = state['image_bytes']
        
        # Convert image bytes to an OpenCV-compatible format
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode image from bytes.")
            
        # Perform OCR
        result, _ = ocr_engine(img)
        
        if not result:
            raw_text = "" # No text found
        else:
            raw_text = "\n".join([line[1] for line in result])

        print(f"OCR successful. Extracted text snippet: {raw_text[:100]}...")
        return {"raw_ocr_text": raw_text, "status": "PROCESSING"}
    
    except Exception as e:
        print(f"ERROR in OCR node: {e}")
        return {
            "raw_ocr_text": "",
            "status": "ERROR_OCR",
            "error_message": f"Failed during OCR: {str(e)}"
        }

def extraction_node(state: AgentState) -> Dict[str, Any]:
    """
    Uses Gemini to extract structured information from the raw OCR text.
    """
    print("---NODE: Extracting with Gemini---")
    try:
        raw_text = state['raw_ocr_text']
        
        if not raw_text or len(raw_text.strip()) < 10: # Basic check for meaningful text
            raise ValueError("Insufficient text from OCR to perform extraction.")

        prompt_template = """
        You are a highly accurate AI assistant specializing in extracting information from Indian KYC documents.
        Based on the raw OCR text provided below, extract the following details:
        - document_type: Identify the type of document (e.g., 'Aadhaar Card', 'PAN Card', 'Passport').
        - name: The full name of the individual.
        - dob: The date of birth. Convert it to YYYY-MM-DD format.
        - address: The complete address as mentioned in the document.

        OCR TEXT:
        ---
        {ocr_text}
        ---

        IMPORTANT:
        1. Your response MUST be a single, valid JSON object.
        2. Do not include any explanatory text, comments, or markdown formatting like ```json.
        3. If a specific field cannot be found, its value in the JSON should be null.
        """
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0, google_api_key=os.getenv("GEMINI_API_KEY"))
        prompt = ChatPromptTemplate.from_template(template=prompt_template)
        json_parser = JsonOutputParser()
        
        chain = prompt | llm | json_parser
        
        structured_response = chain.invoke({"ocr_text": raw_text})
        
        print(f"Gemini extraction successful: {structured_response}")
        return {"structured_data": structured_response}

    except Exception as e:
        print(f"ERROR in extraction node: {e}")
        return {
            "structured_data": {},
            "status": "ERROR_EXTRACTION",
            "error_message": f"Failed during Gemini extraction: {str(e)}"
        }

def persistence_node(state: AgentState) -> Dict[str, Any]:
    """
    Saves the final state of the process (success or failure) to MongoDB.
    This node is the final step and creates an audit trail.
    """
    print("---NODE: Saving to Database---")
    try:
        # Determine final status. If it's still "PROCESSING", it means success.
        final_status = state['status']
        if final_status == "PROCESSING":
            final_status = "SUCCESS"

        document_to_save = {
            "external_transaction_id": state.get("external_transaction_id"),
            "status": final_status,
            "source_filename": state.get("source_filename"),
            "raw_ocr_text": state.get("raw_ocr_text"),
            "error_message": state.get("error_message"),
            "created_at": datetime.utcnow()
        }
        
        # Only add structured data if the process was successful
        if final_status == "SUCCESS":
            document_to_save["extracted_data"] = state.get("structured_data")

        result = collection.insert_one(document_to_save)
        db_id = str(result.inserted_id)
        
        print(f"Successfully saved record to MongoDB with ID: {db_id}")
        return {"db_document_id": db_id, "status": final_status}

    except Exception as e:
        print(f"CRITICAL ERROR in persistence node: {e}")
        # This is a major issue, as we can't even log the failure.
        # The final state will reflect this problem.
        return {
            "db_document_id": None,
            "status": "ERROR_DATABASE",
            "error_message": f"Critical: Failed to save to MongoDB. Reason: {str(e)}"
        }

# ==============================================================================
# 4. DEFINE THE GRAPH EDGES (AGENT'S LOGIC FLOW)
# ==============================================================================

def decide_after_ocr(state: AgentState) -> str:
    """
    A conditional edge that decides the path after the OCR node.
    If OCR failed, we skip extraction and go straight to saving the error record.
    """
    print("---EDGE: Deciding after OCR---")
    if state["status"] == "ERROR_OCR":
        print("Path -> OCR failed, proceeding to save error record.")
        return "save_to_db"
    else:
        print("Path -> OCR successful, proceeding to extraction.")
        return "extract_details"

# ==============================================================================
# 5. ASSEMBLE AND COMPILE THE AGENT GRAPH
# ==============================================================================

print("Assembling the agent graph...")

# Initialize the stateful graph
workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("perform_ocr", ocr_node)
workflow.add_node("extract_details", extraction_node)
workflow.add_node("save_to_db", persistence_node)

# Set the entry point
workflow.set_entry_point("perform_ocr")

# Add the edges
workflow.add_conditional_edges(
    "perform_ocr",
    decide_after_ocr,
    {
        "extract_details": "extract_details",
        "save_to_db": "save_to_db",
    },
)
workflow.add_edge("extract_details", "save_to_db")
workflow.add_edge("save_to_db", END)

# Compile the graph into a runnable executor
agent_executor = workflow.compile()

print("Agent graph compiled successfully. Ready to process requests.")

# You can uncomment the line below for a quick visualization of the graph's structure
# from PIL import Image; Image.open(io.BytesIO(agent_executor.get_graph().draw_mermaid_png())).show()