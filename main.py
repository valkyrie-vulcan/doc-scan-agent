from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

# --- Import our agent executor ---
# This is the compiled LangGraph agent we built in agent_core.py
from agent_core import agent_executor, AgentState

# ==============================================================================
# 1. INITIALIZE THE FASTAPI APP
# ==============================================================================

app = FastAPI(
    title="KYC Document Extraction Agent",
    description="An API for extracting information from Indian KYC documents using an AI agent.",
    version="1.0.0",
)

# ==============================================================================
# 2. DEFINE THE API ENDPOINT
# ==============================================================================

@app.post("/v1/extract", summary="Extract KYC data from a document image")
async def extract_kyc_data(
    image: UploadFile = File(..., description="The image file of the KYC document (e.g., Aadhaar, PAN)."),
    external_transaction_id: Optional[str] = Form(None, description="An optional external ID to track this transaction.")
):
    """
    This endpoint processes an uploaded document image to extract key information.

    - **Receives**: An image file and an optional transaction ID.
    - **Processes**: Uses a LangGraph-based AI agent to perform OCR, extract structured data with Gemini, and log the transaction in MongoDB.
    - **Returns**: A JSON response with the extraction status, data, and a unique database record ID for auditing.
    """
    try:
        # --- Read image bytes from the uploaded file ---
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Image file is empty.")

        # --- Prepare the initial state for the agent ---
        initial_state: AgentState = {
            "image_bytes": image_bytes,
            "source_filename": image.filename,
            "external_transaction_id": external_transaction_id,
            "status": "PROCESSING",
            # The rest of the fields are initialized as None implicitly or will be populated by the agent
        }

        # --- Invoke the agent to run the entire workflow ---
        print(f"Invoking agent for file: {image.filename}...")
        final_state = agent_executor.invoke(initial_state)
        print("Agent finished processing.")

        # --- Format the final response based on the agent's outcome ---
        status = final_state.get("status", "ERROR_UNKNOWN")
        db_id = final_state.get("db_document_id")

        if status == "SUCCESS":
            return JSONResponse(
                status_code=200,
                content={
                    "status": "SUCCESS",
                    "data": final_state.get("structured_data"),
                    "database_record_id": db_id,
                },
            )
        else:
            # For any error status, return a structured error response
            # We use 500 for server-side issues (OCR, Gemini, DB failures)
            return JSONResponse(
                status_code=500,
                content={
                    "status": status,
                    "message": final_state.get("error_message"),
                    "database_record_id": db_id,
                },
            )

    except Exception as e:
        # This is a fallback for unexpected errors in the API layer itself
        print(f"An unexpected error occurred in the API layer: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"An unexpected server error occurred: {str(e)}"
        )

# ==============================================================================
# 3. DEFINE A ROOT ENDPOINT FOR HEALTH CHECKS
# ==============================================================================

@app.get("/", summary="Health Check")
def read_root():
    """A simple health check endpoint to confirm the API is running."""
    return {"status": "ok", "message": "KYC Agent API is running."}


# ==============================================================================
# 4. RUN THE SERVER (for direct execution)
# ==============================================================================

if __name__ == "__main__":
    # This block allows you to run the server directly using "python main.py"
    # Uvicorn is a high-performance ASGI server.
    print("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

