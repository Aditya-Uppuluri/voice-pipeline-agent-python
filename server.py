# server.py

from fastapi import FastAPI, Request, HTTPException
import uvicorn
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
CONTEXT_PATH = "latest_context.txt"

@app.post("/inject-context")
async def inject_context(request: Request):
    try:
        raw_body = await request.body()
        context_string = raw_body.decode("utf-8")
        
        if not context_string.strip():
            logger.warning("Received empty context string")
            return {"status": "warning", "message": "Empty context received", "length": 0}
        
        # Save context to file with proper encoding
        with open(CONTEXT_PATH, "w", encoding="utf-8") as f:
            f.write(context_string)
        
        logger.info(f"Context saved successfully. Length: {len(context_string)} characters")
        return {"status": "success", "length": len(context_string)}
        
    except Exception as e:
        logger.error(f"Error saving context: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save context: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/context-status")
async def context_status():
    """Check if context file exists and return its info"""
    if os.path.exists(CONTEXT_PATH):
        try:
            with open(CONTEXT_PATH, "r", encoding="utf-8") as f:
                content = f.read()
            return {
                "exists": True,
                "length": len(content),
                "preview": content[:200] + "..." if len(content) > 200 else content
            }
        except Exception as e:
            return {"exists": True, "error": str(e)}
    else:
        return {"exists": False}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)