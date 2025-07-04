# server.py

from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()
CONTEXT_PATH = "latest_context.txt"

@app.post("/inject-context")
async def inject_context(request: Request):
    raw_body = await request.body()
    context_string = raw_body.decode("utf-8")

    # Save context to file
    with open(CONTEXT_PATH, "w") as f:
        f.write(context_string)

    return {"status": "received", "length": len(context_string)}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
