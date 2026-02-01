from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import asyncio
import json


class ChatRequest(BaseModel):
    message: str


app = FastAPI(title="Simple Backend")

# Allow Next.js dev server to call this API directly (optional if you proxy later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "service": "fastapi-backend"}


@app.get("/echo")
def echo(q: str = "hello"):
    return {"q": q}


@app.post("/chat")
def chat(req: ChatRequest):
    # dummy response for now (later replace with LLM call)
    return {"reply": f"you said: {req.message}"}


@app.post("/chat_stream")
async def chat_stream():
    async def gen():
        # "delta" events
        md = [
            "# Streaming Markdown Demo\n\n",
            "Here’s a realistic LLM-style response with **Markdown** formatting.\n\n",
            "## Key points\n",
            "- Uses headings, lists, tables, and code blocks\n",
            "- Streams token-by-token (SSE)\n",
            "- Great for testing `react-markdown` + `remark-gfm`\n\n",
            "## Checklist\n",
            "- [x] SSE working\n",
            "- [x] Markdown rendering\n",
            "- [ ] Add syntax highlighting later\n\n",
            "## Mini table\n",
            "| Feature | Status | Notes |\n",
            "|---|---:|---|\n",
            "| SSE | ✅ | `text/event-stream` |\n",
            "| Proxy | ✅ | Next Route Handler |\n",
            "| UI | ✅ | Tailwind + Markdown |\n\n",
            "## Code sample\n",
            "```python\n",
            "def hello(name: str) -> str:\n",
            '    return f"Hello, {name}!"\n',
            "\n",
            'print(hello("stream"))\n',
            "```\n\n",
            "> Tip: If code blocks flicker during streaming, buffer updates (e.g., update UI every 50–100ms).\n\n",
            "Done.\n",
        ]

        for token in md:
            payload = {"data": token}
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(0.2)

        # OpenAI-style terminator
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
