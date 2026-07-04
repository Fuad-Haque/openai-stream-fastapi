import os
import json
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("stream")

app = FastAPI()


client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "llama-3.1-8b-instant"


async def token_generator(request: Request, prompt: str):
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )


    token_count = 0
    try:
        for chunk in stream:
            if await request.is_disconnected():
                logger.warning(
                    f"client disconnected after {token_count} tokens — closing upstream stream"
                )
                stream.close()
                return


            delta = chunk.choices[0].delta.content
            if delta:
                token_count += 1
                yield f"data: {json.dumps({'token': delta})}\n\n"

        yield f"data: {json.dumps({'done': True, 'total_tokens': token_count})}\n\n"
        logger.info(f"stream completed normally — {token_count} tokens sent")

    except Exception as e:
        logger.error(f"stream error after {token_count} tokens: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

    finally:
        stream.close()
        logger.info(f"generator exited — upstream connection closed, {token_count} tokens")

@app.get("/stream")
async def stream_endpoint(request: Request, prompt: str = "Tell me a short fact about the ocean."):
    return StreamingResponse(
        token_generator(request, prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}