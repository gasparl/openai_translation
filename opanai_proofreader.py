import asyncio
import json
import logging
import random
import sys
import time
from pathlib import Path
from functools import lru_cache
from openai import AsyncOpenAI
from openai import OpenAIError, RateLimitError, APIConnectionError, APITimeoutError
from docx import Document
import tiktoken
from datetime import datetime
import argparse
from collections import deque

# --- Model selection (change here!) ---
AVAILABLE_MODELS = [
    "gpt-4o",
    "gpt-4.1",
    "gpt-4o-mini",
    "o1",
    "o1-mini",
    "o1-pro",
    "o3-mini",
]

DEFAULT_MODEL = "gpt-4o"
# --------------------------------------

CHUNK_TOKENS = 800
MAX_COMPLETION_TOKENS = 256
RETRIES = 5
CONCURRENCY = 2      # ≤2 for gpt-4.1 to avoid 429s

# Token/minute rate limits (adjust if needed per model):
TPM_LIMITS = {
    "gpt-4o":     30_000,
    "gpt-4.1":    30_000,
    "gpt-4o-mini":200_000,
    "o1":         30_000,
    "o1-mini":    200_000,
    "o1-pro":     30_000,
    "o3-mini":    200_000,
}

WINDOW = 60  # seconds

WRITE_EVERY = 5   # How often to save the report-in-progress (set 1 to write after each)

script_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
config_path = Path(script_dir, "config.json")

# Set up logging: silence OpenAI SDK's HTTP logs, keep our own info
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
for noisy in ("openai", "httpx", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

try:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    api_key = config["OPENAI_API_KEY"]
except Exception as e:
    logging.critical("Missing or invalid config.json, or OPENAI_API_KEY missing.")
    raise SystemExit(1)

client = AsyncOpenAI(api_key=api_key)

@lru_cache(maxsize=1)
def _encoder(model_name=DEFAULT_MODEL):
    # Safe fallback if tiktoken doesn't know the model
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")

def token_len(text: str, model_name=DEFAULT_MODEL) -> int:
    return len(_encoder(model_name).encode(text))

def read_docx(path: str) -> list[str]:
    return [p.text for p in Document(str(path)).paragraphs if p.text.strip()]

def split_paragraphs(paragraphs: list[str], model_name=DEFAULT_MODEL) -> list[str]:
    chunks, current, tokens = [], [], 0
    for p in paragraphs:
        p_tokens = token_len(p, model_name) + 1  # newline
        # If single paragraph too long, split it by sentence
        if p_tokens > CHUNK_TOKENS:
            import re
            sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', p.replace('\n', ' ')) if s.strip()]
            sent_chunk, sent_tokens = [], 0
            for s in sents:
                s_token = token_len(s, model_name) + 1
                if sent_tokens + s_token > CHUNK_TOKENS and sent_chunk:
                    chunks.append(". ".join(sent_chunk))
                    sent_chunk, sent_tokens = [], 0
                sent_chunk.append(s)
                sent_tokens += s_token
            if sent_chunk:
                chunks.append(". ".join(sent_chunk))
            continue

        if tokens + p_tokens > CHUNK_TOKENS and current:
            chunks.append("\n".join(current))
            current, tokens = [], 0
        current.append(p)
        tokens += p_tokens
    if current:
        chunks.append("\n".join(current))
    return chunks

SYSTEM_PROMPT = (
    "Vizsgáld meg, hogy az adott teljes szekcióban található-e bármilyen egyértelmű "
    "helyesírási, elírási, ragozási vagy nyelvtani egyeztetési hiba, különösen az "
    "egybe- és különírás (pl. mindeközben vs. mind eközben) esetében. "
    "A tárgyas ragozás elhagyása (pl. »fogom a fejem« a »fogom a fejemet« helyett) "
    "nem számít hibának. Ne javasolj stilisztikai, stílus- vagy tartalmi átfogalmazást, és ne írd át a szöveget. "
    "A talált hibákat pontokba szedve sorold fel: idézd a problémás részletet, majd adj rövid magyarázatot "
    "(pl. »tűnődőm - elírás, helyesen tűnődöm«). "
    "Ha az adott teljes szekcióban egyetlen nyilvánvaló hibát sem találsz, "
    "válaszolj teljesen üres sztringgel — ne küldj semmilyen karaktert, sortörést vagy szóközt."
)

# -- Token/minute limiter --
class TokenLimiter:
    def __init__(self, model):
        self.limit = TPM_LIMITS.get(model, 30_000)
        self.window = WINDOW
        self.tokens_window = deque()
        self.lock = asyncio.Lock()
    async def throttle(self, tokens: int):
        async with self.lock:
            now = time.time()
            while self.tokens_window and now - self.tokens_window[0][0] > self.window:
                self.tokens_window.popleft()
            used = sum(t for _, t in self.tokens_window)
            if used + tokens > self.limit:
                to_wait = self.window - (now - self.tokens_window[0][0]) + 0.05
                to_wait = max(to_wait, 0.1)
                logger.info(f"TPM limit reached, sleeping {to_wait:.1f} sec...")
                await asyncio.sleep(to_wait)
            self.tokens_window.append((time.time(), tokens))

# Concurrency limiter (set above)
sem = asyncio.Semaphore(CONCURRENCY)

async def proofread(chunk: str, model_name: str, limiter: TokenLimiter, section_num=None, total=None) -> str:
    async with sem:
        for attempt in range(RETRIES):
            try:
                prompt_tokens = token_len(SYSTEM_PROMPT, model_name)
                chunk_tokens = token_len(chunk, model_name)
                # Defensive context window: gpt-4.1 supports 1M, others 128k
                limit = 1_000_000 if model_name.startswith("gpt-4.1") else 128_000
                available = limit - prompt_tokens - chunk_tokens
                max_completion = min(MAX_COMPLETION_TOKENS, max(32, available))
                # Throttle for TPM
                await limiter.throttle(prompt_tokens + chunk_tokens + max_completion)
                if section_num is not None and total is not None:
                    logger.info(f"Proofreading section {section_num}/{total} ...")
                    
                # choose the right length-limit parameter
                if model_name.startswith(("o", "gpt-4o", "gpt-4.1")):
                    # “reasoning” or new GPT-4o models
                    length_kwarg = {"max_completion_tokens":  max_completion}
                    # o-series ignores temperature/top_p etc.
                else:
                    length_kwarg = {
                        "max_tokens": max_completion,
                        "temperature": 0.1
                    }
                
                resp = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": chunk}
                    ],
                    **length_kwarg
                )

                return resp.choices[0].message.content.strip()
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                if attempt == RETRIES - 1:
                    raise
                wait = 2 ** attempt + random.random()
                logger.warning(f"{type(e).__name__}: {e}. Retrying in {wait:.1f}s")
                await asyncio.sleep(wait)
            except OpenAIError as e:
                raise

async def main(input_docx: str, output_txt: str, model_name: str):
    try:
        paragraphs = read_docx(input_docx)
    except Exception as e:
        logger.error(f"Failed to read DOCX: {e}")
        return

    chunks = split_paragraphs(paragraphs, model_name)
    total = len(chunks)
    logger.info(f"{total} chunks to proofread.")

    now = datetime.now().strftime('%Y%m%d_%H%M')
    header = (
        f"--- Proofreading Report ---\n"
        f"Model: {model_name}\n"
        f"Date: {now}\n"
        f"Source file: {Path(input_docx).name}\n"
        f"---\n\n"
    )

    output_path = Path(output_txt)
    out_lines = [header]
    limiter = TokenLimiter(model_name)

    for i, chunk in enumerate(chunks, 1):
        try:
            result = await proofread(chunk, model_name, limiter, section_num=i, total=total)
        except Exception as e:
            logger.error(f"Chunk {i} failed: {e}")
            result = f"Error during proofreading: {e}"
        out_lines.append(f"-- Section {i} --\n{result}\n\n")

        if i % WRITE_EVERY == 0 or i == total:
            output_path.write_text("".join(out_lines), encoding="utf-8")
            logger.info(f"Progress saved ({i}/{total} sections).")

    logger.info(f"Proofreading completed. Output written to {output_path}")

# -------------- Universal Async Runner Helper --------------
def run_async(coro):
    """
    Run a coroutine in any environment: terminal, Spyder, or Jupyter.
    """
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        task = loop.create_task(coro)
        loop.run_until_complete(task)
    else:
        asyncio.run(coro)

# -------------- Interactive "one-liner" function --------------
def easy_proofread(
    model="gpt-4o",
    input_docx="input.docx",
    output_txt=None,
):
    """
    Run proofreading with a chosen model, input, and (optional) output filename.
    - model:       Model name as string, e.g. 'gpt-4o', 'o3', etc.
    - input_docx:  Input DOCX filename (str or Path)
    - output_txt:  Output TXT filename (str or Path). If None, auto-generates.
    """
    from datetime import datetime
    from pathlib import Path
    import sys

    script_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    input_path = Path(script_dir, input_docx)
    now = datetime.now().strftime('%Y%m%d_%H%M')
    output_path = Path(script_dir, output_txt) if output_txt else Path(
        script_dir, f"proofreading_report_{model}_{now}.txt"
    )
    # Windows event loop fix for Spyder/Windows
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print(f"\nProofreading {input_path} with {model} ...\n")
    run_async(main(input_path, output_path, model))
    print(f"\nDone! Output saved to {output_path}\n")

# -------------- Command-line interface --------------
if __name__ == "__main__":
    # Run CLI only if extra command-line arguments were supplied
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Proofread a DOCX using an OpenAI model.")
        parser.add_argument("--model", type=str, choices=AVAILABLE_MODELS, default=DEFAULT_MODEL, help="OpenAI model to use")
        parser.add_argument("--input", type=str, default="input.docx", help="Input DOCX file")
        parser.add_argument("--output", type=str, default=None, help="Output TXT file (default: auto-named)")
        args = parser.parse_args()
        input_docx = Path(script_dir, args.input)
        now = datetime.now().strftime('%Y%m%d_%H%M')
        output_txt = (
            Path(script_dir, args.output)
            if args.output else
            Path(script_dir, f"proofreading_report_{args.model}_{now}.txt")
        )
        run_async(main(input_docx, output_txt, args.model))

# -------------- Example one-liners for Spyder/Jupyter --------------
# Uncomment and edit as needed for your workflow:
# easy_proofread()  # Default: gpt-4o, input.docx
# easy_proofread(model="o3")
# easy_proofread(input_docx="my_paper.docx")
# easy_proofread(model="o3-pro", input_docx="report.docx", output_txt="proofed_report.txt")
# easy_proofread(model="gpt-4.1", input_docx="document.docx")
