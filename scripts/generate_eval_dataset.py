import argparse
import asyncio
import json
import logging
import os
import random
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("eval_dataset_generator")


def _extract_keywords(text: str, limit: int = 8) -> list[str]:
    words = [
        word.strip(".,:;!?()[]{}\"'").lower()
        for word in text.split()
        if len(word.strip(".,:;!?()[]{}\"'")) >= 4
    ]
    seen = []
    for word in words:
        if word not in seen:
            seen.append(word)
        if len(seen) >= limit:
            break
    return seen


def _with_accuracy_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    source = str(item.get("source", ""))
    item["expected_sources"] = [source] if source else []
    item["expected_context_keywords"] = _extract_keywords(str(item["ground_truth"]))
    return item


async def generate_qa_pair(chunk_text: str, domain: str, source: str) -> Optional[Dict[str, Any]]:
    from src.ollama_provider import chat_completion

    prompt = f"""You are an expert test creator for a RAG evaluation system.
Given the following knowledge text from domain '{domain}' (source: '{source}'), generate:
1. A clear, specific question that can be answered ONLY using the provided text.
2. A direct, accurate answer (ground truth) based ONLY on the text.

Respond ONLY with a JSON object in this format:
{{
  "question": "The question here",
  "ground_truth": "The answer here"
}}

Do not include any explanations, markdown wrappers, or chat text.

KNOWLEDGE TEXT:
---
{chunk_text}
---
JSON:"""

    try:
        content = await chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = re.sub(r"^```json\s*", "", content.strip(), flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content, flags=re.IGNORECASE)
        data = json.loads(content)
        if "question" in data and "ground_truth" in data:
            return _with_accuracy_fields(
                {
                    "question": data["question"],
                    "ground_truth": data["ground_truth"],
                    "context": chunk_text,
                    "domain": domain,
                    "source": source,
                }
            )
    except Exception as exc:
        logger.warning("Failed to generate QA pair for %s: %s", source, exc)
    return None


def scan_knowledge_chunks(docs_dir: str, min_length: int = 300) -> List[Dict[str, Any]]:
    chunks = []
    if not os.path.exists(docs_dir):
        logger.warning("Docs directory not found: %s", docs_dir)
        return chunks

    for root, _, files in os.walk(docs_dir):
        for file_name in files:
            if not file_name.endswith((".md", ".txt")) or file_name.startswith("."):
                continue
            filepath = os.path.join(root, file_name)
            rel_path = os.path.relpath(filepath, docs_dir)
            domain = rel_path.split(os.sep)[0] if rel_path else "unknown"
            try:
                with open(filepath, "r", encoding="utf-8") as file:
                    text = file.read()
                paragraphs = [
                    paragraph.strip()
                    for paragraph in text.split("\n\n")
                    if len(paragraph.strip()) >= min_length
                ]
                for idx, paragraph in enumerate(paragraphs):
                    chunks.append(
                        {
                            "text": paragraph,
                            "domain": domain,
                            "source": f"{rel_path}#p{idx}",
                        }
                    )
            except Exception as exc:
                logger.warning("Error reading %s: %s", filepath, exc)

    logger.info("Scanned %s candidate chunks from docs.", len(chunks))
    return chunks


def _mock_dataset() -> List[Dict[str, Any]]:
    return [
        _with_accuracy_fields(
            {
                "question": "What happens if you eat rotten food in Project Zomboid?",
                "ground_truth": "Eating rotten food in Project Zomboid can cause food poisoning, sickness, or death if untreated.",
                "context": "Eating rotten food will cause sickness and eventually death.",
                "domain": "pz",
                "source": "pz/Food/rotten.md",
            }
        ),
        _with_accuracy_fields(
            {
                "question": "How do you construct a rain collector barrel?",
                "ground_truth": "A rain collector barrel requires Carpentry level 4, 4 planks, 4 nails, and 4 garbage bags.",
                "context": "A basic rain barrel requires Carpentry Level 4, 4 Planks, 4 Nails, and 4 Garbage Bags.",
                "domain": "pz",
                "source": "pz/Crafting/water.md",
            }
        ),
        _with_accuracy_fields(
            {
                "question": "What is the maximum skill level in Project Zomboid?",
                "ground_truth": "The maximum level for any Project Zomboid skill is level 10.",
                "context": "Skills in Project Zomboid range from level 0 to level 10.",
                "domain": "pz",
                "source": "pz/Player/skills.md",
            }
        ),
    ]


async def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QA dataset for evaluation.")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output", default="evaluation/data/qa_dataset.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    mocks = _mock_dataset()

    if args.mock:
        dataset = []
        for idx in range(args.limit):
            item = mocks[idx % len(mocks)].copy()
            item["question"] = f"[MOCK-{idx}] {item['question']}"
            dataset.append(item)
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(dataset, file, indent=2, ensure_ascii=False)
        logger.info("Mocked QA dataset written to %s", args.output)
        return

    chunks = scan_knowledge_chunks(os.path.abspath("knowledge/docs"))
    if not chunks:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(mocks, file, indent=2, ensure_ascii=False)
        logger.info("Mock dataset written to %s", args.output)
        return

    random.shuffle(chunks)
    dataset = []
    for idx, chunk in enumerate(chunks[: args.limit]):
        logger.info("[%s/%s] Generating for source: %s", idx + 1, args.limit, chunk["source"])
        qa = await generate_qa_pair(chunk["text"], chunk["domain"], chunk["source"])
        if qa:
            dataset.append(qa)
        else:
            fallback = mocks[idx % len(mocks)].copy()
            fallback["context"] = chunk["text"]
            fallback["domain"] = chunk["domain"]
            fallback["source"] = chunk["source"]
            dataset.append(_with_accuracy_fields(fallback))
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(dataset, file, indent=2, ensure_ascii=False)

    logger.info("Dataset generation complete. Total %s items written to %s", len(dataset), args.output)


if __name__ == "__main__":
    asyncio.run(main())
