import asyncio
import os
import logging
from knowledge.manager import KnowledgeManager

logging.basicConfig(level=logging.DEBUG)

async def test():
    km = KnowledgeManager()
    print("Docs dir:", km.docs_dir)
    print("Is dir:", os.path.isdir(km.docs_dir))
    domains = km.discover_domains()
    print("Domains:", domains)
    for d in domains:
        docs_path = os.path.join(km.docs_dir, d)
        print(f"Docs path for {d}:", docs_path)
        files = km._scan_files(docs_path)
        print(f"Files for {d}:", len(files))
    
    res = await km.load_all()
    print("Load all result:", res)

if __name__ == "__main__":
    asyncio.run(test())
