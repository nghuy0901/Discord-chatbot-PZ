# Knowledge Base Module

This module provides the **Multi-Domain Knowledge Base** system for NomNom Bot.

## Structure

```
knowledge/
├── docs/               # Static knowledge documents (markdown, txt)
│   ├── pz/             # Project Zomboid domain
│   ├── server_rules/   # Discord server rules
│   └── general/        # General/community knowledge
├── prompts/            # Domain-specific prompts
├── manager.py          # KnowledgeManager — load, chunk, embed, CRUD
├── domain_router.py    # Route queries to relevant knowledge domains
└── README.md           # This file
```

## Adding Knowledge

1. Place `.md` or `.txt` files in the relevant `docs/<domain>/` folder.
2. Use `@Bot reload knowledge` to reload without restarting.
3. Use `@Bot scan knowledge` to see what's loaded.

## Domain Configuration

Each domain can have a custom prompt in `prompts/<domain>.txt`.
The prompt is appended to the system prompt when a query matches that domain.
