# Server Knowledge Base — Chấn thương tâm lý

## Server Overview

- **Server Name**: Chấn thương tâm lý (Psychological Trauma)
- **Server ID**: 1019905921641627699
- **Primary Channel**: 💬┃cư-dân-chat (channel_id: 1267863603382587573)
- **Language**: Vietnamese (Tiếng Việt) — casual, slang-heavy
- **Total Archived Messages**: ~94,000+
- **Active Period**: Up to February 2026

## Community Culture

This is a Vietnamese gaming and social community. The chat style is:

- **Casual & fast-paced**: Short messages, lots of slang, abbreviations, emojis
- **Gaming-focused**: Members discuss and play games together (survival, RPG, co-op)
- **Vietnamese internet slang**: Common abbreviations like "z" = "vậy", "r" = "rồi", "đc" = "được", "bt" = "biết", "ae" = "anh em"
- **Emoji-heavy**: Discord custom emotes used frequently (e.g., `:dafoe:`, `:bruh:`, `:Meme:`)
- **Reply chains**: Conversations flow through Discord replies — context is often in the parent message

## Known Members (from chat history)

| Display Name | Role / Notes |
|---|---|
| Harvey | Active member, trades in-game items |
| T Rí | Active chatter |
| plamm | Frequent poster, shares game moments |
| Rec.Kaiz (Seitatsu) | Active member, jokes around |
| 3KuDespot | Posts screenshots, game commentary |
| Anh Tùng Thợ Điện | Occasional chatter |
| Ddaof | Early riser, casual chat |

*(This list is partial — there are many more members in the server)*

## Common Topics

1. **Gaming**: In-game economics (hunger, copper as currency), combat, farming, survival mechanics
2. **Daily life**: Food, schedules, work, joking around
3. **Server events**: Trades, co-op sessions, group activities
4. **Memes & reactions**: Gif sharing, emote spam, inside jokes

## Data Format Notes

The archived chat data has this structure per message:

```json
{
  "id": "message_id",
  "timestamp": "ISO-8601",
  "author": "display_name",
  "author_id": "discord_user_id",
  "content": "message text",
  "reply_to": {
    "author": "parent_author",
    "author_id": "parent_author_id",
    "content": "parent_message_content",
    "message_id": "parent_message_id"
  },
  "total_reactions": 0,
  "embeds": null,
  "is_edited": false,
  "attachments": ["url1", "url2"]
}
```

### Key field mappings for RAG ingestion:
- `id` → message_id
- `author` → author_name (string, not dict)
- `author_id` → author_id
- `reply_to.message_id` → parent reference (edge)
- `channel_id` → may be absent (default: `1267863603382587573`)
- Messages with empty `content` but having `attachments` are image-only posts (skip for text RAG)

## RAG Usage Guidelines

When the bot retrieves historical messages from this dataset:

1. **Language matching**: The retrieved content is in Vietnamese — respond in the same language the user writes in
2. **Context awareness**: Reply chains are important — a message's `reply_to` gives crucial context
3. **Slang translation**: Don't try to "correct" Vietnamese slang — mirror the community's casual tone
4. **Citation format**: When citing retrieved history, use `[1]`, `[2]` etc. with author name and timestamp
5. **Gaming context**: Many messages reference in-game activities — understand they're about game mechanics, not real life
6. **Emoji handling**: Custom Discord emotes like `<:dafoe:1272526970328191100>` can be ignored or treated as reactions
