# ChatGPT Discord Bot

> ### Build your own Discord bot with multiple AI providers

---
> [!IMPORTANT]
>
> **Major Refactor (2026/06):**
> - **Provider-neutral LLM path**: OpenAI, Gemini, and OpenAI-compatible endpoints
> - **Text-first knowledge ingestion**: Markdown, plain text, reStructuredText, and Discord JSON exports


# Setup
## Prerequisites
* **Python 3.9 or later**
* **Rename the file `.env.example` to `.env`**
* Running `pip3 install -r requirements.txt` to install the required dependencies
* Optional: API keys for premium providers (OpenAI, Claude, Gemini, Grok)
---
## Step 1: Create a Discord bot

1. Go to https://discord.com/developers/applications create an application
2. Build a Discord bot under the application
3. Get the token from bot setting
4. Store the token to `.env` under the `DISCORD_BOT_TOKEN`
5. Turn MESSAGE CONTENT INTENT `ON`
6. Invite your bot to your server via OAuth2 URL Generator



## Step 2: Run the bot on the desktop

1. Open a terminal or command prompt

2. Navigate to the directory where you installed the ChatGPT Discord bot

3. Run `python3 main.py` or `python main.py` to run the bot
---
## Step 2: Run the bot with Docker

1. Build the Docker image & run the Docker container with `docker compose up -d`

2. Inspect whether the bot works well `docker logs -t chatgpt-discord-bot`

   ### Stop the bot:

   * `docker ps` to see the list of running services
   * `docker stop <BOT CONTAINER ID>` to stop the running bot

### Have a good chat!
---

## Supported Knowledge Inputs

NomNom currently supports text-based knowledge sources:

- Markdown (`.md`)
- Plain text (`.txt`)
- reStructuredText (`.rst`)
- Discord JSON exports through the ingestion script

OCR, image parsing, PDF extraction, and browser auto-login ingestion are intentionally out of scope for this version.

## Provider Configuration

### LLM Providers

#### OpenAI
1. Obtain your API key from https://platform.openai.com/api-keys
2. Add to `.env`: `LLM_PROVIDER=openai` and `LLM_API_KEY=your_api_key_here`

#### Gemini (Google)
1. Get API key from https://ai.google.dev/
2. Add to `.env`: `LLM_PROVIDER=gemini` and `LLM_API_KEY=your_api_key_here`

#### OpenAI-compatible
1. Serve a model behind an OpenAI-compatible `/v1` endpoint.
2. Add to `.env`: `LLM_PROVIDER=openai_compatible`, `LLM_BASE_URL`, and `LLM_API_KEY` if required by the endpoint.

## Image Generation

Image generation is now integrated with the provider system:

### OpenAI DALL-E 3
- Requires OpenAI API key
- High-quality image generation
- Use `/draw [prompt] openai`

### Google Gemini
- Requires Gemini API key  
- Free tier available
- Use `/draw [prompt] gemini`

### Fallback Options
- Image generation requires an OpenAI-compatible image model or OpenAI image API support

## Optional: Setup system prompt

* A system prompt would be invoked when the bot is first started or reset
* You can set it up by modifying the content in `system_prompt.txt`
* All the text in the file will be fired as a prompt to the bot
* Get the first message from ChatGPT in your discord channel!
* Go Discord setting turn `developer mode` on

   1. Right-click the channel you want to recieve the message, `Copy  ID`

   2. paste it into `.env` under `DISCORD_CHANNEL_ID`

## Optional: Disable logging

* Set the value of `LOGGING` in the `.env` to False

## Commands

### Core Commands
* `/chat [message]` - Chat with the current AI provider
* `/provider` - Switch between configured AI providers
* `/draw [prompt] [model]` - Generate images with specified provider
* `/reset` - Clear conversation history
* `/help` - Display all available commands

### Persona Commands
* `/switchpersona [persona]` - Switch AI personality (admin-only for jailbreaks)
   * `standard` - Standard helpful assistant
   * `creative` - More creative and imaginative responses  
   * `technical` - Technical and precise responses
   * `casual` - Casual and friendly tone
   * `jailbreak-v1` - BYPASS mode (admin only)
   * `jailbreak-v2` - SAM mode (admin only)
   * `jailbreak-v3` - Developer Mode Plus (admin only)

### Bot Behavior
* `/private` - Bot replies only visible to command user
* `/public` - Bot replies visible to everyone (default)
* `/replyall` - Bot responds to all messages in channel (toggle)
## Security Features

### Admin-Only Jailbreak Access
Jailbreak personas require admin privileges for enhanced security:

1. Set `ADMIN_USER_IDS` in `.env` with comma-separated Discord user IDs
2. Only admin users can access jailbreak personas
3. Regular users see only safe personas in `/switchpersona`

> **Warning**
> Jailbreak personas may generate content that bypasses normal AI safety measures. Admin access required.

### Environment Security
- Secure API key management via environment variables
- Docker security hardening with non-root user
- Read-only filesystem for container security
