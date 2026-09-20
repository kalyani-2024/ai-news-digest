# AI News Aggregator

A daily digest pipeline that follows AI news for me so I don't have to. It scrapes OpenAI, Anthropic, and YouTube, summarizes each item with an LLM, ranks everything against a personal interest profile, and emails the top stories as a formatted digest.

## How it works

The pipeline runs as five sequential stages, orchestrated by `run_daily_pipeline()`:

```
Scrape  ->  Extract  ->  Summarize  ->  Rank  ->  Email
```

1. **Scrape** — pulls the last N hours from three sources: the OpenAI and Anthropic news feeds, and any YouTube channels listed in config. New items are written to Postgres; duplicates are skipped by primary key.
2. **Extract** — converts article pages to markdown with Docling, and fetches transcripts for YouTube videos.
3. **Summarize** — a digest agent (Gemini Flash Lite) turns each item into a title and a 2–3 sentence summary.
4. **Rank** — a curator agent (Gemini Flash) scores every digest 0–10 against the user profile: interests, expertise level, and preferences such as favoring technical depth over marketing copy.
5. **Email** — an email agent writes an intro, and the top N articles go out as HTML over SMTP.

## Project layout

| Path | What's in it |
| --- | --- |
| `app/scrapers/` | Source-specific scrapers: OpenAI, Anthropic, YouTube |
| `app/services/` | One module per pipeline stage |
| `app/agent/` | Gemini client and the three agents: digest, curator, email |
| `app/database/` | SQLAlchemy models, repository, connection |
| `app/profiles/` | The interest profile driving ranking |
| `app/daily_runner.py` | Stage orchestration and logging |
| `run_pipeline.py` | Deployment entrypoint (creates tables, then runs) |

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp example.env .env
```

Fill in `.env`:

| Variable | Notes |
| --- | --- |
| `GEMINI_API_KEY` | Required. Get one at [aistudio.google.com](https://aistudio.google.com/apikey) |
| `DIGEST_MODEL` / `CURATOR_MODEL` / `EMAIL_MODEL` | Optional overrides. See Models below |
| `MY_EMAIL` / `APP_PASSWORD` | Gmail address and an [app password](https://support.google.com/accounts/answer/185833) — not your account password |
| `DIGEST_RECIPIENTS` | Comma-separated. Defaults to `MY_EMAIL` |
| `DATABASE_URL` | Full connection string. Leave blank locally to use the `POSTGRES_*` parts |
| `USER_NAME`, `USER_TITLE`, `USER_BACKGROUND` | Shape how the curator ranks |
| `YOUTUBE_CHANNELS` | Comma-separated channel IDs |

Start Postgres and create the schema:

```bash
docker compose -f docker/docker-compose.yml up -d
uv run python -m app.database.create_tables
```

## Running it

```bash
uv run python main.py            # last 24 hours, top 10 articles
uv run python main.py 48 15      # last 48 hours, top 15
```

Individual stages run standalone, which is useful when debugging one part:

```bash
uv run python -m app.runner                  # scrape only
uv run python -m app.services.process_digest # summarize only
uv run python -m app.services.process_email  # rank and send only
```

## Deploying to Render

The pipeline is a scheduled batch job, not a web service, so it deploys as a Render **Cron Job**. The database lives outside Render (Neon, Supabase, or any managed Postgres). The job is declared in `render.yaml`.

1. Create a Postgres database with your provider and copy its connection string.
2. Push this repo to GitHub.
3. In Render, choose **New > Blueprint** and point it at the repo. It reads `render.yaml` and creates the cron job.
4. Set the secrets marked `sync: false` in the Render dashboard: `DATABASE_URL` (the connection string from step 1), `GEMINI_API_KEY`, `MY_EMAIL`, `APP_PASSWORD`, `DIGEST_RECIPIENTS`, `USER_NAME`, `YOUTUBE_CHANNELS`.
5. Trigger a manual run from the dashboard to verify, then let the schedule take over.

The default schedule is `0 6 * * *` — 06:00 UTC daily. Render cron schedules are always UTC, so adjust the hour for your timezone. `run_pipeline.py` calls `create_tables()` on every run, so the first run provisions the schema itself.

Notes on the deployment shape:

- **Docker, not a native Python runtime.** Docling pulls in torch, transformers, and OpenCV, which need system libraries (`libgl1`, `libglib2.0-0`) that the Dockerfile installs.
- **Cron job, not a web service.** There is no HTTP surface, and a full run takes minutes — well past the request timeouts that serverless platforms impose. This is also why it won't deploy to Vercel.
- **External database.** Keeping Postgres off Render avoids the free plan's expiry and keeps the data portable. Any provider works as long as `DATABASE_URL` is reachable and allows SSL connections.

## Models

Runs on the Gemini API. Each agent's model is set by an environment variable, so they can be swapped without touching code:

| Variable | Default | Why |
| --- | --- | --- |
| `DIGEST_MODEL` | `gemini-3.5-flash-lite` | One call per article, so this drives volume. Cheapest tier is the right fit |
| `CURATOR_MODEL` | `gemini-3.5-flash` | Ranking is the reasoning-heavy step, but runs once per pipeline |
| `EMAIL_MODEL` | `gemini-3.5-flash` | Writes the intro, once per pipeline |

Google retires model names faster than most providers, and a stale name fails at runtime with a `404 ... no longer available`. To see what a key can currently reach:

```bash
uv run python -c "from app.agent.client import get_client; [print(m.name) for m in get_client().models.list()]"
```

Note that the list endpoint advertises models that `generateContent` will still refuse, so confirm with a real call before relying on one. The Pro tier models return `429 RESOURCE_EXHAUSTED` on a free API key — stick to Flash unless the key is on a paid plan.

## Cost

One Flash Lite call per article plus two Flash calls per run. Comfortably inside the Gemini free tier for a handful of sources a day.
