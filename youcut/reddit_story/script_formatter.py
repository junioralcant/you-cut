"""Claude formata o raw Reddit em script narrável long-form (~4500-5200 palavras)."""

from __future__ import annotations

import anthropic

from youcut.reddit_story.models import RedditStorySource


_FORMAT_PROMPT = """You are formatting a viral Reddit story into a 22-25 minute narration script for a YouTube long-form video targeting an English-speaking audience.

Input is the raw Reddit post (markdown). Your job:

1. **Open with a tight 25-second HOOK** — punchy first sentence, tease the most satisfying moment, then say "Today's story comes from r/<<SUBREDDIT>>, and it's one of the highest-voted of all time. Here's what happened." Then dive in.

2. **Clean the body**:
   - Remove all markdown (**bold**, [links], asterisks)
   - Remove TL;DR sections (the whole story replaces them)
   - Remove "Edit:" and "Update:" header lines but KEEP the content
   - Expand abbreviations on first use (HOA = "Homeowners Association", OP can stay if natural)
   - Convert internet-isms ("ngl", "tbh", "lmao") to plain English
   - Replace URLs/image links with "and I'll spare you the visuals" or similar
   - Keep first-person voice and original tone (don't sanitize the bite)

3. **Pace it for narration**:
   - Break long paragraphs into shorter narrated sentences
   - Add brief connectors ("Now here's where it gets good", "But they weren't done", "What they didn't realize was...") at section boundaries
   - Lean into the satisfying moments — don't rush past the payoff

4. **Close with a 15-second OUTRO**:
   - One sentence reflection on the karma/lesson
   - Then: "What would YOU have done in OP's position? Drop a comment, and if you enjoyed this story, hit that subscribe button for more from the Reddit vault. Thanks for listening."

5. **Target length**: <<TARGET_WORDS>> words (will narrate in ~22-25 min at speed 1.05).

Output ONLY the final narration script. No preamble, no headers, no labels, no markdown. Pure prose meant to be read aloud.

Raw Reddit post:
<<RAW>>"""


def format_script_with_claude(
    client: anthropic.Anthropic,
    source: RedditStorySource,
    *,
    model: str,
    target_words: int,
) -> str:
    prompt = (
        _FORMAT_PROMPT.replace("<<SUBREDDIT>>", source.subreddit)
        .replace("<<TARGET_WORDS>>", f"{target_words - 200}-{target_words + 200}")
        .replace("<<RAW>>", source.selftext)
    )
    msg = client.messages.create(
        model=model,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()
