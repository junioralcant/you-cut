---
name: thumbnail-generator
description: Generates high-quality YouTube thumbnails based on a title and the user's personalized visual style. Proposes ideas and generates images via DALL-E 3.
---

# Thumbnail Generator

This skill takes a Title and creates 5 distinct thumbnail concepts, then generates the final image for the chosen concept.

## When to use this skill

- When the user selects a title from the `title-generator` or asks for a thumbnail.
- When the user uses the syntax `@[thumbnail-generator] "Title"`.

## How to use it

### 1. Conceptualize (The "Idea Phase")
**Goal**: Translate the Title into a Visual Hook.
- **Input**: Video Title (e.g., "Why I Stopped Using Zapier").
- **Action**: Generate 5 distinct visual concepts using the user's requested style.
- **Variables**:
    - **Face Expression**: (Confused, Happy, Serious, Shocked, Intense Focus).
    - **Text on Thumb**: Short, punchy (max 4 words). *Different from title!*
    - **Palette**: The user's specific color scheme:
        - Background: `[Background Color Preference]`
        - Core Accent 1: `[Primary Brand Color]`
        - Core Accent 2: `[Secondary Brand Color]`
        - Primary Text: `[Text Color]`
    - **Composition**:
        - **Split**: Text Left / Subject Right.
        - **Rule of Thirds**: Subject on vertical third, text balancing on the opposite side.
        - **Centered**: Subject center, text split top & bottom.
        - **Dynamic Diagonal**: Angled text or subject for a tech-forward "vibe".
    - **Graphics**: Logos (Zapier, n8n, OpenAI), 3D Glass-morphism icons, Glowing Arrows, Connection Nodes.

**Output Format**:
Presentation of the 5 concepts for User Selection:
1.  **The "V.S." Battle**: Split screen. Red "Zapier" vs Green "n8n". Dan in middle looking conflicted. Text: "Cheaper?".
2.  **The "Trash Can"**: Dan throwing "Zapier" logo in trash. Orange arrow pointing to "n8n". Text: "I Quit.".
3.  ...

### 2. User Selection
- Ask the user to pick one concept (e.g., "Option 2") OR suggest a custom idea.

### 3. Image Generation (The "Creation Phase")
**Goal**: Create the actual High-Res Thumbnail (16:9).

#### Logo Selection Logic
Before running the command, check the **Selected Concept** for any mentioned Apps/Tools (e.g., n8n, Claude, Antigravity).
1.  **Scan Folder**: Look into `.agent/skills/thumbnail-generator/resources/app-logos/` for matching PNG files.
2.  **Match Found**: If a match exists (e.g., `n8n-logo.png`), include it in the command. 
3.  **Multiple Logos**: You can include multiple logos if the concept requires it.
4.  **Reference in Prompt**: In the prompt, refer to logos by their index. `input[0]` is always the **Subject** (from `.agent/skills/thumbnail-generator/resources/subject.png`). `input[1]`, `input[2]`, etc., are the **Logos**.

- **Safe-Zone**: ALWAYS append the instruction: *"Keep all text and critical subjects in the vertical center safe-zone for 16:9 cropping."* to the end of the prompt. 

- **Action**: Use the helper script `generate_thumbnail.py`.
- **Command**:
    ```bash
    python3 .agent/skills/thumbnail-generator/scripts/generate_thumbnail.py "<PROMPT>" "<OUTPUT_FILENAME>" ".agent/skills/thumbnail-generator/resources/subject.png" ".agent/skills/thumbnail-generator/resources/app-logos/<LOGO_1>.png" ".agent/skills/thumbnail-generator/resources/app-logos/<LOGO_2>.png"
    ```

- **Process**:
    1.  The script uses `gpt-image-1.5` and passes all images as reference "source" images.
    2.  It automatically applies a 16:9 crop (1536x864) to the final output.
    3.  It saves the output to `thumbnails/<OUTPUT_FILENAME>`.

**Prompt Construction**:
- **Safe-Zone & Framing**: Since the script crops 80px from the top (about 8% of the height), you MUST leave exactly enough header space.
    - *Example Instruction*: "The top 12% of the vertical frame must be empty background space. The subject's head MUST start just below the 12% mark from the top. This ensures a tight but safe crop for 16:9."

*\"A premium YouTube thumbnail. [Composition Detail]. Background is a deep charcoal (#121212). Subject is input[0] (person) positioned so the top of the head is approximately **12% away from the top edge**. **CRITICAL**: The very top of the image (top 12%) must be empty background to survive the 80px crop. Incorporate input[1] (App Logo) as a 3D glass element. Lighting: Cinematic teal rim light. Colors: Charcoal, Teal, Orange. Text: \'[Short Text]\' in massive bold white. Keep all text and subjects within the **middle 75% vertical safe-zone** for a perfectly balanced 16:9 frame.\"*

### 4. Final Output
- Save the image to `thumbnails/thumb_<title_slug>.png`.
- Show the image to the user.
