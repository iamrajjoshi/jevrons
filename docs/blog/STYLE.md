# Style guide for the Jevrons post

Distilled from three sources: the Karpathy article-writing skill (IrtezaAsadRizvi/article-writing-skills), the Evergreen `eng-blog-post` skill, and Raj's published posts in `~/code/blog/src/content/blog/`. Raj's posts and his writing rules set the voice. Where the other two disagree with him, he wins; the conflicts are listed at the end.

## The one rule

Write about something built and measured. Every section should be anchored to a run, a table, a figure or a code path in this repo. If a paragraph has no number, name, file or result in it, it's probably filler.

## Voice

First person singular for what Raj did ("I sent", "I folded the bias in"). "We" only when walking the reader through a calculation. No passive voice that hides who did what.

Casual and direct, the way the Willow post reads: it opens with why he built the thing, admits which parts were tedious, and says "it's pretty alpha" instead of hedging for a paragraph. Contractions everywhere. Short honest reactions are fine ("I didn't expect that", "this one was frustrating") but keep them rare and earned.

Assume the reader knows what a neural network, a gradient and a sigmoid are. Explain Jev-specific things (Noul, state, questions) once, plainly, then use them.

Hedge only where the data is actually thin (one seed, not converged), and say exactly why it's thin. Don't hedge defensively.

## Numbers

Precision over adjectives. "Dropped from 94.2% to 67.8%" beats "dropped sharply". Give before/after pairs and the n behind them (500 validation images, 3 seeds, 60 neurons per cell).

Every number must trace to `docs/results/*.md`, `docs/PLAN.md`, `docs/proposal.md` or `runs/*/summary*.json` / `result-*.json`. No rounding that changes the claim. No invented wall-clock or cost figures.

Use tables for comparisons with more than two cells (stage 1 accuracy by term count, stage 5 arms). Keep tables small; cut columns the prose doesn't use.

## Structure

Opening: one or two short paragraphs. Start from the concrete precursor (the NAND-gate ALU) and the question, not a statement about AI in general. Offer the repo or results early if they're public (Willow does "If you can't wait, here is the code"); Jevrons is private, so skip that line.

Middle: `##` headings that name the thing, not tease it ("The neuron over-trusts the bias", not "A surprising twist"). Paragraphs of 2-5 sentences with varied length. One idea per section. Results embedded in the prose where they're discussed, not saved for the end.

Code: fenced, with a language tag, short enough to read in one glance (under ~15 lines), and immediately followed by a sentence saying what it does or why it's shaped that way. Willow's pattern: show the struct, then one sentence on the consequence. Show real state/JSON from the journals where possible, trimmed with `...`.

Figures: markdown images with descriptive alt text (the Sentry post writes full-sentence alt text). Put the figure next to the paragraph that interprets it and say what to look at.

Footnotes (`[^1]`) for side facts and references, as in Willow and the Sentry post. Keep them to one or two sentences.

Closing: no recap, no moral. End on the next concrete thing (the demo, stage 7) or a practical pointer. The Willow ending ("If you want to try it:" then a command) is the model.

## Frontmatter

Match the blog's Astro schema exactly: `title` (10-60 chars), `description` (max 160), `pubDate`, `updatedDate`, optional `hero`/`heroAlt`. There's no `draft` field in the schema, so drafts are marked by staying out of `src/content/blog/`.

## What to avoid

Raj's banned words: delve, dive into, navigate (figurative), underscore, bolster, foster, harness, leverage, unpack, shed light on, pave the way, pivotal, groundbreaking, cutting-edge, transformative, game-changing, innovative, robust, comprehensive, seamless, intricate, nuanced (as praise), vibrant, multifaceted, holistic, testament, landscape (figurative), realm.

Banned structures: "It's not just X, it's Y", "Not only X but Y", "No X. No Y. Just Z.", "This isn't about X. It's about Y."

Banned phrases: "It's worth noting", "When it comes to", "At its core", "Let's break it down", "This is where X comes in", "plays a crucial role", sweeping openers about the state of the world.

Also avoid: "Bold term: explanation" bullet lists (the "key factors" list in the Cursor post is the pattern not to repeat); signposting ("Let's look at", "Here's where it gets interesting"); marketing cadence; performative enthusiasm ("exciting", "amazing"); a closing summary or inspirational wrap-up; more than one em dash in the whole post.

## Conflicts and how they were resolved

1. Karpathy's guide encourages em-dashed asides as texture. Raj's rule is at most one em dash per piece. Use parentheses and commas instead.
2. Karpathy's guide suggests bold topic sentences in the middle sections. Raj bans the "Bold term: explanation" pattern, and his newer posts use headings instead. Use `##` headings; no bolded lead-ins. (Willow uses a few bold lead-ins under "Things I learned"; don't copy that here.)
3. Karpathy's guide allows emoticons like ¯\\\_(ツ)\_/¯. Raj's posts never use them. Leave them out.
4. Karpathy likes cross-domain analogies and time-travel framing. Allowed once, briefly, if it explains something (the straight-through estimator as "pretend the step was a sigmoid"). Don't build sections around them.
5. Raj's older posts (Sentry cross-posts) use words and structures now on his banned list ("underscores", "navigate", "seamless", "isn't just a nice-to-have, it's a must-have", an inspirational sign-off). His current rules override his older posts.
6. The Evergreen skill is written for the Zip company blog (house voice, Notion/Webflow output, author interview). Only its writing rules carry over: no banned words, no AI tells, no soft em dashes, an outline before drafting, and a self-review pass at the end. Its company voice and publishing workflow don't apply to a personal post. (Only the skill's header was readable in this session; its `references/` files weren't read, so nothing here comes from them.)
