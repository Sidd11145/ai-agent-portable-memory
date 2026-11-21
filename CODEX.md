# Codex compatibility

Codex support is provided by the additional `memory codex` commands. Existing
Claude Code hooks, commands, compression providers and storage backends keep
their existing behavior. Codex memories appear in the same `memory list` and
`memory search` results, with separate IDs for each Codex source session.

If the package is installed, use `memory codex ...`. From this source checkout,
use `python3 -m memory codex ...` instead. The examples below use the source form.

## Automatic saving and loading with local Codex

Run once from this checkout to install hooks for your Codex user:

```bash
python3 -m memory codex setup
```

To install only for this repository instead:

```bash
python3 -m memory codex setup --project-local
```

Choose one scope. Installing both can cause both hooks to run. Existing hook
definitions are preserved, and repeating setup in one scope is idempotent.
The generated command uses the current Python interpreter and source path,
so it also works without installing a `memory` executable. Re-run setup if you
move the checkout or change the Python environment.

Restart Codex, open `/hooks`, and review and trust the new memory hooks. Codex
requires trust before non-managed hooks run; setup does not bypass that review.
On versions where hooks are disabled, enable the supported `hooks` feature.
This integration was checked against the rollout format in Codex CLI 0.170.0.

The added hooks:

| Event | Behavior |
| --- | --- |
| `SessionStart` | Load this session's saved memory, or the latest memory for the project. Also runs after compaction when Codex emits a compact session start. |
| `Stop` | Save after a completed agent turn. |
| `PostCompact` | Save a checkpoint after context compaction. |
| `SessionEnd` | Save the final available transcript when Codex ends the session normally. |

Automatic saves use the existing `MEMORY_LLM_PROVIDER` and its API key or local
provider. Codex sign-in alone does not supply a compression API key. No API key
is needed to load memories or save a summary that Codex has already written.
Unchanged conversations skip repeated compression. Saves for one source session
are serialized and update that session's memory. Codex saves do not change
Claude's active-session pointer. `memory stop` and `memory start` also apply.

The project label defaults to the absolute workspace path, preventing two
projects with the same folder name from sharing context accidentally. Set
`MEMORY_CODEX_PROJECT` in the environment used to launch Codex to use a shared
label, including a label already used by Claude Code. Existing memories can
also be selected explicitly with `memory codex load --session ID`.

Hook failures are reported to stderr and do not block or continue Codex's turn.
If Codex provides no local transcript path, the hook skips the save. Session-end
hooks cannot cover a forced termination or system crash; turn checkpoints
preserve the last completed save.

Remove just this integration's hooks with the matching scope:

```bash
python3 -m memory codex setup --remove
python3 -m memory codex setup --project-local --remove
```

## Manual use

Run from the workspace whose session you want to save:

```bash
memory codex save --no-edit
memory codex save --source-session EXACT_CODEX_SESSION_ID --no-edit
memory codex save --transcript /path/to/rollout.jsonl --no-edit
memory codex save --provider ollama --model llama3.2 --no-edit
memory codex load
memory codex load --session SAVED_MEMORY_ID
```

Session discovery honors `CODEX_HOME`, searches active and archived rollout files,
and filters by the current workspace unless you select an exact source session.
The importer reads user and assistant message text and skips duplicate event
records, developer/system messages, reasoning and tool output. Codex's transcript
format is not a stable hook interface; future formats may require an importer
update. Compaction saves what remains available in the rollout, which can include
the previous conversation as well as subsequent messages.

## Saving directly from Codex without another API call

Have Codex write a structured summary to a Markdown file, then run:

```bash
memory codex checkpoint --summary-file memory-summary.md
memory codex load
```

Include `## Title` followed by a title, and optionally `## Project` and `## Tags`.
Use the existing summary format for completed work, decisions, current state and
next steps. Direct checkpoints use a separate workspace memory ID by default;
`--source-session` can target a specific Codex source session. They run the existing
secret scrubber and store the summary without calling a compression model.

For a Codex environment that cannot execute local hooks, append guidance like
this to the project's existing `AGENTS.md`, adjusting the command to your install:

```markdown
At task start, run `memory codex load` and use the result as historical project
context. Verify it against current files and user instructions.
At meaningful checkpoints and before finishing, write a structured summary to
memory-summary.md, then run `memory codex checkpoint --summary-file memory-summary.md`.
Include completed work, decisions, current state, and next steps, and avoid secrets.
```

This is an agent workflow and requires the memory package and storage to be
accessible from that environment. It does not depend on lifecycle hooks.
Command/local-config hooks are unsupported with Codex cloud orchestration;
managed policies can also restrict user hooks. Setup installs local hook
definitions, without changing those policies or existing AGENTS.md files.

Official references: [Codex hooks](https://developers.openai.com/codex/hooks),
[configuration](https://developers.openai.com/codex/config-reference/), and
[AGENTS.md](https://developers.openai.com/codex/guides/agents-md/).
