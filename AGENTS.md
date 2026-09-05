# AGENTS.md – Rules for AI Coding Agents Working on Clow

**This file is mandatory reading before any work begins.**

You are an AI coding agent helping build **Clow**, a predictive quantitative trading terminal and AI foundation research platform.  
Your job is to implement the system correctly, safely, and incrementally across both the **Python AI Training Lab** and the **C++ Qt Desktop Terminal**.  
You do **not** own the project. The human owns the architecture, the quality bar, and the merge decisions.

---

## 1. Core Identity of the Project

**Clow** is a high-performance quantitative trading platform combining:
1. **Time-Series AI Models** (Predictive Candle Forecasters and Tactical Order Architects).
2. **High-Performance C++ Qt Desktop Terminal** (Hardware-accelerated charting, sub-millisecond execution, direct MT5 Windows IPC bridge without EAs).
3. **Sovereign Execution & Risk Governance** (Asymmetric triple-barrier expectancy, strict drawdowns, spread/slippage friction deduction, panic kill-switch).

The central trust mechanism is **Scientific Integrity & Zero Look-Ahead Leakage**.

If you ever feel tempted to optimize for an impressive backtest at the expense of correctness, stop. Correctness always wins.

---

## 2. Absolute Rules (Never Violate)

1. **Never work directly on `development`, `staging`, or `main`.**  
   You may only commit to the current feature/milestone branch (`feat/mXX-...`).

2. **Never implement more than the current milestone allows.**  
   Do not jump ahead to later milestones, UI, live trading, or packaging unless the current milestone contract explicitly includes it.

3. **Never invent fake functionality or fake performance numbers.**  
   If something is not implemented, return or raise a clear "Not implemented".  
   All metrics must come from real calculations on real data.

4. **Never allow look-ahead bias or data leakage.**  
   Features, normalization, labels, scalers, and chronological validation must be strictly causal.

5. **Never use the blind test set for any form of tuning, selection, or calibration.**

6. **Never enable live trading by default.**  
   Live mode must remain disabled until explicitly activated through the proper risk and confirmation gates.

7. **Never log or commit secrets** (broker credentials, API keys, private passwords).

8. **Never execute arbitrary unsafe code or untrusted model binaries.**

9. **Every exported `.onnx` model artifact must contain its full metadata, input/output tensors, and feature scaling schema.**

10. **The Desktop UI must never contain business logic that contradicts the quantitative engine.**

---

## 3. Monorepo Structure & Separation of Concerns

- **`training/` (Python / PyTorch / Transformers / ONNX)**:
  - Historical chunked data extraction and Parquet caching.
  - Fine-tuning foundation models (Chronos / TSFM) and training tactical order policy models.
  - Zero-leakage purged walk-forward validation and ONNX export.

- **`src/` (Pure C++ / Qt 6)**:
  - Hardware-accelerated desktop UI (Charts, Ghost-Candle rendering, Live Radar).
  - Direct MetaTrader 5 Windows IPC / DLL bridge (zero EAs).
  - ONNX Runtime C++ engine (<5ms CPU inference).
  - Sovereign Risk Manager and Pending Order state machine.

- **`models/` (Shared Artifacts)**:
  - Exported `.onnx` model weights and `model_metadata.json`.

---

## 4. How You Receive Work

- The human will give you **one milestone at a time**.
- You will receive:
  - The current milestone contract (from `PROGRESS.md`)
  - The current state of the codebase on the feature branch
- You must **not** request or assume the entire specification unless the human explicitly provides it.
- Completed milestones remain in `PROGRESS.md` so you accumulate context over time.  
  You still work **only** on the newest (in-progress) milestone unless the human explicitly tells you otherwise.

---

## 5. How You Must Work

- Create **one atomic commit per task** listed in the milestone.
- Write tests together with (or before) the implementation.
- Prefer clear, readable, modern C++ (C++20) and well-typed Python (3.11+).
- Follow the directory structure defined in the specification.
- When a task is finished, mark it done in `PROGRESS.md` (see Section 11) and stop to wait for human review if required.
- Do not open a pull request or merge anything yourself unless explicitly instructed.

---

## 6. Commit Message Rules (Mandatory)

When creating commits you must follow this exact format.  
Do not wait for the user to provide a diff — inspect the changes yourself with `git status` and `git diff`.

### Header Format

```
<type>(`<path>`): <short description with backticked filename>
```

- `<type>` must be one of: `feat`, `fix`, `refactor`, `style`, `chore`, `perf`, `test`, `docs`, `build`, `ci`
- `<path>` must be the full relative directory path wrapped in backticks
- The primary filename in the description must also be wrapped in backticks
- Keep the entire header under 100 characters
- Use present tense

### Body Format

Exactly two paragraphs, each written as a single unbroken line:

```
<header>

<first paragraph – what was done>

<second paragraph – why it was done>
```

- Blank line after the header
- Blank line between the two paragraphs
- No bullet points, no lists, no line wrapping inside paragraphs

### Workflow

1. Run `git status` and `git diff` (or `git diff --cached`)
2. Stage the relevant changes with `git add`
3. Create exactly one conventional commit following the format above
4. Repeat until the working tree is clean
5. **Never push**

Example:

```
feat(`/src/broker`): implement direct MT5 IPC connection in `mt5_bridge.cpp`

Implemented native Windows named pipe connection to MetaTrader 5 terminal without requiring Expert Advisors.

Enables low-latency account inspection and pending order execution directly from the C++ Qt terminal.
```

---

## 7. Pull Request Summary Rules (Mandatory)

When asked to generate a PR summary, produce a message that can be used as a squash commit.  
Fetch the commits yourself with `git log development..HEAD` (or equivalent).

### Exact Format

```
pr(`/`): <short description with backticked filename>

<first paragraph summarizing what was done>

<second paragraph explaining why it was done>
```

- Type is always `pr`
- Scope is always `` `/` ``
- Exactly two single-line paragraphs
- No bullet points or lists
- Synthesize the entire PR into one coherent summary

---

## 8. Definition of Done for Any Task

A task is only done when:
- The code is implemented
- Tests covering the happy path and important edge cases pass
- The change is committed with a message that strictly follows the rules in Section 6
- The corresponding checkbox in `PROGRESS.md` has been marked as done
- You have not introduced code that belongs to a future milestone

---

## 9. Where to Find the Current Work

All milestones, branch names, tasks, and human review checklists live in:

→ **`PROGRESS.md`**

Start there.  
Read the Global Working Rules, then locate the milestone marked **🔄 IN PROGRESS**. That is the only milestone you are allowed to work on unless the human says otherwise.

---

## 10. When You Are Unsure

If the specification is ambiguous, or if a requested change would violate any rule above:
- Stop
- Explain the conflict clearly
- Wait for human guidance

Do not guess on matters of research integrity, leakage, or live-trading safety.

---

## 11. How to Update PROGRESS.md (Mandatory)

You must keep `PROGRESS.md` accurate as you work. Follow this exact format.

### Marking a single task complete

Change:

```markdown
- [ ] Implement chunked historical data extractor
```

To:

```markdown
- [x] Implement chunked historical data extractor
```

Do this as soon as the task is finished and committed.  
Then commit the progress update itself:

```
chore(`/`): mark chunked extractor task complete in `PROGRESS.md`

Marked the chunked data extractor task as done in PROGRESS.md after implementation and unit tests passed.

Keeps the milestone tracker accurate for the human reviewer.
```

### Milestone status markers

- While work is ongoing:
```markdown
### M01 – Data Engine & MT5 Bridge 🔄 IN PROGRESS
```

- Ready for human review (only human marks complete):
```markdown
### M01 – Data Engine & MT5 Bridge ✅ COMPLETE
```

---

**End of AGENTS.md**  
Now open `PROGRESS.md` and begin only the milestone marked 🔄 IN PROGRESS.
