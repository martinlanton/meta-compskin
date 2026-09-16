---
name: plan-feature
description: Plan a feature, fix, or refactor before writing any code — investigate the relevant code, clarify ambiguous or assumption-laden requirements with the user, draft spec.md/design.md/tasks.md under plan/<feature-slug>/, self-review the plan against a fixed checklist (goals, hard vs. soft constraints, alternatives considered, maintainability/performance impact, scope discipline), revise until the checklist is clean, then present the plan for approval. Use whenever a request calls for planning, designing, or scoping non-trivial work before implementation begins, or when the user invokes /plan-feature explicitly.
---

# /plan-feature

Plan a feature or fix before writing code. The point of this skill is to
slow down: investigate before assuming, ask before guessing, and interrogate
the plan itself before handing it over. Do not start implementing — the
output of this skill is a reviewed plan under `plan/<feature-slug>/`, not
code.

Work through the six steps below in order. Do not skip the self-review step
even when the plan feels obviously right — its job is to catch what
"obviously right" hides.

## 1. Investigate the issue and the code

- Read the request closely enough to restate it in your own words before
  touching code. If you can't restate it, you don't understand it yet —
  reread it or ask (step 2).
- Find and read the code this actually touches: the relevant modules,
  their callers, their tests, and any docs or paper sections
  (`paper/compressed_skinning_for_facial_blendshapes.md`) they implement.
  Note concrete file:line references as you go — the plan will need them.
- Check history: `git log -p` / `git blame` on the affected files, and any
  prior plans under `plan/` that touch the same area. Prior art (including
  prior *rejected* approaches) changes what's worth proposing now.
- Verify assumptions against the code, not against memory of the codebase or
  the paper. If a doc or comment claims something, open the code it
  describes and confirm it still matches.

## 2. Clarify assumptions with the user

Do not silently resolve ambiguity by picking the option that seems most
likely — ask. Bring a question to the user when:

- the request is ambiguous enough that two reasonable readings would produce
  materially different plans;
- a requirement's priority is unclear (see the hard vs. soft distinction in
  step 4) and the choice changes the design;
- you're about to assume a constraint, target platform, performance budget,
  or scope boundary that isn't stated anywhere in the request or the repo;
- an existing codebase pattern conflicts with what would otherwise be the
  better solution, and which one wins isn't obvious.

Batch questions into one round when possible. Don't ask about things you can
verify yourself by reading the code (verify, don't ask).

## 3. Draft the plan

Write the plan as a set of documents under `plan/<feature-slug>/`, following
the structure already used in this repo (see e.g.
`plan/gradual_sparsity_annealing/` for a worked example):

- **`spec.md`** — the *what* and *why*. Summary, background/motivation,
  current behavior with file:line references, the proposed change, scope
  (explicitly list what's out of scope), and open questions if any remain.
- **`design.md`** — the *how*. Architecture/data flow (a diagram earns its
  keep here), the concrete API/data model changes, file-by-file breakdown of
  what changes.
- **`tasks.md`** — an ordered, checkbox checklist sized for one commit per
  task, each with a concrete "done when" condition. Follow this repo's TDD
  law: test first, see it fail, minimal code, see it pass, refactor. Include
  a "before you start" section (env setup, which fast test loop to use,
  which existing code/paper sections to read) and a pre-PR checklist
  (`/pre-commit-checks`, full test suite).
- **`research.md`** (only when the task is mathematical, research-adjacent,
  or otherwise needs a documented "why"; skip it for plumbing/refactor
  work) — the underlying problem, relevant paper sections/equations, and any
  papers or techniques considered and rejected, with reasons.

Cross-link the documents (each should link the others at the top). State a
revision number and the commit the plan is written against, so later
revisions can say what changed and why.

## 4. Self-review

Before showing the plan to anyone, review it against every question below.
Write the answers down in the plan (a `## Self-review` section in `spec.md`
is a reasonable place) — a self-review that isn't visible didn't happen.

1. What are the goals and objectives? State them explicitly; if the plan
   doesn't map cleanly onto them, that's a gap.
2. What are the constraints and limitations? For each one, is it a **hard**
   requirement (violating it makes the solution wrong or unacceptable) or a
   **soft** one (a preference that could be traded off)? Conflating the two
   is a common source of over- or under-engineered plans.
3. Is this the best solution? Are there solutions you haven't considered
   yet? If you only evaluated one approach, that's not a review — go find
   at least one real alternative and say why it lost.
4. For mathematical or research-related tasks: are there papers or
   techniques you haven't considered? Skimming the references section of a
   paper you already have open counts; not looking at all doesn't.
5. Does this make the code more maintainable, extensible, testable, and
   reliable? Answer each separately — a plan can improve one and hurt
   another.
6. Does this improve performance and scalability? If it's neutral or worse,
   say so and say why that's still acceptable.
7. Overall, does this improve quality? This is the roll-up of 5 and 6, not a
   restatement — it should be answerable in one sentence once those are
   done.
8. Where the plan follows an existing codebase guideline or pattern, is it
   because that standard is actually good, or just because "it's always
   been done like this here"? Cargo-culting a bad existing pattern into new
   code is a defect, not consistency.
9. Scope discipline: legacy code *adjacent* to the area being touched may be
   improved freely as part of this plan. Code *outside* that target is left
   alone unless there's a concrete, stated reason to touch it — "while I'm
   here" is not a reason. Check the task list for tasks that drift outside
   the stated scope and either justify them explicitly or cut them.

## 5. Revise the plan

Any question in step 4 that surfaced a gap, an unjustified assumption, a
missed alternative, or scope creep means the plan is not done. Fix the
plan's documents directly (bump the revision number, note what changed and
why), then re-run the specific self-review questions that were affected —
not the whole list from scratch, but enough to confirm the fix didn't open a
new gap. Repeat until nothing on the checklist is unresolved.

If a gap can only be closed by asking the user (e.g. it turns on a
hard-vs-soft call only they can make), go back to step 2 rather than
guessing to unblock yourself.

## 6. Present the plan

Hand the plan back to the user with:

- the file paths under `plan/<feature-slug>/` that were written;
- a short summary (a few sentences, not a re-read of spec.md) of what's
  being proposed and why;
- anything from the self-review worth flagging explicitly — a real
  trade-off made, a hard requirement that constrained the design, or scope
  intentionally left out;
- the next step: usually "review `tasks.md` and say go" — do not start
  implementing without that confirmation.
