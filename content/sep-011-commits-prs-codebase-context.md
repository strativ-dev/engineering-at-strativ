Title: SEP-011: Improving Commits, PR Descriptions & Codebase Context in Git
Date: 2026-04-30
Category: Engineering Process
Slug: sep-011-commits-prs-codebase-context
Authors: Saqibur Rahman
Tags: git, commits, pull-requests, conventional-commits, engineering-process
Summary: SEP-011 sets the standard for commit messages, branch names, PR titles, and PR descriptions at Strativ. We adopt Conventional Commits and explain why good Git writing matters for every tool and person that reads our codebase.

| Field | Value |
|---|---|
| **SEP** | 011 |
| **Title** | Improving Commits, PR Descriptions & Codebase Context in Git |
| **Author** | Saqibur Rahman |
| **Status** | Draft |
| **Type** | Process |
| **Created** | 2026-04-30 |
| **Target Adoption** | 2026-Q2 |
| **Requires** | SEP-001 |
| **Related** | SEP-004, SEP-005, SEP-010 |

---

## Abstract

This SEP sets the standard for what Strativ developers write in Git: commit messages, branch names, PR titles, and PR descriptions. We adopt Conventional Commits as the required format, define what a useful PR description looks like, and explain why the quality of these artefacts now matters beyond just good hygiene.

AI assistance is not required. It is permitted and encouraged for developers who find writing commits tedious, but the Pilot is always responsible for what gets written.

---

## 1. Motivation

Every commit message and PR description we write becomes part of the permanent record. That record gets read by:

- The developer who wrote it, six months later, when something breaks.
- A new joiner trying to understand why code looks the way it does.
- A reviewer trying to understand what changed and why.
- A client asking what was delivered in the last sprint.
- **Strativ Brain** - our internal context engine - which indexes commit history, PR descriptions, and code to answer questions across projects.
- Every AI Copilot that works in the repository, which can only go on what was written down.

The current state is uneven. Some repositories have clear, structured histories that read like documentation. Others have commits that say "fix" or "wip" and PR descriptions that contain only the branch name. This SEP exists because that gap is now costly in ways it was not before.

### 1.1 The cost of a meaningless commit

Consider two commits touching the same authentication module:

| ✓ GOOD | ✗ BAD |
|---|---|
| `feat(auth): add JWT refresh token rotation` | `fix` |
| Replaces single long-lived JWT with rotating refresh tokens. Refresh tokens are single-use; reuse triggers session revocation across all devices. Closes BMS-247. | *(no body)* |

Both change the same number of lines. Both pass tests. Both ship. But only the first one is useful to anyone who reads the repository later. A Copilot asked to summarise recent auth work can answer from the first commit and nothing from the second.

### 1.2 The brain only knows what we write down

Strativ Brain is not a magic source of truth. It is a search layer over what we have already written. Its output quality is limited by its input quality. A brain trained on a repository where 40% of commits say "fix" can only tell you that 40% of recent work involved fixing something.

This is not something we can fix later by improving the brain. It is fixed earlier, in the repository, by writing better commits and PR descriptions.

---

## 2. Specification

### 2.1 Commit message standard

Strativ adopts the **Conventional Commits specification, version 1.0.0** as the required format for all commit messages in repositories under active development.

The structure is:

```
<type>(<scope>): <short description>

<optional body>

<optional footer(s)>
```

#### 2.1.1 Required types

Every commit must begin with one of the following types. The list is fixed - "misc," "update," and similar are not valid types.

| Type | Use for |
|---|---|
| `feat` | A new feature visible to a user, API consumer, or another developer. |
| `fix` | A bug fix. Reference the bug or ticket in the body or footer. |
| `refactor` | Changes that neither add features nor fix bugs. Internal restructuring. |
| `perf` | Changes that improve performance. State the measured improvement in the body. |
| `test` | Adding or updating tests. No production code changes. |
| `docs` | Documentation only - README, ADRs, code comments, runbooks. |
| `build` | Changes to build system, dependencies, or packaging. |
| `ci` | Changes to CI/CD pipelines, GitHub Actions, or deployment scripts. |
| `chore` | Maintenance tasks that don't fit the above. Use sparingly. |

#### 2.1.2 Scope

Scope is optional but strongly preferred. It names the area of the codebase being changed, typically a module, package, app, or domain. Each repository should document its scope vocabulary in its `CONTRIBUTING.md` or `CLAUDE.md`.

Common Strativ scope examples:

- `auth`, `billing`, `consumption`, `agreement`, `placement` (domain modules)
- `api`, `web`, `worker`, `deploy` (technical layers)
- `BMS`, `IMS`, `LumberScan` (project codes for cross-cutting work)

#### 2.1.3 Description

The description follows the colon. It must:

- Use the **imperative mood** - "add", not "added" or "adds". Think of it as completing: "If applied, this commit will..."
- Start with a lowercase letter.
- Not end with a period.
- Stay under 72 characters total (type + scope + colon + description).
- Describe **what the change does**, not what files were touched.

#### 2.1.4 Body

The body is where the **why** lives. It is optional for simple commits but required for anything that is not obvious from the diff. The body should cover:

- Why the change is being made (what problem it solves).
- What approach was chosen and what alternatives were considered, if relevant.
- Side effects, migration steps, or follow-up work the reader should know about.

Wrap body lines at 72 characters. Separate paragraphs with a blank line.

#### 2.1.5 Breaking changes

A breaking change is indicated by either:

- An exclamation mark after the type/scope: `feat(auth)!: replace token format`.
- A footer line starting with `BREAKING CHANGE:` that explains what breaks and how to migrate.

Both are acceptable. The footer form is preferred when the migration steps are non-trivial.

#### 2.1.6 Footers

Footers carry structured metadata that tools can parse:

```
Closes BMS-247
Refs IMS-031, IMS-032
Reviewed-by: Reaz
Co-authored-by: Contributor <contributor@example.com>
BREAKING CHANGE: token storage moved from cookie to header
```

### 2.2 Branch naming

Branch names are short-lived but show up in PR titles, deployments, and history searches. Follow this pattern:

```
<type>/<scope>-<short-slug>

feat/auth-jwt-refresh-rotation
fix/consumption-monthly-range-off-by-one
refactor/placement-selector
chore/upgrade-django-5
```

Avoid branch names that are only a ticket number (`BMS-247`), only a person's name (`saqibur-stuff`), or fully unstructured (`new-auth-flow-v2-final`).

### 2.3 Pull request titles

PR titles follow the same Conventional Commits format as commit messages. When a PR has a single commit, the PR title and commit subject should be identical. When the PR is squash-merged, the PR title becomes the commit subject and must follow §2.1.

Avoid PR titles that:

- Restate the branch name as a sentence ("Add JWT refresh token rotation work").
- Only reference a ticket ("BMS-247").
- Use "Update X" or "Fix X" without saying what.

### 2.4 Pull request descriptions

PR descriptions are where context that does not belong in commits goes: screenshots, deployment notes, testing steps, reviewer guidance. Every PR uses this template.

**PR descriptions should be generated by a Copilot and reviewed by the Pilot before submitting.** The Copilot drafts from the commit history and diff; the Pilot checks it for accuracy, fills in anything the AI could not know, and signs off. This is the expected workflow, not an optional shortcut.

```markdown
## What
One paragraph: what this PR changes. Written for a reviewer
who hasn't seen the ticket.

## Why
One paragraph: the problem this solves or the goal it advances.
Link to the ticket, ADR, or upstream discussion.

## How
Bullet list of the technical approach. Note any non-obvious
choices, deviations from convention, or trade-offs accepted.

## Verification
How to confirm this works. Test commands, manual steps,
URLs to staging, screenshots, before/after metrics.

## Risks & Follow-up
Anything the reviewer or future maintainer should know.
Known limitations, deferred work, monitoring to watch.

---
Pilot: <name>
AI involvement: none / drafting / extensive
```

Trivial PRs (typo fixes, dependency bumps) may collapse to a single sentence under **What**. The five-section structure is the default; any deviation should be intentional.

### 2.5 Forbidden patterns

The following are not acceptable in any repository under active development. PRs containing them should be revised before merge.

| ✓ GOOD | ✗ BAD |
|---|---|
| `fix(consumption): correct off-by-one in monthly range split` | `fix` |
| `feat(auth): add Microsoft Entra OIDC login flow` | `wip` |
| `refactor(placement): switch site selector to placement-based` | `updates` |
| `docs(adr): record decision on Knox vs JWT for session tokens` | `asdf` |
| | `merge branch 'feature/...'` |
| | `fix bug` |
| | `review changes` |

### 2.6 The role of AI

Per SEP-001, Pilots may use Copilots for any drafting task as long as the Pilot is accountable for the output. Commit messages and PR descriptions are explicitly included.

A Copilot can read a diff and produce a solid first-draft commit message in seconds. Pilots who find writing commits tedious are encouraged to use AI for the first draft and then edit. Those who prefer to write their own can do that too. Either is fine; neither is not an option.

> **PILOT ACCOUNTABILITY**
>
> If the Copilot's draft is wrong, misleading, or generic, the Pilot edits or rewrites it before committing. Shipping an unread AI-generated commit message is the same failure as shipping an unread AI-generated function. The Pilot's name is on the commit either way.

**What a Copilot is good at:**

- Writing the first-draft body when you have the diff but don't want to write prose.
- Suggesting the right Conventional Commits type based on the actual change.
- Drafting PR descriptions from a series of commits across the What/Why/How/Verification sections.
- Flagging commits about to ship as "fix" and prompting for a better description.

**Where the Pilot must add value:**

- Knowing *why* a change was made when that reason is not in the diff.
- Choosing the right scope when scope vocabulary is project-specific.
- Spotting breaking changes that are semantic, not just structural.
- Writing the Risks & Follow-up section honestly - that requires knowing the surrounding system.

---

## 3. Rationale

Three reasons drive this SEP, in increasing order of importance.

**First, reviewer experience.** A reviewer who can read a PR's What/Why/How in thirty seconds can form an opinion and move on. A reviewer who has to reverse-engineer intent from the diff is slower, more likely to miss things, and over time, more reluctant to review at all. Thirty seconds of writing pays back many times over.

**Second, your future self.** Every developer eventually opens `git blame` on code they wrote two years ago and wonders why. A commit that says "fix" is useless. A commit that says `fix(consumption): correct off-by-one in monthly range split, caused first day of February to be excluded from billing` is a gift.

**Third - and this is the new factor in 2026 - readability for context engines.** Strativ Brain, AI Copilots, and any future tooling that reasons over our codebase all depend on what we write. They cannot improve quality from a corpus of "fix" commits. This is already visible today: repositories with good commit hygiene are the ones where Brain gives useful answers. The connection is direct.

We chose Conventional Commits specifically because it is widely adopted, well-tooled (commitlint, semantic-release, changelog generation), and familiar to developers joining from other teams. A global standard means no extra onboarding overhead.

---

## 4. Backwards Compatibility

We are not rewriting existing history. Commits before the adoption date stay as they are - retroactively rewriting Git history would break references, links, and tooling for little gain.

From the adoption date forward:

- New commits in repositories under active development must follow §2.1.
- New PRs must use the §2.4 template (trivial changes may simplify it).
- Repositories should add a `CONTRIBUTING.md` (or extend their `CLAUDE.md`) listing their scope vocabulary.
- Archived or read-only repositories are exempt.

Enforcement starts as advisory. After one quarter, repositories may opt in to commitlint or similar CI checks that block non-conforming commits. Universal enforcement is a separate decision made after the team has used the standard for a while.

---

## 5. Reference Implementation

### 5.1 commitlint configuration

Repositories that want automated enforcement should use commitlint with the conventional-commits preset:

```js
// commitlint.config.js
module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [2, 'always', [
      'feat', 'fix', 'refactor', 'perf', 'test',
      'docs', 'build', 'ci', 'chore'
    ]],
    'subject-case': [2, 'always', 'lower-case'],
    'subject-max-length': [2, 'always', 72],
    'body-max-line-length': [2, 'always', 100],
  },
};
```

### 5.2 PR template

Add `.github/PULL_REQUEST_TEMPLATE.md` containing the §2.4 structure as a blank form. GitHub will show it automatically when a PR is opened.

### 5.3 Reference repositories

Good examples of this standard in practice include the Boo Energi backend (recent BMS-2xx work), the IMS maintenance branch, and the Strativ skill registry. New repositories can copy their `CONTRIBUTING.md` and PR template directly.

---

## 6. Open Issues

- Should commitlint enforcement be opt-in indefinitely, or required after a transition window?
- How do we handle repositories shared with clients who have their own conventions? *(Provisional answer: client conventions win in client-owned repos; Strativ conventions apply in Strativ-owned repos.)*
- Squash-merge vs. merge-commit policy is currently per-repository. Should it be standardised?
- How do we share scope vocabulary with a Copilot working in a repository without making it brittle?
- Co-authorship trailers when a Copilot drafted the commit - currently not required. Revisit if external tooling expects them.

---

## Appendix A - The Strativ Brain Connection

Strativ Brain is our internal context engine. It indexes commits, PR descriptions, ADRs, internal documents, and code to answer questions across projects. Both humans ("what was the last thing we did on auth?") and AI tooling (Copilots needing cross-project context) use it.

Brain is a retrieval and synthesis layer. It does not invent context. Its output quality is capped by its input quality, and those inputs are mostly Git artefacts.

### A.1 What Brain can do with a good commit

Given a commit like:

```
feat(auth): add JWT refresh token rotation

Replaces single long-lived JWT with rotating refresh tokens.
Refresh tokens are single-use; reuse triggers session
revocation across all devices.

Closes BMS-247.
```

Brain can correctly answer:

- "What changed in auth recently?" - surfaces this commit with a clear summary.
- "Why did we add session revocation?" - points to this commit and the ticket.
- "Show me feature work on Boo Energi this month" - correctly categorises this as `feat`.
- "Did we change how refresh tokens work?" - direct match on the body text.

### A.2 What Brain cannot do with a bad commit

Given the same change committed as `fix` with no body, Brain cannot answer any of the above usefully. The work is invisible to anyone who was not there when it shipped. Six months later, that includes the person who shipped it.

### A.3 Practical implication

The investment is small: about sixty seconds per commit, two minutes per PR. The return compounds across every future query. A repository's commit hygiene is the single biggest predictor of how useful Brain is there. This is not a theory; it is what we see today.

Commits and PR descriptions are not just for the reviewer. **They are the corpus. Treat them accordingly.**
