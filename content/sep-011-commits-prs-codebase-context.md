Title: SEP-011: Improving Commits, PR Descriptions & Codebase Context in Git
Date: 2026-04-30
Category: Engineering Process
Slug: sep-011-commits-prs-codebase-context
Authors: Saqibur Rahman
Tags: git, commits, pull-requests, conventional-commits, engineering-process
Summary: SEP-011 establishes a binding standard for commit messages, branch names, PR titles, and PR descriptions at Strativ  -  adopting Conventional Commits and explaining why Git artefact quality is a direct input to every system that depends on codebase context.

| Field | Value |
|---|---|
| **SEP** | 011 |
| **Title** | Improving Commits, PR Descriptions & Codebase Context in Git |
| **Author** | Saqibur Rahman &lt;saqibur.rahman@strativ.se&gt; |
| **Status** | Draft |
| **Type** | Process |
| **Created** | 2026-04-30 |
| **Target Adoption** | 2026-Q2 |
| **Requires** | SEP-001 |
| **Related** | SEP-004, SEP-005, SEP-010 |

---

## Abstract

This SEP establishes a binding standard for what Strativ developers write in Git  -  commit messages, branch names, pull request titles, and pull request descriptions. It adopts the Conventional Commits specification as the required commit format, defines the structure of useful PR descriptions, and explains why the quality of these artefacts is no longer just an engineering hygiene concern but a direct input to the systems  -  both human and AI  -  that depend on Strativ's accumulated codebase context.

It does not require AI assistance to be used. It does explicitly permit and encourage it for developers who find writing commit messages tedious, while making clear that the Pilot remains accountable for what gets written either way.

---

## 1. Motivation

Every commit message and PR description Strativ produces becomes part of the historical record. That record is read by:

- The developer who wrote it, six months later, when something breaks.
- A new joiner trying to understand why a piece of code looks the way it does.
- A reviewer trying to compress "what changed and why" into thirty seconds of attention.
- A client asking what was delivered in the last sprint.
- **Strativ Brain**  -  our internal context engine  -  which indexes commit history, PR descriptions, and code together to answer questions across projects.
- Every AI Copilot that subsequently works in the repository, all of which have only what was written down to go on.

The current state is uneven. We have repositories with thoughtful, structured histories that read like documentation. We also have repositories where half the commits say "fix" or "wip" or "updates," and PR descriptions that contain only the auto-generated branch name. This SEP exists because that gap is now expensive in ways it was not before.

### 1.1 The cost of a meaningless commit

Consider two commits, both touching the same authentication module:

| ✓ GOOD | ✗ BAD |
|---|---|
| `feat(auth): add JWT refresh token rotation` | `fix` |
| Replaces single long-lived JWT with rotating refresh tokens. Refresh tokens are single-use; reuse triggers session revocation across all devices. Closes BMS-247. | *(no body)* |

Both commits change the same number of lines. Both pass tests. Both ship. But only the first one survives contact with anyone who reads the repository later. The second is dead weight  -  present in history, useless for understanding it. A Copilot asked to summarise recent auth work in this repository can produce a coherent answer from the first commit and nothing useful from the second.

### 1.2 The brain only knows what we write down

Strativ Brain (and every other context system, internal or AI) is not a magic source of truth  -  it is a search and retrieval layer over what we, the developers, have already written. Its output quality is bounded by the quality of its input. "Garbage in, garbage out" is unusually literal here: a brain trained on a repository where 40% of commits say "fix" will cheerfully tell you that 40% of recent work involved fixing something, with no further detail available.

This is not a problem we can solve later by improving the brain. It is a problem solved earlier, in the repository, by writing better commits and PR descriptions in the first place.

---

## 2. Specification

### 2.1 Commit message standard

Strativ adopts the **Conventional Commits specification, version 1.0.0**, as the required format for all commit messages in repositories under active development.

The structure is:

```
<type>(<scope>): <short description>

<optional body>

<optional footer(s)>
```

#### 2.1.1 Required types

Every commit must begin with one of the following types. The list is closed  -  "misc," "update," and similar are not types.

| Type | Use for |
|---|---|
| `feat` | A new feature visible to a user, an API consumer, or another developer. |
| `fix` | A bug fix. Reference the bug or ticket in the body or footer. |
| `refactor` | Changes that neither add features nor fix bugs. Internal restructuring. |
| `perf` | Changes that improve performance. State the measured improvement in the body. |
| `test` | Adding or updating tests. No production code changes. |
| `docs` | Documentation only  -  README, ADRs, code comments, runbooks. |
| `build` | Changes to build system, dependencies, packaging. |
| `ci` | Changes to CI/CD pipelines, GitHub Actions, deployment scripts. |
| `chore` | Maintenance tasks that don't fit the above. Use sparingly. |

#### 2.1.2 Scope

Scope is optional but strongly preferred. It names the area of the codebase being changed  -  typically a module, package, app, or domain. Each repository should document its scope vocabulary in its `CONTRIBUTING.md` or `CLAUDE.md`.

Common Strativ scope examples:

- `auth`, `billing`, `consumption`, `agreement`, `placement` (domain modules)
- `api`, `web`, `worker`, `deploy` (technical layers)
- `BMS`, `IMS`, `LumberScan` (project codes for cross-cutting work)

#### 2.1.3 Description

The description follows the colon. It must:

- Be written in the **imperative mood**  -  "add", not "added" or "adds". Read it as completing the sentence "If applied, this commit will…"
- Start with a lowercase letter.
- Not end with a period.
- Stay under 72 characters total (type + scope + colon + description).
- Describe **what the change does**, not what files were touched.

#### 2.1.4 Body

The body is where the **why** lives. It is optional for trivial commits but required for any change that is not self-evident from the diff. The body explains:

- Why the change is being made (what problem it solves).
- What approach was chosen and what alternatives were considered, when relevant.
- Side effects, migration steps, or follow-up work the reader should be aware of.

Wrap body lines at 72 characters. Separate paragraphs with a blank line.

#### 2.1.5 Breaking changes

A breaking change is indicated by either:

- An exclamation mark after the type/scope: `feat(auth)!: replace token format`.
- A footer line beginning with `BREAKING CHANGE:` explaining what breaks and how to migrate.

Both are acceptable. The footer form is preferred when the migration story is non-trivial.

#### 2.1.6 Footers

Footers carry structured metadata that downstream tools can parse:

```
Closes BMS-247
Refs IMS-031, IMS-032
Reviewed-by: Ludwig
Co-authored-by: Contributor <contributor@example.com>
BREAKING CHANGE: token storage moved from cookie to header
```

### 2.2 Branch naming

Branch names are short-lived but inform PR titles, deployments, and history searches. They should follow:

```
<type>/<scope>-<short-slug>

feat/auth-jwt-refresh-rotation
fix/consumption-monthly-range-off-by-one
refactor/placement-selector
chore/upgrade-django-5
```

Avoid: branch names that are only a ticket number (`BMS-247`), only a person's name (`saqib-stuff`), or fully unstructured (`new-auth-flow-v2-final`).

### 2.3 Pull request titles

PR titles follow the same Conventional Commits format as commit messages. When a PR contains a single commit, the PR title and the commit subject should be identical. When the PR will be squash-merged, the PR title becomes the commit subject and must satisfy §2.1.

Avoid PR titles that:

- Restate the branch name as a sentence ("Add JWT refresh token rotation work").
- Reference only the ticket ("BMS-247").
- Use "Update X" or "Fix X" without specifying what.

### 2.4 Pull request descriptions

PR descriptions are the place where context that doesn't belong in commit messages goes  -  screenshots, deployment notes, testing instructions, reviewer guidance. Every PR uses the following template:

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

Trivial PRs (one-line typo fixes, dependency bumps) may collapse the template to a single sentence under **What**. The five-section structure is the default; deviations are deliberate.

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

Per SEP-001, Pilots may use Copilots for any drafting task where the Pilot remains accountable for output. Commit messages and PR descriptions are explicitly included.

A well-framed Copilot can read a diff and produce a competent first-draft commit message in seconds. Pilots who find writing commits tedious are encouraged to delegate the first draft and edit. Pilots who prefer to write their own remain free to do so. Either choice is acceptable; no choice is.

> **PILOT ACCOUNTABILITY**
>
> If the Copilot's draft is wrong, misleading, or generic, the Pilot edits or rewrites it before committing. Shipping an unread Copilot-generated commit message is the same accountability failure as shipping an unread Copilot-generated function. The Pilot's name is on the commit either way.

**What a Copilot is genuinely good at:**

- Producing the first-draft body when the developer has the diff in hand but doesn't want to write English at the end of a long day.
- Suggesting the correct Conventional Commits type from the actual change content.
- Drafting PR descriptions from a series of commits, scaffolding the What/Why/How/Verification sections.
- Catching commits about to ship as "fix" and asking the Pilot for a real description.

**What a Copilot is unreliable at, and where the Pilot must add value:**

- Knowing *why* a change was made when that reason isn't in the diff.
- Choosing the right scope when scope vocabulary is project-specific.
- Identifying breaking changes correctly when the breaking aspect is semantic, not structural.
- Writing the Risks & Follow-up section honestly  -  that requires Pilot judgement about the surrounding system.

---

## 3. Rationale

Three reasons drive this SEP, in increasing order of weight.

**First, reviewer experience.** A reviewer who can read a PR's What/Why/How block in thirty seconds and form an opinion is a reviewer who reviews promptly and reviews well. A reviewer who has to reverse-engineer the intent from the diff is slower, more error-prone, and  -  over time  -  more reluctant. The cost of a thirty-second write is recouped many times over in faster, sharper reviews.

**Second, future-self experience.** Every developer eventually opens `git blame` on a line they wrote two years ago and wonders why. A commit that says "fix" is a small betrayal of the future self who will need to know. A commit that says `fix(consumption): correct off-by-one in monthly range split  -  caused first day of February to be excluded from billing` is a gift to that same person.

**Third  -  and this is the new weight in 2026  -  context-engine readability.** Strativ Brain, AI Copilots, and any future tools that reason over our codebase are all consumers of what we write. They cannot improve quality on a corpus of "fix" commits. This is not a hypothetical concern; it is already shaping how useful (or not) our internal AI tooling is per repository. The repositories with disciplined commit hygiene are the repositories where Brain is genuinely useful. The correlation is direct.

Conventional Commits specifically  -  rather than a Strativ-invented format  -  was chosen because it is widely adopted, well-tooled (commitlint, semantic-release, changelog generation), and immediately recognisable to any developer joining from outside. Using a global standard means we are not asking new joiners to learn Strativ-specific conventions on day one.

---

## 4. Backwards Compatibility

Existing repositories are not retroactively rewritten. Commit history before the adoption date stays as it is  -  rewriting Git history to enforce a new standard would break references, links, and tooling for negligible benefit.

From the adoption date forward:

- New commits in repositories under active development must follow §2.1.
- New PRs must use the §2.4 template (with deviation permitted for genuinely trivial changes).
- Repositories add a `CONTRIBUTING.md` (or extend their `CLAUDE.md`) listing their scope vocabulary.
- Archived or read-only repositories are exempt.

Enforcement is initially advisory. After one quarter, repositories may opt in to commitlint or equivalent CI checks that reject non-conforming commits. Universal enforcement is a separate decision, taken after the team has lived with the standard.

---

## 5. Reference Implementation

### 5.1 commitlint configuration

Repositories opting into automated enforcement should use commitlint with the conventional-commits preset. A reference configuration:

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

Repositories add `.github/PULL_REQUEST_TEMPLATE.md` (or the equivalent for their host) containing the §2.4 structure as an empty form. The template is shown to PR authors automatically when they open a PR.

### 5.3 Reference repositories

Repositories that exemplify the standard well  -  and may be used as templates  -  include the Boo Energi backend (recent BMS-2xx work), the IMS maintenance branch, and the Strativ skill registry. New repositories may copy their `CONTRIBUTING.md` and PR template directly.

---

## 6. Open Issues

- Should commitlint enforcement be opt-in indefinitely, or required after a transition window?
- How do we handle repositories shared with clients who have their own conventions? *(Provisional answer: client conventions win in client-owned repos; Strativ conventions apply in Strativ-owned repos.)*
- Squash-merge vs. merge-commit policy is currently per-repository. Should it be standardised?
- How do we surface the scope vocabulary to a Copilot working in a repository without making it brittle?
- Co-authorship trailers when a Copilot drafted the commit  -  currently not required. Revisit if external tooling expects them.

---

## Appendix A  -  The Strativ Brain Connection

Strativ Brain is the working name for our internal context engine  -  a system that indexes commits, PR descriptions, ADRs, internal documents, and code to answer questions across projects. It is consumed both by humans ("what was the last thing we did on auth?") and by AI tooling (Copilots that need cross-project context to give good answers).

Brain is a retrieval and synthesis layer. It does not invent context that does not exist. Its outputs are bounded above by the quality of its inputs, and the inputs are largely Git artefacts.

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

- "What changed in auth recently?"  -  surfaces this commit with a meaningful one-line summary.
- "Why did we add session revocation?"  -  points to this commit and the ticket.
- "Show me feature work on Boo Energi this month"  -  categorises this as a `feat` correctly.
- "Did we change how refresh tokens work?"  -  direct hit on the body text.

### A.2 What Brain cannot do with a bad commit

Given the same change committed as `fix` with no body, Brain can answer none of the above usefully. The work is invisible to anyone who was not in the room when it shipped. Six months later, that includes the person who shipped it.

### A.3 Practical implication

The unit of investment is small  -  sixty seconds of writing per commit, two minutes per PR  -  and the unit of return compounds across every future query. A repository's commit hygiene is the single largest predictor of how useful Brain is in that repository. This is not a theoretical correlation; it is what we observe today.

Commits and PR descriptions are not just for the reviewer. **They are the corpus. Treat them accordingly.**
