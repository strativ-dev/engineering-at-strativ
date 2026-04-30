# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Engineering @ Strativ AB** is a technical blog built with [Pelican](https://getpelican.com/), a Python static site generator. Markdown content in `content/` is processed into static HTML in `docs/`, which is deployed to GitHub Pages at `https://strativ-dev.github.io/engineering-at-strativ`.

## Commands

```bash
# Install dependencies (requires Python 3.10.8)
pip install -r requirements.txt

# Local development with auto-reload
make devserver           # Serves at http://localhost:8000

# Generate static output only
make html                # Outputs to docs/

# Build for production (uses publishconf.py)
make publish

# Deploy to GitHub Pages
make github

# Clean generated output
make clean
```

## Architecture

**Content flow:**
```
content/*.md  →  Pelican (pelicanconf.py)  →  docs/ (static HTML)  →  GitHub Pages
```

- `content/` — Markdown articles with YAML frontmatter (Title, Date, Category, Tags, Authors, Slug, Summary). Images go in `content/images/`, static files (favicon, robots.txt) in `content/extra/`.
- `theme/` — Jinja2 templates and CSS. `theme/templates/` has per-page-type templates; `theme/static/` has the stylesheet. The base template loads Bootstrap 5.2.2 and FontAwesome 5.15.3 from CDN.
- `pelicanconf.py` — Development configuration. `publishconf.py` overrides for production (sets `SITEURL`, disables relative URLs, enables `DELETE_OUTPUT_DIRECTORY`).
- `docs/` — Pre-generated HTML committed to the repo; GitHub Pages serves directly from this directory on `main`.

**Plugins:** `pelican-series` — groups articles into ordered series via `Series` and `Series_index` frontmatter fields.

## Writing Articles

New articles are Markdown files in `content/`. Required frontmatter fields:

```markdown
Title: Your Article Title
Date: YYYY-MM-DD HH:MM
Category: Category Name
Tags: tag1, tag2
Authors: Author Name
Summary: Short description shown in listings
Slug: url-friendly-slug
```

To add an article to a series, include:
```markdown
Series: Series Name
Series_index: 1
```
