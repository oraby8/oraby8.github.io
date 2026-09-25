# Design Spec: Adam Rackis Style Blog Redesign

**Date:** 2026-09-25  
**Project:** oraby8.github.io  
**Reference Site:** [https://adamrackis.dev/](https://adamrackis.dev/)

---

## 1. Overview & Objectives

Transform the current Jekyll-based personal engineering blog (`oraby8.github.io`) from its legacy, heavy full-bleed theme (Centrarium with Bourbon/Neat) into an ultra-clean, minimalist, content-first technical blog modeled directly on **Adam Rackis's personal site** (`https://adamrackis.dev/`).

The new design emphasizes:
- **Maximum readability**: Centered 708px content column, Georgia serif body text with 1.666 line height, and bold modern sans-serif headings.
- **Direct personal branding**: Header with a circular avatar, title, subtitle, inline SVG social links, and an introductory bio directly on the homepage.
- **Streamlined feed**: Posts listed cleanly with bold blue title links (`#007acc`), italicized publication date & category, and concise excerpts.
- **Distraction-free post reading**: Simple back-link (`← Ahmed's Blog`), clean headings, responsive images, elegant callout blockquotes, and crisp dark code blocks.
- **Modern lightweight CSS**: Pure CSS/SCSS with CSS custom properties (variables), eliminating deprecated Bourbon/Neat mixin libraries while maintaining 100% GitHub Pages native compatibility.
- **Automatic Dark Mode**: Seamless support for `prefers-color-scheme: dark`.

---

## 2. Visual Architecture & Design System

### 2.1 Container & Layout
- **Page Container**: Centered column with `max-width: 708px`, `margin: 0 auto`, `padding: 0 1rem`.
- **Top Bar**: Remove the bulky navigation header bar, mobile drawer menu, and full-bleed hero banner image with dark scrim.
- **Footer**: Minimalist copyright and RSS feed link matching the clean aesthetic.

### 2.2 Color Tokens
CSS variables defined on `:root` and overridden in `@media (prefers-color-scheme: dark)`:

| Token | Light Mode (Default) | Dark Mode |
|---|---|---|
| `--bg-color` | `#ffffff` | `#121212` |
| `--text-color` | `#1f2937` (soft charcoal) | `#e5e7eb` (off-white) |
| `--heading-color` | `#111827` | `#f9fafb` |
| `--link-color` | `#007acc` (TypeScript / VS Code blue) | `#38bdf8` (sky blue) |
| `--link-hover-color` | `#005999` | `#7dd3fc` |
| `--meta-color` | `#6b7280` | `#9ca3af` |
| `--blockquote-bg` | `#f3f4f6` / `#e9e7e2` | `#1f2937` |
| `--border-color` | `#e5e7eb` | `#2d3748` |
| `--code-inline-bg` | `#f1f5f9` | `#1e293b` |
| `--code-inline-color`| `#0f172a` | `#e2e8f0` |

### 2.3 Typography Stack
- **Body & Paragraphs**: `Georgia, "Times New Roman", Times, serif`
  - Font size: `1rem` (16px) on mobile, `1.0625rem` (17px) on desktop.
  - Line height: `1.666` for optimal reading cadence.
- **Headings (`h1`–`h6`)**: `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif`
  - `font-weight: 700`, `line-height: 1.15`, tight tracking.
  - `h1`: `1.875rem`–`2.25rem` (30px–36px).
  - `h2`: `1.5rem` (24px).
  - `h3`: `1.25rem` (20px).
- **Code & Monospace**: `ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace`.

---

## 3. Component Specifications

### 3.1 Homepage Header (`index.html` / `_includes/header.html`)
Matches Adam Rackis's homepage header layout:
```html
<div class="blog-header">
  <div class="avatar-container">
    <img src="/assets/avatar.png" alt="Ahmed Samir Oraby" class="avatar" />
  </div>
  <div class="header-titles">
    <h1 class="site-title">Ahmed Samir Oraby</h1>
    <h3 class="site-subtitle">Senior NLP & ML Engineer | Arabic LLMs, Speech & Autonomous AI Agents</h3>
    <div class="personal-links">
      <a href="https://github.com/oraby8" class="social-link" target="_blank" rel="noopener">
        <!-- SVG GitHub Icon -->
        <span>oraby8</span>
      </a>
      <a href="https://www.linkedin.com/in/ahmed-oraby-7b076881/" class="social-link" target="_blank" rel="noopener">
        <!-- SVG LinkedIn Icon -->
        <span>Ahmed Samir Oraby</span>
      </a>
    </div>
  </div>
</div>
```
- **Desktop**: Flex row, avatar 125px × 125px with `border-radius: 50%`, titles aligned vertically.
- **Mobile**: Flex row or compact stack, avatar 96px × 96px.

### 3.2 Bio Introduction Section
Immediately following the header:
```html
<div class="bio-intro">
  <p>Hi, I'm Ahmed 👋</p>
  <p>Welcome to my technical blog. I specialize in Large Language Models (fine-tuning, RLHF/DPO, LoRA), Arabic NLP & Speech (TTS, phonemization), and production AI agent architectures. Here I share deep-dive engineering insights, benchmarks, and production lessons.</p>
</div>
```

### 3.3 Post Feed Items
Each post in `index.html` rendered as:
```html
<div class="blog-list-item">
  <h2 class="post-item-title">
    <a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
  </h2>
  <div class="post-item-meta">
    <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%B %-d, %Y" }}</time>
    {% if post.categories.size > 0 %}
      <span class="meta-separator">•</span>
      <span class="meta-category">{{ post.categories | join: ", " | upcase }}</span>
    {% endif %}
  </div>
  <p class="post-item-excerpt">
    {{ post.content | strip_html | truncatewords: 35 }}
  </p>
</div>
```
- Margin bottom: `2rem` (32px) between items.

### 3.4 Post Page Layout (`_layouts/post.html`)
- **Back link at top**:
  ```html
  <div class="back-nav">
    <a href="{{ '/' | prepend: site.baseurl }}" class="back-link">
      <svg ...><!-- Left Arrow SVG --></svg>
      <span>Ahmed's Blog</span>
    </a>
  </div>
  ```
- **Post Header**:
  - `h1.post-title`: Post title in bold sans-serif.
  - `div.post-date`: Formatted publication date + reading time.
- **Post Body (`post-content`)**:
  - Georgia serif body text.
  - Code blocks (`pre code`): Dark background (`#1e1e1e`), syntax highlighted, rounded corners (`6px`), padding, horizontal scroll.
  - Blockquotes: Monospace/callout style with soft background (`var(--blockquote-bg)`), left border accent, padded box.
  - Tables: Bordered, clean typography, horizontal scroll if overflow.
  - Images: Centered block, max-width 100%, slight border radius.
- **Post Footer**:
  - Tags and category links.
  - Prev/Next navigation links (if enabled).

### 3.5 Pagination Controls
Clean, minimal pagination at the bottom of the list:
- Simple "← Newer Posts" and "Older Posts →" or page numbers with no bulky box shadows or square fa-stacks.

---

## 4. File Structure Changes

1. **`css/main.scss`**:
   - Streamline Sass imports. Remove legacy Bourbon and Neat mixins.
   - Introduce modern CSS resets, CSS custom property definitions (with dark mode media queries), typography rules, header styles, post list styles, and article formatting.
2. **`_layouts/default.html`**:
   - Modernize HTML5 doctype and wrapper `<main class="site-container">`.
   - Remove legacy navigation bar and bulky footer.
3. **`_layouts/post.html`**:
   - Replace cover image scrim with the clean `← Ahmed's Blog` back-link and centered post article layout.
4. **`_includes/header.html`**:
   - Clean, lightweight header with avatar, titles, and SVG social links.
5. **`_includes/footer.html`**:
   - Subtle, minimal footer (RSS link, copyright).
6. **`index.html`**:
   - Clean bio introduction and post feed matching Adam Rackis's structure.

---

## 5. Verification & Testing Criteria

1. **Jekyll Build Integrity**:
   - Run `bundle exec jekyll build` (or verify standard Kramdown/Sass syntax) with zero compilation or syntax errors.
2. **Responsive Fidelity**:
   - Validate on desktop (>768px): Avatar 125px, max-width 708px centered, clean typography.
   - Validate on mobile (<480px): Avatar responsive, horizontal margins intact, no horizontal overflow.
3. **Theme & Dark Mode Verification**:
   - Verify light theme colors match reference aesthetic (`#ffffff`, `#1f2937`, `#007acc`).
   - Verify dark theme colors activate on `prefers-color-scheme: dark`.
4. **Content & Post Rendering**:
   - Existing post (`2026-09-23-fine-tuning-arabic-llms-production-lessons.md`) renders cleanly with back link, code blocks, lists, and headings.
