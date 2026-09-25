# Adam Rackis Style Blog Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign `oraby8.github.io` from the legacy Centrarium theme into an ultra-clean, minimalist technical blog modeled on [adamrackis.dev](https://adamrackis.dev/) with light and automatic dark mode support.

**Architecture:** Single centered 708px content column (`max-width: 708px`), clean CSS variables for light/dark theming, Georgia serif body reading font paired with modern sans-serif headings, circular avatar header with inline SVG social links, and distraction-free article layout with a top back-link.

**Tech Stack:** Jekyll, Liquid templates, modern SCSS / CSS Custom Properties, HTML5, SVG icons.

---

## File Structure

- **Modified:**
  - `_config.yml` - Clean configuration, author metadata, theme settings.
  - `_includes/head.html` - Modern HTML5 head, responsive viewport, clean syntax theme, remove legacy CDNs.
  - `_includes/header.html` - Homepage header with circular avatar, titles, and inline SVG social links (GitHub & LinkedIn).
  - `_includes/footer.html` - Minimal, low-key footer with copyright and RSS link.
  - `_layouts/default.html` - Main container wrapper `<main class="site-container">` constrained to 708px.
  - `_layouts/post.html` - Distraction-free post page with back link (`← Ahmed's Blog`), bold title, date, and clean typography.
  - `_layouts/page.html` - Clean page layout adhering to the 708px container.
  - `_layouts/archive.html` - Minimalist archive listing for tags and categories.
  - `index.html` - Homepage feed with avatar header, bio intro, post cards, and clean pagination.
  - `css/main.scss` - Core stylesheet using CSS custom properties, light/dark themes, and typography.
- **Created:**
  - `_sass/_variables.scss` - CSS custom properties for light and dark themes, spacing, typography scales.
  - `_sass/_reset.scss` - Modern box-sizing and HTML reset.
  - `_sass/_typography.scss` - Georgia serif body and system sans-serif heading rules.
  - `_sass/_layout.scss` - Centered 708px grid/flex containers, header, bio, post feed, and post detail styles.
  - `scripts/verify_site.py` - Python verification script validating front matter, HTML templates, and CSS variables.

---

### Task 1: Create Modern SCSS Architecture & CSS Tokens

**Files:**
- Create: `_sass/_variables.scss`
- Create: `_sass/_reset.scss`
- Create: `_sass/_typography.scss`
- Create: `_sass/_layout.scss`
- Modify: `css/main.scss`

- [ ] **Step 1: Write `_sass/_variables.scss` with CSS variables for light and dark modes**

```scss
// _sass/_variables.scss
:root {
  --bg-color: #ffffff;
  --text-color: #1f2937;
  --heading-color: #111827;
  --link-color: #007acc;
  --link-hover-color: #005999;
  --meta-color: #6b7280;
  --meta-italic-color: #4b5563;
  --border-color: #e5e7eb;
  --blockquote-bg: #f3f4f6;
  --blockquote-border: #d1d5db;
  --code-inline-bg: #f1f5f9;
  --code-inline-color: #0f172a;
  --code-block-bg: #1e1e1e;
  --code-block-text: #d4d4d4;
  --back-link-color: #1f2937;
  
  --font-serif: Georgia, "Times New Roman", Times, serif;
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  
  --max-width: 708px;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg-color: #121212;
    --text-color: #e5e7eb;
    --heading-color: #f9fafb;
    --link-color: #38bdf8;
    --link-hover-color: #7dd3fc;
    --meta-color: #9ca3af;
    --meta-italic-color: #9ca3af;
    --border-color: #27272a;
    --blockquote-bg: #1e1e1e;
    --blockquote-border: #3f3f46;
    --code-inline-bg: #27272a;
    --code-inline-color: #f3f4f6;
    --code-block-bg: #18181b;
    --code-block-text: #e4e4e7;
    --back-link-color: #e5e7eb;
  }
}
```

- [ ] **Step 2: Write `_sass/_reset.scss` with clean base styling**

```scss
// _sass/_reset.scss
*, *::before, *::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

html {
  -webkit-text-size-adjust: 100%;
  tab-size: 4;
}

body {
  background-color: var(--bg-color);
  color: var(--text-color);
  font-family: var(--font-serif);
  font-size: 1.0625rem;
  line-height: 1.666;
  transition: background-color 0.2s ease, color 0.2s ease;
  min-height: 100vh;
  padding: 2.5rem 1rem 4rem;
}

a {
  color: var(--link-color);
  text-decoration: none;
  transition: color 0.15s ease;
  
  &:hover {
    color: var(--link-hover-color);
    text-decoration: underline;
  }
}

img, svg {
  display: block;
  max-width: 100%;
}

ul, ol {
  padding-left: 1.5rem;
}
```

- [ ] **Step 3: Write `_sass/_typography.scss` with typography rules**

```scss
// _sass/_typography.scss
h1, h2, h3, h4, h5, h6 {
  font-family: var(--font-sans);
  color: var(--heading-color);
  font-weight: 700;
  line-height: 1.2;
}

h1 {
  font-size: 2.125rem;
  letter-spacing: -0.025em;
  margin-bottom: 0.5rem;
}

h2 {
  font-size: 1.5rem;
  letter-spacing: -0.02em;
  margin-top: 2rem;
  margin-bottom: 0.75rem;
}

h3 {
  font-size: 1.25rem;
  margin-top: 1.5rem;
  margin-bottom: 0.5rem;
}

p {
  margin-bottom: 1.5rem;
}

code {
  font-family: var(--font-mono);
  font-size: 0.9em;
  background-color: var(--code-inline-bg);
  color: var(--code-inline-color);
  padding: 0.15rem 0.35rem;
  border-radius: 4px;
}

pre {
  background-color: var(--code-block-bg);
  color: var(--code-block-text);
  border-radius: 6px;
  padding: 1rem 1.25rem;
  overflow-x: auto;
  margin-bottom: 1.5rem;
  line-height: 1.5;
  border: 1px solid var(--border-color);
  
  code {
    background: transparent;
    color: inherit;
    padding: 0;
    font-size: 0.875rem;
  }
}

blockquote {
  background-color: var(--blockquote-bg);
  border-left: 4px solid var(--blockquote-border);
  padding: 1rem 1.25rem;
  margin: 1.5rem 0;
  border-radius: 0 4px 4px 0;
  font-style: italic;
  
  p:last-child {
    margin-bottom: 0;
  }
}

hr {
  border: 0;
  border-top: 1px solid var(--border-color);
  margin: 2.5rem 0;
}
```

- [ ] **Step 4: Write `_sass/_layout.scss` with Adam Rackis layout components**

```scss
// _sass/_layout.scss
.site-container {
  max-width: var(--max-width);
  margin: 0 auto;
  width: 100%;
}

// Homepage Header
.blog-header {
  display: flex;
  align-items: center;
  gap: 1.25rem;
  margin-bottom: 2rem;

  .avatar-container {
    width: 125px;
    height: 125px;
    min-width: 125px;
    min-height: 125px;
    border-radius: 50%;
    overflow: hidden;
    background-color: var(--blockquote-bg);
    border: 2px solid var(--border-color);

    img.avatar {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
  }

  .header-titles {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;

    .site-title {
      font-size: 1.875rem;
      font-weight: 700;
      line-height: 1.1;
      margin: 0;
    }

    .site-subtitle {
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--meta-color);
      line-height: 1.35;
      margin: 0;
    }

    .personal-links {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.75rem;
      margin-top: 0.25rem;

      .social-link {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-family: var(--font-sans);
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--link-color);

        svg {
          width: 15px;
          height: 15px;
          fill: currentColor;
        }

        &:hover {
          text-decoration: underline;
        }
      }
    }
  }
}

// Bio intro
.bio-intro {
  margin-bottom: 2.25rem;
  font-size: 1.0625rem;
  line-height: 1.666;

  p {
    margin-bottom: 0.75rem;
    &:last-child {
      margin-bottom: 0;
    }
  }
}

// Post list item
.blog-list-item {
  margin-bottom: 2.25rem;

  .post-item-title {
    font-size: 1.5rem;
    font-weight: 700;
    line-height: 1.25;
    margin: 0 0 0.35rem;

    a {
      color: var(--link-color);
      &:hover {
        color: var(--link-hover-color);
      }
    }
  }

  .post-item-meta {
    font-family: var(--font-sans);
    font-size: 0.85rem;
    font-style: italic;
    color: var(--meta-italic-color);
    margin-bottom: 0.5rem;

    .meta-separator {
      margin: 0 0.35rem;
      font-style: normal;
    }

    .meta-category {
      font-weight: 600;
      font-style: normal;
      font-size: 0.75rem;
      letter-spacing: 0.05em;
    }
  }

  .post-item-excerpt {
    font-size: 1rem;
    line-height: 1.6;
    margin: 0;
  }
}

// Post view
.post-view {
  .back-nav {
    margin-bottom: 1.5rem;

    .back-link {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      font-family: var(--font-sans);
      font-size: 1rem;
      font-weight: 600;
      color: var(--back-link-color);

      svg {
        width: 16px;
        height: 16px;
        fill: currentColor;
      }

      &:hover {
        color: var(--link-color);
        text-decoration: none;
      }
    }
  }

  .post-title {
    font-size: 2.25rem;
    font-weight: 700;
    line-height: 1.15;
    margin-bottom: 0.5rem;
  }

  .post-date {
    font-family: var(--font-sans);
    font-size: 0.875rem;
    font-style: italic;
    color: var(--meta-italic-color);
    margin-bottom: 2rem;
  }

  .post-content {
    line-height: 1.7;

    img {
      border-radius: 6px;
      margin: 1.5rem auto;
    }
  }
}

// Pagination
.pagination {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 3rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--border-color);
  font-family: var(--font-sans);
  font-size: 0.9rem;
  font-weight: 600;

  .page-number {
    color: var(--meta-color);
    font-style: italic;
    font-weight: 400;
  }
}

// Footer
.site-footer {
  margin-top: 4rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--border-color);
  font-family: var(--font-sans);
  font-size: 0.85rem;
  color: var(--meta-color);
  display: flex;
  justify-content: space-between;
  align-items: center;

  a {
    color: var(--meta-color);
    &:hover {
      color: var(--link-color);
    }
  }
}

// Responsive
@media (max-width: 600px) {
  body {
    padding-top: 1.5rem;
  }

  .blog-header {
    gap: 1rem;

    .avatar-container {
      width: 90px;
      height: 90px;
      min-width: 90px;
      min-height: 90px;
    }

    .header-titles {
      .site-title {
        font-size: 1.5rem;
      }
      .site-subtitle {
        font-size: 0.85rem;
      }
    }
  }

  .post-view .post-title {
    font-size: 1.75rem;
  }
}
```

- [ ] **Step 5: Replace `css/main.scss` with imports to the new architecture**

```scss
---
# Main Sass entry point
---
@charset "utf-8";

@import "variables";
@import "reset";
@import "typography";
@import "layout";
```

- [ ] **Step 6: Commit SCSS Architecture**

```bash
git add _sass/_variables.scss _sass/_reset.scss _sass/_typography.scss _sass/_layout.scss css/main.scss
git commit -m "feat(css): implement modern SCSS architecture with Adam Rackis styling and dark mode"
```

---

### Task 2: Modernize Site Includes (`head.html`, `header.html`, `footer.html`)

**Files:**
- Modify: `_includes/head.html`
- Modify: `_includes/header.html`
- Modify: `_includes/footer.html`

- [ ] **Step 1: Update `_includes/head.html`**

Keep modern meta tags, clean syntax highlighting styles, SVG favicon, and link to `/css/main.css`. Remove heavy font-awesome and lightbox2 CDN scripts.

```html
<head>
  <meta charset="utf-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <title>{% if page.title %}{{ page.title }} | {{ site.title }}{% else %}{{ site.title }} - {{ site.subtitle }}{% endif %}</title>
  <meta name="description" content="{% if page.excerpt %}{{ page.excerpt | strip_html | strip_newlines | truncate: 160 }}{% else %}{{ site.description }}{% endif %}">
  {% if site.name %}
  <meta name="author" content="{{ site.name }}">
  {% endif %}

  <!-- Favicons -->
  <link rel="icon" type="image/png" sizes="32x32" href="{{ '/assets/icons/favicon-32x32.png' | prepend: site.baseurl }}">
  <link rel="icon" type="image/png" sizes="16x16" href="{{ '/assets/icons/favicon-16x16.png' | prepend: site.baseurl }}">
  <link rel="apple-touch-icon" sizes="180x180" href="{{ '/assets/icons/apple-icon-180x180.png' | prepend: site.baseurl }}">

  <!-- Syntax highlighting theme (clean dark style matching Adam Rackis) -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">

  <!-- Main Stylesheet -->
  <link rel="stylesheet" href="{{ '/css/main.css' | prepend: site.baseurl }}">
  <link rel="canonical" href="{{ page.url | replace:'index.html','' | prepend: site.baseurl | prepend: site.url }}">
  <link rel="alternate" type="application/rss+xml" title="{{ site.title }}" href="{{ '/feed.xml' | prepend: site.baseurl | prepend: site.url }}" />
</head>
```

- [ ] **Step 2: Update `_includes/header.html` to Adam Rackis header**

```html
<header class="blog-header">
  <div class="avatar-container">
    <img src="{{ site.avatar | default: '/assets/avatar.png' | prepend: site.baseurl }}" alt="{{ site.name | default: site.title }}" class="avatar">
  </div>
  <div class="header-titles">
    <h1 class="site-title"><a href="{{ site.baseurl }}/" style="color: inherit; text-decoration: none;">{{ site.title }}</a></h1>
    {% if site.subtitle %}<p class="site-subtitle">{{ site.subtitle }}</p>{% endif %}
    <div class="personal-links">
      {% for item in site.social %}
        {% if item.url != "" and item.username != "" %}
          <a class="social-link" href="{{ item.url }}" target="_blank" rel="noopener noreferrer">
            {% if item.name == "GitHub" %}
            <svg viewBox="0 0 496 512" xmlns="http://www.w3.org/2000/svg">
              <path d="M165.9 397.4c0 2-2.3 3.6-5.2 3.6-3.3.3-5.6-1.3-5.6-3.6 0-2 2.3-3.6 5.2-3.6 3-.3 5.6 1.3 5.6 3.6zm-31.1-4.5c-.7 2 1.3 4.3 4.3 4.9 2.6 1 5.6 0 6.2-2s-1.3-4.3-4.3-5.2c-2.6-.7-5.5.3-6.2 2.3zm44.2-1.7c-2.9.7-4.9 2.6-4.6 4.9.3 2 2.9 3.3 5.9 2.6 2.9-.7 4.9-2.6 4.6-4.6-.3-1.9-3-3.2-5.9-2.9zM244.8 8C106.1 8 0 113.3 0 252c0 110.9 69.8 205.8 169.5 239.2 12.8 2.3 17.3-5.6 17.3-12.1 0-6.2-.3-40.4-.3-61.4 0 0-70 15-84.7-29.8 0 0-11.4-29.1-27.8-36.6 0 0-22.9-15.7 1.6-15.4 0 0 24.9 2 38.6 25.8 21.9 38.6 58.6 27.5 72.9 20.9 2.3-16 8.8-27.1 16-33.7-55.9-6.2-112.3-14.3-112.3-110.5 0-27.5 7.6-41.3 23.6-58.9-2.6-6.5-11.1-33.3 2.6-67.9 20.9-6.5 69 27 69 27 20-5.6 41.5-8.5 62.8-8.5s42.8 2.9 62.8 8.5c0 0 48.1-33.6 69-27 13.7 34.7 5.2 61.4 2.6 67.9 16 17.7 25.8 31.5 25.8 58.9 0 96.5-58.9 104.2-114.8 110.5 9.2 7.9 17 22.9 17 46.4 0 33.7-.3 75.4-.3 83.6 0 6.5 4.6 14.4 17.3 12.1C428.2 457.8 496 362.9 496 252 496 113.3 383.5 8 244.8 8z"/>
            </svg>
            {% elsif item.name == "LinkedIn" %}
            <svg viewBox="0 0 448 512" xmlns="http://www.w3.org/2000/svg">
              <path d="M416 32H31.9C14.3 32 0 46.5 0 64.3v383.4C0 465.5 14.3 480 31.9 480H416c17.6 0 32-14.5 32-32.3V64.3c0-17.8-14.4-32.3-32-32.3zM135.4 416H69V202.2h66.5V416zm-33.2-243c-21.3 0-38.5-17.3-38.5-38.5S80.9 96 102.2 96c21.2 0 38.5 17.3 38.5 38.5 0 21.3-17.2 38.5-38.5 38.5zm282.1 243h-66.4V312c0-24.8-.5-56.7-34.5-56.7-34.6 0-39.9 27-39.9 54.9V416h-66.4V202.2h63.7v29.2h.9c8.9-16.8 30.6-34.5 62.9-34.5 67.2 0 79.7 44.3 79.7 101.9V416z"/>
            </svg>
            {% endif %}
            <span>{{ item.username }}</span>
          </a>
        {% endif %}
      {% endfor %}
    </div>
  </div>
</header>
```

- [ ] **Step 3: Update `_includes/footer.html`**

```html
<footer class="site-footer">
  <div>&copy; {{ site.time | date: '%Y' }} {{ site.name | default: site.title }}</div>
  <div>
    <a href="{{ '/feed.xml' | prepend: site.baseurl }}">RSS Feed</a>
  </div>
</footer>
```

- [ ] **Step 4: Commit includes changes**

```bash
git add _includes/head.html _includes/header.html _includes/footer.html
git commit -m "feat(templates): update head, header, and footer includes to minimalist style"
```

---

### Task 3: Update Layouts (`default.html`, `post.html`, `page.html`, `archive.html`)

**Files:**
- Modify: `_layouts/default.html`
- Modify: `_layouts/post.html`
- Modify: `_layouts/page.html`
- Modify: `_layouts/archive.html`

- [ ] **Step 1: Update `_layouts/default.html`**

```html
<!DOCTYPE html>
<html lang="en">

  {% include head.html %}

  <body>
    <main class="site-container">
      {{ content }}
      {% include footer.html %}
    </main>
  </body>

</html>
```

- [ ] **Step 2: Update `_layouts/post.html`**

Add top back link `← Ahmed's Blog`, clean title, date, post body, and tag metadata.

```html
---
layout: default
---
<article class="post-view">
  <div class="back-nav">
    <a href="{{ '/' | prepend: site.baseurl }}" class="back-link">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
        <path d="M8.309 189.836L184.313 37.851C199.719 24.546 224 35.347 224 56.015v80.053c160.629 1.839 288 34.032 288 186.258 0 61.441-39.581 122.309-83.333 154.132-13.653 9.931-33.111-2.533-28.077-18.631 45.344-145.012-21.507-183.51-176.59-185.742V360c0 20.7-24.3 31.453-39.687 18.164l-176.004-152c-11.071-9.562-11.086-26.753 0-36.328z"/>
      </svg>
      <span>{{ site.title }}</span>
    </a>
  </div>

  <h1 class="post-title">{{ page.title }}</h1>
  
  <div class="post-date">
    <time datetime="{{ page.date | date_to_xmlschema }}">{{ page.date | date: "%B %-d, %Y" }}</time>
    {% if page.categories.size > 0 %}
      <span> &middot; {{ page.categories | join: ", " | upcase }}</span>
    {% endif %}
  </div>

  <div class="post-content">
    {{ content }}
  </div>
</article>
```

- [ ] **Step 3: Update `_layouts/page.html`**

```html
---
layout: default
---
<article class="post-view">
  <div class="back-nav">
    <a href="{{ '/' | prepend: site.baseurl }}" class="back-link">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
        <path d="M8.309 189.836L184.313 37.851C199.719 24.546 224 35.347 224 56.015v80.053c160.629 1.839 288 34.032 288 186.258 0 61.441-39.581 122.309-83.333 154.132-13.653 9.931-33.111-2.533-28.077-18.631 45.344-145.012-21.507-183.51-176.59-185.742V360c0 20.7-24.3 31.453-39.687 18.164l-176.004-152c-11.071-9.562-11.086-26.753 0-36.328z"/>
      </svg>
      <span>{{ site.title }}</span>
    </a>
  </div>

  <h1 class="post-title">{{ page.title }}</h1>

  <div class="post-content">
    {{ content }}
  </div>
</article>
```

- [ ] **Step 4: Update `_layouts/archive.html`**

```html
---
layout: default
---
<div class="post-view">
  <div class="back-nav">
    <a href="{{ '/' | prepend: site.baseurl }}" class="back-link">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
        <path d="M8.309 189.836L184.313 37.851C199.719 24.546 224 35.347 224 56.015v80.053c160.629 1.839 288 34.032 288 186.258 0 61.441-39.581 122.309-83.333 154.132-13.653 9.931-33.111-2.533-28.077-18.631 45.344-145.012-21.507-183.51-176.59-185.742V360c0 20.7-24.3 31.453-39.687 18.164l-176.004-152c-11.071-9.562-11.086-26.753 0-36.328z"/>
      </svg>
      <span>{{ site.title }}</span>
    </a>
  </div>

  <h1 class="post-title">{{ page.title }}</h1>

  <div class="archive-list" style="margin-top: 2rem;">
    {% for post in page.posts %}
    <div class="blog-list-item">
      <h2 class="post-item-title">
        <a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
      </h2>
      <div class="post-item-meta">
        <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%B %-d, %Y" }}</time>
      </div>
    </div>
    {% endfor %}
  </div>
</div>
```

- [ ] **Step 5: Commit layouts changes**

```bash
git add _layouts/default.html _layouts/post.html _layouts/page.html _layouts/archive.html
git commit -m "feat(layouts): update default, post, page, and archive layouts"
```

---

### Task 4: Redesign Homepage (`index.html`) & Configuration

**Files:**
- Modify: `index.html`
- Modify: `_config.yml`

- [ ] **Step 1: Update `_config.yml`**

Add `avatar: "/assets/avatar.png"` and ensure social accounts have accurate SVG handles.

- [ ] **Step 2: Update `index.html`**

Implement header, bio intro, post list, and clean pagination.

```html
---
layout: default
pagination:
  enabled: true
---
{% include header.html %}

<div class="bio-intro">
  <p>Hi, I'm Ahmed 👋</p>
  <p>Welcome to my technical blog. I specialize in Large Language Models (fine-tuning, RLHF/DPO, LoRA), Arabic NLP & Speech (TTS, acoustic modeling), and production autonomous AI agents. Here I share engineering deep dives, benchmarks, and production lessons.</p>
</div>

<div class="posts-feed">
  {% for post in paginator.posts %}
  <div class="blog-list-item">
    <h2 class="post-item-title">
      <a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
    </h2>
    <div class="post-item-meta">
      <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%B %-d, %Y" }}</time>
      {% if post.categories.size > 0 %}
        <span class="meta-separator">&middot;</span>
        <span class="meta-category">{{ post.categories | join: ", " | upcase }}</span>
      {% endif %}
    </div>
    <p class="post-item-excerpt">
      {{ post.content | strip_html | truncatewords: 35 }}
    </p>
  </div>
  {% endfor %}
</div>

{% if paginator.total_pages > 1 %}
<nav class="pagination" role="navigation">
  {% if paginator.previous_page %}
    {% if paginator.page == 2 %}
      <a class="newer-posts" href="{{ site.baseurl }}/">&larr; Newer Posts</a>
    {% else %}
      <a class="newer-posts" href="{{ site.baseurl }}{{ paginator.previous_page_path }}">&larr; Newer Posts</a>
    {% endif %}
  {% else %}
    <span class="disabled" style="opacity: 0.4;">&larr; Newer Posts</span>
  {% endif %}

  <span class="page-number">Page {{ paginator.page }} of {{ paginator.total_pages }}</span>

  {% if paginator.next_page %}
    <a class="older-posts" href="{{ site.baseurl }}{{ paginator.next_page_path }}">Older Posts &rarr;</a>
  {% else %}
    <span class="disabled" style="opacity: 0.4;">Older Posts &rarr;</span>
  {% endif %}
</nav>
{% endif %}
```

- [ ] **Step 3: Commit homepage and config changes**

```bash
git add index.html _config.yml
git commit -m "feat(home): update homepage feed, bio, and site configuration"
```

---

### Task 5: Automated Verification & Fidelity Testing

**Files:**
- Create: `scripts/verify_site.py`

- [ ] **Step 1: Write `scripts/verify_site.py`**

Test Jekyll YAML front matter, HTML file syntax, and SCSS variables presence.

```python
#!/usr/bin/env python3
import os
import sys
import yaml

def check_front_matter(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    if not content.startswith('---'):
        return True, "No front matter"
    parts = content.split('---', 2)
    if len(parts) < 3:
        return False, "Malformed front matter"
    try:
        yaml.safe_load(parts[1])
        return True, "Valid YAML"
    except Exception as e:
        return False, f"YAML error: {e}"

def test_files():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    errors = []
    
    # 1. Check YAML front matter
    files_to_check = [
        os.path.join(root, 'index.html'),
        os.path.join(root, '_config.yml'),
        os.path.join(root, '_layouts', 'default.html'),
        os.path.join(root, '_layouts', 'post.html'),
        os.path.join(root, '_layouts', 'page.html'),
        os.path.join(root, '_layouts', 'archive.html'),
        os.path.join(root, 'css', 'main.scss')
    ]
    for post in os.listdir(os.path.join(root, '_posts')):
        if post.endswith('.md'):
            files_to_check.append(os.path.join(root, '_posts', post))

    for fpath in files_to_check:
        if os.path.exists(fpath):
            ok, msg = check_front_matter(fpath)
            if not ok:
                errors.append(f"{fpath}: {msg}")

    # 2. Check SCSS files exist
    scss_files = [
        '_sass/_variables.scss',
        '_sass/_reset.scss',
        '_sass/_typography.scss',
        '_sass/_layout.scss',
        'css/main.scss'
    ]
    for rel in scss_files:
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            errors.append(f"Missing SCSS file: {rel}")

    # 3. Check CSS variables in _variables.scss
    var_file = os.path.join(root, '_sass', '_variables.scss')
    if os.path.exists(var_file):
        with open(var_file, 'r', encoding='utf-8') as f:
            vcontent = f.read()
        for expected in ['--bg-color', '--text-color', '--link-color', '--max-width', 'prefers-color-scheme: dark']:
            if expected not in vcontent:
                errors.append(f"_sass/_variables.scss missing token: {expected}")

    if errors:
        print("FAIL: Verification errors encountered:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("PASS: All site templates, front-matter, and stylesheets verified successfully!")

if __name__ == '__main__':
    test_files()
```

- [ ] **Step 2: Run verification script**

Run: `python3 scripts/verify_site.py`
Expected: `PASS: All site templates, front-matter, and stylesheets verified successfully!`

- [ ] **Step 3: Commit verification script**

```bash
git add scripts/verify_site.py
git commit -m "test: add verification script for Jekyll templates, front-matter, and SCSS"
```
