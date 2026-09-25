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
    posts_dir = os.path.join(root, '_posts')
    if os.path.exists(posts_dir):
        for post in os.listdir(posts_dir):
            if post.endswith('.md'):
                files_to_check.append(os.path.join(posts_dir, post))

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
