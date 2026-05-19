import pathlib

remaining = """

### 1.3 LM Studio (lmstudio.ai)

**URL:** https://lmstudio.ai
**Position:** Desktop-first local AI platform

**Color Palette and Design Language:**
- Pure white background, system font stack
- Hero screenshot of actual desktop app
- Minimal color

**Hero Section Structure:**
- Headline: "Run AI models, locally and privately."
- Model name buttons (gpt-oss, Qwen3, Gemma3, DeepSeek)
- Two CTAs: "Desktop App" and "Daemon"
- Direct download link + architecture chooser

**Feature Presentation:**
- Headless deployment, developer resources, API compatibility
- Install command snippets for Mac/Linux and Windows
- SDK examples (JS and Python)

**What Makes It Effective:**
1. App screenshot shows exactly what you get
2. Model name badges = instant compatibility signal
3. Dual-CTAs serve end-users and developers
4. Architecture selector for download
5. Mascot adds personality

---

### 1.4 vLLM (vllm.ai)

**URL:** https://vllm.ai
**Stars:** 80.5k | **Position:** Industry standard inference serving

**Color Palette and Design Language:**
- Pure white background, Inter font, JetBrains Mono for code
- Dark blue text, spacious layout
- Interactive install configurator
- Dark mode toggle

**Hero Section Structure:**
- Headline: "High-Throughput and Memory-Efficient inference"
- Subtitle: "Easy, fast, and cost-efficient"
- Two CTAs: "Get Started" and "Documentation"
- Three pillar cards: Easy | Fast | Cost Efficient

**Feature Presentation:**
- Three-pillar framework (memorable, scannable)
- Interactive install command generator with selectors
- "Universal Compatibility" with expandable lists
- Sponsor logos for institutional trust

**What Makes It Effective:**
1. Interactive install configurator is best-in-class UX
2. Three-pillar framework is memorable
3. Sponsor logos = trust
4. Hardware/model category links show breadth

---

### 1.5 SGLang (sglang.io)

**URL:** https://sglang.io
**Stars:** 28k | **Position:** High-performance serving (LMSYS)

**Color Palette and Design Language:**
- Clean white with warm tint
- Inter body, Space Grotesk headings, JetBrains Mono code
- Professional, enterprise-adjacent
- Interactive install configurator (vLLM pattern)

**Hero Section Structure:**
- Headline: "High-Performance Serving Framework"
- Three feature badges above fold
- "Get Started" + "See full list" dual-CTAs

**Feature Presentation:**
- Four-step "Get Started in Seconds"
- Interactive install configurator
- Community section with GitHub, Slack, Discord

**What Makes It Effective:**
1. Adopted proven vLLM install configurator pattern
2. Three feature badges as scannable bullets above fold
3. Clean typographic hierarchy

---

### 1.6 LLaMA-Factory (GitHub README)

**URL:** https://github.com/hiyouga/LLaMA-Factory
**Stars:** 71.4k | **Position:** Most popular fine-tuning framework

**README Patterns:**
- Centered logo + badges row
- Video demo embed (extremely effective)
- "Used by Amazon, NVIDIA, Aliyun" — social proof logos
- TOC for navigation through long README
- Model-specific performance table
- Colab "Start for free" buttons

**What Makes It Effective:**
1. Video in README shows instant value
2. "Used by" logos build institutional credibility
3. Colab buttons lower friction to zero
4. TOC makes long README navigable

---

### 1.7 llama.cpp (GitHub README)

**URL:** https://github.com/ggml-org/llama.cpp
**Stars:** 111k | **Position:** OG local LLM inference

**README Patterns:**
- Minimalist — logo, badges, "LLM inference in C/C++"
- "Hot topics" section for dynamic content
- Quick start with multiple install methods
- Massive model support checklist
- Links to discussions

**What Makes It Effective:**
1. "Hot topics" surfaces what is new
2. Multiple install paths shown
3. Model support checklist is aspirational
4. Pure technical content = trust through transparency

---

## 2. CROSS-CUTTING DESIGN PATTERNS

### 2.1 Hero Section Patterns

| Pattern | Used By | Effectiveness |
|---------|---------|---------------|
| Install command in hero | Ollama, vLLM, SGLang | Very high for devs |
| App screenshot in hero | Unsloth, LM Studio | High |
| Tagline + dual CTA | All sites | Standard |
| Model name badges | LM Studio | Shows compatibility |
| Interactive configurator | vLLM, SGLang | Best-in-class |
| Announcement chip | Unsloth, LM Studio | Good |

### 2.2 Feature Presentation

| Pattern | Used By |
|---------|---------|
| Alternating image+text cards | Unsloth |
| Three-pillar framework | vLLM |
| Icon + heading + description | Replicate |
| Code snippets as features | Ollama, LM Studio |

### 2.3 Performance/Benchmark

| Pattern | Used By |
|---------|---------|
| Specific comparison numbers | Unsloth |
| Comparison tables | LLaMA-Factory |
| Pill badges with stats | Unsloth |
| Research paper links | vLLM, SGLang |

### 2.4 CTA Patterns

| Pattern | Used By |
|---------|---------|
| "Start for free" | Unsloth, Replicate |
| "Download" | Ollama, LM Studio |
| "Get Started" | vLLM, SGLang |
| "Launch Studio" | Unsloth |
| "Quickstart" | Unsloth, LLaMA-Factory |

---

## 3. DESIGN CRITIQUE OF CURRENT MOJOLLAMA SITE

### Current Strengths:
1. Clean dark theme — distinctive from white competitor pages
2. Benchmark table with real numbers
3. Direct Studio + Chat links
4. "Connect to Live Server" shows it is real/running
5. Strong headline "Inference at the speed of Mojo"

### Current Gaps:
1. No hero install command
2. No product UI screenshots
3. No social proof (GitHub stars, Discord count, user logos)
4. No competitor comparison in benchmarks
5. Single CTA style — no secondary options
6. Flat feature list, not benefit-focused
7. Sparse navigation — no Docs, Blog, GitHub link
8. No pricing or license info
9. No community signup
10. Features lack dual-CTAs

---

## 4. ACTIONABLE RECOMMENDATIONS

### Priority 1: Hero Redesign
- Keep "Inference at the speed of Mojo"
- Add subtitle with positioning
- Show install command
- Dual-CTAs: "Launch Studio" + "Quickstart"
- GitHub star + Discord in nav

### Priority 2: Product Screenshots
- Studio chat UI, benchmark dashboard, fine-tuning config
- Authentic screenshots, not mockups

### Priority 3: Restructure Features
- Alternating image+text cards
- Each card: Icon + Heading + Benefit + "Quickstart" + "Learn More"

### Priority 4: Benchmark Upgrade
- Color-code Mojo backend rows
- Add competitor speedup percentages
- Visual bar charts
- "How we benchmark" methodology link

### Priority 5: Social Proof
- GitHub star count, Discord count
- "Used by" logos (early adopters)
- "Built with Mojo" badge

### Priority 6: Interactive Install Configurator
- Platform, backend, install method selectors
- Output exact command

### Priority 7: Typography and Layout
- Space Grotesk headings, Inter body, JetBrains Mono code
- More whitespace, consistent rhythm
- Light/dark mode toggle

### Priority 8: Navigation
- Add: Docs | Blog | GitHub | Discord | Models
- "Launch Studio" CTA in nav
- "Star on GitHub" badge

### Priority 9: Footer
- Product, Community, Legal links + Newsletter signup

### Priority 10: Mobile Responsiveness

---

## 5. CSS PATTERNS

### Typography
```css
--font-heading: 'Space Grotesk', system-ui, sans-serif;
--font-body: 'Inter', system-ui, -apple-system, sans-serif;
--font-mono: 'JetBrains Mono', monospace;
```

### Color Palette
```css
--primary: #6C3CE1;       /* Purple */
--primary-light: #8B5CF6;
--accent: #22D3EE;         /* Cyan */
--bg-dark: #0F172A;
--bg-card: #1E293B;
--text-primary: #F8FAFC;
--text-secondary: #94A3B8;
--success: #34D399;
```

### Button Patterns
- Primary: Solid brand color, rounded 8-12px, weight 600
- Secondary: Outlined/border variant
- Install: Terminal-themed, dark bg, monospace, green/cyan accent
- GitHub star: Inline badge with icon + count

### Nav
Fixed top, backdrop blur. Left: logo | Center: links | Right: GitHub + CTA

### Feature Cards
Two-column grid, alternating image side with nth-child

---

## 6. COPYWRITING FRAMEWORK

### Headline Options:
1. "Inference at the speed of Mojo" (keep — it is strong)
2. "CPU-first LLM inference, built in Mojo"
3. "The open-source LLM platform engineered in Mojo"
4. "MojoLlama: Inference, fine-tuning, and serving in Mojo"

### Value Propositions:
- Developers: "Ship LLM features faster. CPU-first inference, GGUF quantization, concurrent serving."
- ML Engineers: "Fine-tune and deploy on any hardware. No GPU required."
- OSS Enthusiasts: "Mojo-powered alternative to Unsloth and LLaMA-Factory."

### CTA Copy:
| Context | Copy |
|---------|------|
| Hero primary | "Launch Studio" |
| Hero secondary | "Quickstart" |
| Install | "Install in 1 minute" |
| Docs | "Read the Docs" |
| GitHub | "Star on GitHub" |
| Community | "Join our Discord" |

---

## 7. KEY DIFFERENTIATORS

1. Mojo language — built in Mojo+MAX (unique among competitors)
2. CPU-first — runs efficiently on CPU, not just GPU
3. GGUF native — optimized GGUF quantization
4. Concurrent serving — built-in, not add-on
5. All-in-one — inference, fine-tuning, quantization, serving, benchmarking
6. Fully open source — vs. Unsloth proprietary features

---

## 8. COMPETITOR LANDSCAPE

GPU-focused: vLLM, SGLang, LLaMA-Factory, Unsloth
CPU-focused: llama.cpp, Ollama, LM Studio
All-in-one: MojoLlama (unique intersection)

MojoLlama occupies a strong, defensible position at the intersection of CPU-first and all-in-one that no competitor fully occupies.

---

Report compiled from direct research of 8+ competitor sites.
All URLs verified as of May 19, 2026.
"""

pathlib.Path('/onedev-workspace/work/competitor-design-research.md').write_text(
    pathlib.Path('/onedev-workspace/work/competitor-design-research.md').read_text() + remaining
)
print(f"File updated. Total size: {len(pathlib.Path('/onedev-workspace/work/competitor-design-research.md').read_text())} bytes")
