# MojoLlama CODING-SOUL

## Identity

MojoLlama is a hybrid LLM inference engine built with Mojo + MAX.
It runs on CPU (AVX2/AVX512/NEON), GPU (CUDA/SYCL/Vulkan), and hybrid
setups — competing with vLLM/SGLang while also running on machines without
a discrete GPU.

No Python in the hot path. No hand-rewriting kernels for every model.
The architecture IS the advantage.

## Principles

### 1. Mojo + MAX, not C + llama.cpp

We use Mojo and the MAX ecosystem. When Mojo's heap/pointer APIs
mature in nightly, we adopt them. When MAX becomes installable, we
integrate its graph compiler and GPU kernels.

Until then, the Python backend (bridge.py) is a placeholder — not a
compromise. Every line of Python is temporary, clearly marked, and
designed to be swapped for Mojo without changing the op graph.

### 2. Ops, not models

An `AttentionOp` is an `AttentionOp`. It doesn't know or care whether
it's part of Llama, AttnRes, MLA, or something unreleased.

NEW ARCHITECTURE = new graph wiring in `graph/ops.mojo`.
No new kernels. No new backends. No new anything except the DAG.

This is why we beat the vLLM/llama.cpp approach: they write kernels
per architecture. We write ops once, compose them differently.

### 3. The graph is the source of truth

The computation is defined in `graph/ops.mojo`:
```
MatmulOp, AttentionOp, RMSNormOp, RoPEOp, SiLUOp
AddOp, EmbedOp, TransformerBlock
```

These compile in Mojo 0.26.2 with zero errors. They are the invariant.
Backends are swappable: Python → Mojo SIMD → MAX GPU → whatever comes next.

No runtime type dispatch. No `if model_type == "llama"`. The graph
structure IS the model.

### 4. Every line of SIMD is deliberate

The Q4_0 kernel (`kernels/q4_matmul.mojo`) is hand-written for AVX2.
Not because we couldn't use BLAS — because we want the ASM to be
exactly what we specify.

Every `SIMD[DType.float32, 8]` is an explicit 256-bit vector register.
Every `reduce_add()` is a horizontal sum. Every nibble extraction is
a sequence of `& 15`, `>> 4`, `- 8` that maps directly to AVX2
instructions. No surprises, no compiler guessing.

When VNNI/AVX512/CUDA are available, `comptime if` dispatches.
The code changes at compile time, not runtime.

### 5. Future architectures are graph changes, not kernel changes

Attention Residuals: one new op (`DepthAttentionOp`), same backends.
MLA: one new op (`LatentAttentionOp`), same backends.
Whatever comes next: same pattern.

The backends (CPU SIMD, CUDA, Vulkan, SYCL) are stable. The graph evolves.
This is how we stay ahead of llama.cpp and vLLM.

## Non-Negotiable

- Mojo stdlib imports only. No Python in the forward pass.
- The `ops.mojo` graph compiles in current Mojo stable.
- Every kernel has a test with deterministic expected values.
- No hand-writing CUDA kernels for new architectures — the graph compiler handles it.

## What We Ship When

"Works on my machine" is not the standard. The standard is:
- AVX2 today (our dev box)
- AVX512 tomorrow (server-class)
- CUDA/SYCL/Vulkan when MAX is installable
- Any graph architecture (Llama, AttnRes, MLA, ...) = same binary, different input

## The Bet

We're betting that Mojo's pointer APIs stabilize before MAX becomes
installable, and that MAX's graph compiler beats hand-written kernels.
History suggests this is right: specialized kernels don't scale to
50+ architectures. A graph compiler does.

If we're wrong, we rewrite the backends in C. The ops stay.

---

# MojoLlama Brand Guidelines

## Brand Identity

MojoLlama is positioned at the intersection of:
- **CPU-first** — runs efficiently without a GPU (unlike vLLM/SGLang)
- **All-in-one** — inference, fine-tuning, quantization, serving (unlike llama.cpp/Ollama)
- **Mojo+MAX native** — engineered in Mojo, not a C/Python wrapper
- **Open source** — Apache 2.0, no proprietary features

### Tagline
**"Inference at the speed of Mojo"**

Alternative taglines for different contexts:
- "CPU-first LLM inference, engineered in Mojo+MAX"
- "The open-source LLM platform built in Mojo"
- "Fine-tune, quantize, and serve — all on CPU"

### Brand Voice
- Technical but approachable — we speak to ML engineers and developers
- Confident, not arrogant — back claims with real benchmarks
- Generous — open source, no lock-in, free forever
- Mojo-first — we're proud of the technology stack, not hiding it

## Color Palette

### Primary (Mojo Purple)
| Token | Hex | Usage |
|-------|-----|-------|
| `--mojo` | `#6C3CE1` | Primary brand color, buttons, links |
| `--mojo-light` | `#8B5CF6` | Hover states, highlights |
| `--mojo-glow` | `rgba(108,60,225,0.3)` | Glow effects, shadows |

### Accent (Cyan)
| Token | Hex | Usage |
|-------|-----|-------|
| `--cyan` | `#22D3EE` | Secondary accent, code syntax, highlights |
| `--cyan-glow` | `rgba(34,211,238,0.25)` | Glow effects |

### Semantic
| Token | Hex | Usage |
|-------|-----|-------|
| `--green` | `#34D399` | Success, benchmarks highlight |
| `--pink` | `#E879F9` | Feature accent, creative sections |
| `--orange` | `#FB923C` | Warning, speed accent |
| `--red` | `#F87171` | Error, danger |

### Backgrounds
| Token | Hex | Usage |
|-------|-----|-------|
| `--bg` | `#080B12` | Main background (darkest) |
| `--bg2` | `#0C101A` | Section backgrounds |
| `--surface` | `#111624` | Card/panel surfaces |
| `--surface2` | `#171D2E` | Hover states |
| `--surface3` | `#1E2540` | Active states, code blocks |
| `--border` | `#252D4A` | Borders, dividers |
| `--border2` | `#1A2238` | Subtle borders |

### Text
| Token | Hex | Usage |
|-------|-----|-------|
| `--text` | `#E2E8F0` | Primary text |
| `--text2` | `#C8D0DC` | Secondary text |
| `--dim` | `#7F8EA3` | Muted text |
| `--dim2` | `#5A6A82` | Subtle/meta text |

### Gradient Signature
`linear-gradient(135deg, #6C3CE1 0%, #8B5CF6 40%, #22D3EE 100%)`

This is the MojoLlama gradient — used for logo text, hero headings, primary buttons.

### Gradient Accents
- **Success**: `linear-gradient(135deg, #22D3EE 0%, #34D399 100%)`
- **Warm**: `linear-gradient(135deg, #FB923C 0%, #F87171 100%)`

## Typography

| Context | Font | Weights |
|---------|------|---------|
| Headings | Space Grotesk | 600, 700, 800, 900 |
| Body | Inter | 300, 400, 500, 600, 700 |
| Code | JetBrains Mono | 400, 500, 600 |

### CSS Variables
```css
--font-heading: 'Space Grotesk', system-ui, sans-serif;
--font-body: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
--font-mono: 'JetBrains Mono', 'Fira Code', monospace;
```

## Logo / Icon

The MojoLlama logo mark is the llama emoji 🦙 on a Mojo gradient background.
- The llama represents the "Llama" in MojoLlama
- The gradient background (mojo purple → cyan) represents the "Mojo" part
- Always use on dark backgrounds

### Logo Usage
- **Navbar**: 🦙 + "MojoLlama" in gradient text
- **Footer**: 🦙 + "MojoLlama" in gradient text
- **Favicon**: 🦙 on a rounded mojo gradient square

## UI Patterns

### Buttons
- **Primary**: Solid mojo gradient, `border-radius: 10px`, font-weight 700, lift on hover
- **Secondary**: Dark surface with border, subtle hover with mojo-light border
- **Install/Code**: Terminal-themed, dark background, cyan text, monospace
- **Ghost**: Border-only, used in nav

### Cards
- Dark surface (`--surface`) with subtle border (`--border`)
- Hover: lift 2px, slight glow, top gradient border reveal
- Icon in a colored circle matching accent type

### Navigation
- Fixed top, backdrop blur, border-bottom
- Left: logo | Right: links + CTA button
- GitHub badge: inline star count

### Install Block
- Terminal window mockup with traffic light dots
- Green prompt `$`, cyan commands, dim output
- "Copy All" button in the footer bar

## Component Hierarchy (Landing Page)

1. **Navbar** — Logo, Studio, Chat, GitHub stars, Launch Studio CTA
2. **Hero** — Badge (open source), Headline, Subtitle, Feature checks, CTAs, Install block
3. **Features** — 2-column grid of feature cards with dual CTAs
4. **Benchmarks** — Dark section with real benchmark table + visual bars
5. **Getting Started** — 3-step numbered cards
6. **Open Source** — License badges, GitHub star CTA
7. **Footer** — 4-column grid: brand + Product + Community + Ecosystem

## Key Differentiators (Messaging)

Always lead with these when describing MojoLlama:
1. **Mojo language** — built in Mojo+MAX (unique among all competitors)
2. **CPU-first** — runs efficiently on CPU, not just GPU
3. **GGUF native** — optimized GGUF quantization pipeline
4. **Concurrent serving** — built-in, not add-on
5. **All-in-one** — inference, fine-tuning, quantization, serving, benchmarking
6. **Fully open source** — Apache 2.0, no proprietary features

## Competitor Positioning

| Competitor | MojoLlama Advantage |
|------------|---------------------|
| vLLM | CPU-first, Mojo engine, all-in-one (not just serving) |
| SGLang | CPU-first, Mojo engine, simpler architecture |
| Unsloth | Fully open source, Mojo-native, no GPU required |
| LLaMA-Factory | CPU-first benchmarks, concurrent serving built-in |
| Ollama | Open source, programmable, fine-tuning pipeline |
| llama.cpp | Mojo engine, concurrent serving, all-in-one platform |

## Social Proof Patterns

- GitHub star count in navbar and hero
- "Built with Mojo+MAX" badge
- Benchmark comparison tables (Mojo vs competitors)
- Real hardware specs in benchmarks (AMD Threadripper 3970X)
- Apache 2.0 license badge

## Website Structure (www/)

```
www/
├── index.html     # Landing page (sellable, branded)
├── studio.html    # MojoLlama Studio UI (training/quantization dashboard)
├── chat.html      # Chat interface
└── ...            # Additional pages as needed
```

## Build Spec

The `.onedev-buildspec.yml` deploys `www/` content to OneDev's static site at:
`https://git.bamse.cloud/a730/MojoLlama/~site/`

The PublishSiteStep must come BEFORE any container steps (container
isolation loses the checked-out files). The buildspec is structured as:
1. Build Kernels (job) — compiles C AVX2 kernels in parallel
2. Deploy Site (job) — publishes www/ to ~site, depends on Build Kernels
3. Quick Test (job) — fast validation for dev branches
