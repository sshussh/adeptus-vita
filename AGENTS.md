# AGENTS.md

Adeptus Vita is an AI-powered Alzheimer's/dementia diagnosis platform. Users upload brain MRI scans via a Next.js web app, which calls a Python ML backend using TensorFlow and EfficientNetB0 to classify images into Non-Demented, Mild Demented, or Moderate Demented categories.

## Build/Run Commands (web-app/)
- `pnpm dev` - Development server with Turbopack
- `pnpm build` - Production build
- `pnpm lint` - Run ESLint
- Python API requires `uv` in `web-app/app/api/` with Python 3.10+

## Code Style

### TypeScript/React
- Strict TypeScript enabled; use `import type` for type-only imports
- Path alias: `@/*` maps to project root
- Components: PascalCase, files: kebab-case, hooks: `use` prefix (camelCase)
- Use `"use client"` directive only when needed; prefer server components
- UI: shadcn/ui + Tailwind CSS; use `cn()` from `@/lib/utils` for class merging
- Icons: lucide-react only
- Error handling: try/catch with `console.error`, return proper HTTP status codes in API routes

### Python (web-app/app/api/)
- Functions: snake_case, constants: UPPER_SNAKE_CASE
- Docstrings with Parameters/Returns sections
- Use `# type: ignore` for Keras imports if needed
- Print errors to stderr: `print(f"Error: {e}", file=sys.stderr)`

## Project Structure
- `web-app/` - Next.js 15 app with App Router
- `web-app/app/api/` - Python ML inference (TensorFlow, EfficientNetB0)
- `dataset/` - Training data (AlzheimerDataset, ResampledDataset)
- `improved-ann-model/` - ANN model notebooks
