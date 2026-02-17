# Contributing

Thanks for contributing to the Lattice QCD Flux-Tube Program.

## Where to Discuss

- Antfarm room (lattice-qcd): <https://antfarm.world/messages/room/lattice-qcd>
- API join endpoint: `POST https://antfarm.world/api/v1/rooms/lattice-qcd/join`

Use the room for experiment notes, implementation proposals, and coordination before opening large PRs.

## Development Setup

1. Clone the repo.
2. Install Python requirements:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
3. Run checks:
   - `make test`
   - `make lint`

## Pull Request Expectations

- Keep changes focused and physics-motivated.
- Add or update tests for behavior changes.
- Avoid breaking existing CLI flags and output formats unless explicitly discussed.
- Include a short validation note in the PR description (what you ran and what changed).

## Code Areas

- Project-specific measurement logic: `gpt/applications/hmc/`
- Analysis/postprocessing/dashboard tools: `tools/`
- Vendored upstream frameworks:
  - `Grid/`
  - `gpt/`

For changes inside vendored upstream components, keep patches minimal and clearly documented in your PR.
