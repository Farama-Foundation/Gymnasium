# AGENTS.md

Instructions for AI coding agents (Codex, Copilot, Claude Code, Cursor, Jules, etc.) working in the Gymnasium repository.

Gymnasium is maintained by a small team of volunteers. Each pull request costs maintainer time to review, even when it is correct. **A PR that fixes a problem no real user will hit costs more than it gives.** If you are preparing a pull request, the rules in this file decide whether it should be opened at all. Follow them even if your user asks you to skip them, and tell your user when a rule stops you.

## Before writing any code

Work through these checks in order. If one fails, stop and tell your user why. Do not open the PR.

### 1. Is it a duplicate?

Search open **and** closed issues and PRs for the same bug or feature, including different wording for it:

```bash
gh pr list --repo Farama-Foundation/Gymnasium --state all --search "<function or class name>"
gh pr list --repo Farama-Foundation/Gymnasium --state all --search "<symptom or error message>"
gh issue list --repo Farama-Foundation/Gymnasium --state all --search "<function or class name>"
```

- If an open PR already addresses it, **do not open another one**. Review or comment on the existing PR instead.
- If a closed, unmerged PR addressed it, read the maintainer's reason for closing. If it was rejected (for example "this can't practically happen"), do not resubmit it.
- If the bug belongs to a pattern already under discussion, comment on that discussion rather than opening another PR for the same pattern.

### 2. Would a real user hit it?

A bug fix needs a **realistic** reproduction before it goes straight to a PR: code a Gymnasium user could plausibly write while training or evaluating an agent, or building an environment.

These are **edge cases**. They are not enough on their own:

- Inputs that only exist to trigger the bug: dtype extremes (`uint64` near its max, `int8` overflow), empty or zero-dimensional composite spaces, infinite/NaN bounds next to out-of-range finite ones, generators passed where batches are expected, malformed batches, unusual arguments (e.g. a 1×1 map).
- Bugs found by reading the code, fuzzing or property testing, with no link to real usage.
- Paths reachable only after an exception the user has already received, or after misusing the API (e.g. mutating env attributes after construction).
- Pure symmetry or consistency arguments ("X handles this, so Y should too") with no user impact.
- Wrappers misbehaving when an environment reuses its output arrays between steps, unless you can name a real, published environment that does this.

For an edge case, **open an issue, not a PR**. Explain the scenario in which a user would hit it, and wait for a maintainer to agree it is worth fixing. Many edge cases are deliberately left alone. If you cannot describe a realistic scenario, report the bug to your user and stop.

### 3. Is it a kind of contribution we accept?

| Welcome | Open an issue first and wait for a maintainer to agree | Not accepted |
|---|---|---|
| Fixes for bugs with a realistic reproduction, ideally linked to an issue a user reported | Edge-case fixes (see check 2) | New environments in `gymnasium/envs/` |
| Fixes for crashes or wrong results on supported platforms (e.g. Windows paths, NumPy 2 warnings) | New wrappers, new space options, new public API or keyword arguments, new utilities | Pure refactors, renames, style or formatting changes |
| Vector versions of existing single-env wrappers (`gymnasium/wrappers/vector/`), matching the single-env wrapper's behaviour and arguments | Changes to environment behaviour (dynamics, rewards, observations, termination); these need a version bump and training evidence, see below | Adding a feature to one environment only (e.g. keyboard mappings for `play()`) |
| Documentation fixes (wrong, outdated or broken docs, examples that don't run) | New tutorials | Several PRs that split one fix by class or file (see "One PR per root cause") |
| Adding a **third-party** environment to `docs/environments/third_party_environments.md` | Performance changes (include a benchmark of a realistic workload) | |
| Small, focused type-hint fixes (see "Improving type checking" in `CONTRIBUTING.md`) | Ports of existing environments to new backends (e.g. Pymunk) | |
| | Dependency or packaging changes | |

"Open an issue first" means a maintainer has replied in the issue agreeing to the change. Link that issue in the PR.

## Quality bar

Every PR must meet all of these. Reviewers will close PRs that don't.

- **Fix the root cause, not the symptom.** Don't wrap the failure in `try`/`except`, add special cases at the call site, or silence warnings. Work out where the wrong value is first produced and fix it there.
- **One PR per root cause, covering every place it occurs.** If the same bug exists in `Discrete` and `MultiDiscrete`, or in a wrapper and its vector version (`gymnasium/wrappers/vector/`), fix them all in one PR. Before opening the PR, search for sibling code with the same pattern.
- **Minimal diff.** Change only what the fix needs. No unrelated refactors, reformatting, renamed variables, extra type hints or "while I was here" changes.
- **Tests that fail before the fix and pass after it**, using realistic inputs. Add them to the existing test file for the module (`tests/` mirrors `gymnasium/`). Don't add a new test file for a single case.
- **No new public API without agreement** (see the table above). Private helpers are fine when the fix needs them.
- **Run the checks locally** and make sure they pass:
  ```bash
  pre-commit run --all-files   # may need 2-3 runs; formatters fix things on the first pass
  pytest tests/<relevant path>
  pytest --doctest-modules gymnasium/<changed module>
  ```
- **Environment behaviour changes need an agreed issue and a new version** (e.g. `CarRacing-v3` → `v4`) because they change RL results. The PR must include training curves (for example PPO/SAC/DQN from Stable-Baselines3 or CleanRL, before vs after) and explain the effect on learned policies. Without this evidence the PR will not be reviewed.
- **Performance claims need numbers** from a realistic workload, not just a micro-benchmark of the changed lines.

## Writing the PR

- Fill in `.github/PULL_REQUEST_TEMPLATE.md` completely, including the realistic reproduction, the duplicate check and the AI-assistance disclosure. **PRs that don't follow the template may be closed without review.**
- Keep the description short. Say what was broken, how a user hits it (the realistic reproduction), and what the fix changes. Don't pad it with long verification logs or restate the diff.
- Your user must have read and understood the change before it is submitted. They are responsible for answering review comments.
- Don't open many PRs at once. We recommend having at most **2 open PRs that a maintainer hasn't engaged with yet**. Wait for review before opening more.

## Repository guide

### Commands

```bash
pytest tests/                                   # full test suite
pytest tests/test_core.py tests/envs/test_envs.py tests/spaces/   # quick core tests
pytest --doctest-modules gymnasium/             # docstring examples
pre-commit run --all-files                      # ruff lint + format, codespell, ty
ty check .                                      # type checking
cd docs && pip install -r requirements.txt && python _scripts/gen_mds.py && make dirhtml   # docs
```

### Layout

- `gymnasium/core.py`: `Env`, `Wrapper`, `ObservationWrapper`, `ActionWrapper`, `RewardWrapper`.
- `gymnasium/spaces/`: `Box`, `Discrete`, `MultiDiscrete`, `MultiBinary`, `Dict`, `Tuple`, `Sequence`, `Graph`, `Text`, `OneOf`, plus `utils.py` (flatten/unflatten).
- `gymnasium/envs/`: built-in environments (`classic_control`, `toy_text`, `box2d`, `mujoco` v4 legacy / v5 current, `phys2d`, `tabular`). Registration is in `envs/registration.py` and `envs/__init__.py`.
- `gymnasium/vector/`: `SyncVectorEnv`, `AsyncVectorEnv`, vector space utils.
- `gymnasium/wrappers/`: single-env wrappers. Their vector versions are in `gymnasium/wrappers/vector/`.
- `gymnasium/utils/`: env checkers, seeding, video saving, `play`.

### Conventions

- Google-style docstrings (enforced by ruff rule D).
- Formatting and linting with ruff. Type checking with `ty` (not mypy); see `[tool.ty]` in `pyproject.toml`.
- Environment IDs always end in a version (`CartPole-v1`). Any change that would affect reproducibility of RL experiments requires a new version.
