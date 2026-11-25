<!--
Before opening this PR, read CONTRIBUTING.md (and AGENTS.md if you used an AI tool).
PRs that skip the checks below, or that fix problems no real user would hit, will be closed without a detailed review.
-->

# Description

What was broken or missing, and what this PR changes. Keep it short.

Fixes # (issue)

## How a user hits this

For bug fixes: a short, realistic code snippet a Gymnasium user could plausibly write that shows the problem.
If the bug only appears with inputs constructed to trigger it (dtype extremes, empty composite spaces, malformed batches, etc.), open an issue first and link the maintainer's agreement here.

```python

```

## Type of change

Please delete options that are not relevant.

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] Documentation only change (no code changed)
- [ ] Third-party environment added to the docs
- [ ] New feature (link to the issue where a maintainer agreed to it)
- [ ] Environment behaviour change (requires a new environment version and training curves, before vs after)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)

### Screenshots

Please attach before and after screenshots of the change if applicable.

<!--
Example:

| Before | After |
| ------ | ----- |
| _gif/png before_ | _gif/png after_ |


To upload images to a PR -- simply drag and drop an image while in edit mode and it should upload the image directly. You can then paste that source into the above before/after sections.
-->

# Checklist:

- [ ] I have searched open **and** closed PRs and issues, and this is not a duplicate (link any related ones above)
- [ ] The fix addresses the root cause, and covers every sibling occurrence (other spaces, the vector version of the wrapper, etc.) in this one PR
- [ ] The diff contains only changes needed for this fix (no unrelated refactors, formatting or type hints)
- [ ] I have added tests that fail without this change and pass with it
- [ ] I have run the [`pre-commit` checks](https://pre-commit.com/) with `pre-commit run --all-files` (see `CONTRIBUTING.md` instructions to set it up)
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have made corresponding changes to the documentation

## AI assistance

- [ ] No AI tools were used
- [ ] AI tools were used (name them: ______). I have read and understood every line of this change and can answer review questions about it myself

<!--
As you go through the checklist above, you can mark something as done by putting an x character in it

For example,
- [x] I have done this task
- [ ] I have not done this task
-->
