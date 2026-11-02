# Repository protections

Configuration for `gboddaer/strateu`. JSON files are API payloads; committing them
alone does not change live GitHub settings. Use `gh api` to apply them.

## Main branch policy

- `main-safety` blocks deletion and force pushes. It has **no bypass actors**, so
  these restrictions also apply to the owner.
- `owner-approval` requires a PR, one approval including the code owner, fresh
  approval after changes, approval from someone other than the last pusher,
  resolved conversations, and the seven checks below against an up-to-date branch.
- `.github/CODEOWNERS` assigns every path to `@gboddaer`, including this configuration.
- The repository admin role (GitHub role ID 5) has an **always** bypass on
  `owner-approval`. This is intentional: the owner can make normal direct pushes
  to main without a PR or passing CI, and can merge their own PRs.
- There is currently one admin, `gboddaer`. Adding another admin gives that account
  the same bypass. Other contributors do not get this exception. Ordinary public
  fork contributors cannot push to the repository at all.
- An admin can change rulesets themselves. These rules are not an immutable lock
  against a compromised owner account; protect that account with strong MFA.

Required checks, tied to the GitHub Actions App (integration ID 15368):

1. `title`
2. `node-test`
3. `Linux / runtime requirements only`
4. `ubuntu-latest / Python 3.10`
5. `ubuntu-latest / Python 3.11`
6. `windows-latest / Python 3.10`
7. `windows-latest / Python 3.11`

These workflows run on every PR without path filters. A rename requires updating
this policy; an unavailable check blocks non-bypassing contributors. Metal is
still conditional and is not a required check. These checks do not qualify GPU
runtime correctness or replace engine accuracy gates.

## Actions and security policy

- Default workflow token permissions: read; workflows cannot approve PRs.
- All external contributors need approval for fork PR workflow runs. This does
  not protect `pull_request_target` or other privileged events: the intake/reopen
  workflows must continue to use trusted default-branch scripts only, never fork code.
- Selected Actions only: checkout, setup-node, setup-python and github-script
  under the `actions` organization. Full commit-SHA pinning is required. Existing
  action versions are pinned without changing their major versions.
- All workflow checkouts disable credential persistence. Privileged jobs still
  explicitly request the permissions they need.
- Dependabot alerts and automatic security-update PRs are enabled. Dependabot
  cannot bypass owner review; merging its updates is still the owner's decision.
- Secret scanning, push protection and private vulnerability reporting remain enabled.
- Action restrictions do not pin tools downloaded by shell commands or Python/npm
  dependencies. Review those supply chains separately.
- Tags, releases, other branches and source-history privacy are outside these rulesets.

## Apply or update

From the repository root, authenticated as the repository admin. Review the
payloads first. These commands do not change repository visibility.

Find existing ruleset IDs:

```sh
gh api repos/gboddaer/strateu/rulesets --jq '.[] | {id,name,enforcement}'
```

At initial activation: `main-safety` was ID `26201611` and `owner-approval` was
ID `26201578`. Verify IDs before updating. For a new repository use POST once;
repeated POSTs create duplicates. For this repository update with PUT:

```sh
gh api --method PUT repos/gboddaer/strateu/rulesets/26201611 \
  --input .github/rulesets/main-safety.json
gh api --method PUT repos/gboddaer/strateu/rulesets/26201578 \
  --input .github/rulesets/owner-approval.json
```

Land SHA-pinned workflows **before** turning on SHA enforcement. Set selected
mode first, then its allowlist. Do not dispatch workflows in between these calls.

```sh
gh api --method PUT repos/gboddaer/strateu/actions/permissions \
  --input .github/rulesets/actions-permissions.json
gh api --method PUT repos/gboddaer/strateu/actions/permissions/selected-actions \
  --input .github/rulesets/actions-selected.json
gh api --method PUT repos/gboddaer/strateu/actions/permissions/fork-pr-contributor-approval \
  -f approval_policy=all_external_contributors
gh api --method PUT repos/gboddaer/strateu/vulnerability-alerts
gh api --method PUT repos/gboddaer/strateu/automated-security-fixes
```

On a fresh repository, keep the existing safe token settings or set them with:

```sh
gh api --method PUT repos/gboddaer/strateu/actions/permissions/workflow \
  -f default_workflow_permissions=read -F can_approve_pull_request_reviews=false
```

## Verify

```sh
gh api repos/gboddaer/strateu/rulesets/26201611
gh api repos/gboddaer/strateu/rulesets/26201578
gh api repos/gboddaer/strateu/rules/branches/main
gh api 'repos/gboddaer/strateu/codeowners/errors?ref=main'
gh api repos/gboddaer/strateu/collaborators --jq '.[] | {login,role_name}'
gh api repos/gboddaer/strateu/actions/permissions
gh api repos/gboddaer/strateu/actions/permissions/selected-actions
gh api repos/gboddaer/strateu/actions/permissions/fork-pr-contributor-approval
gh api repos/gboddaer/strateu/actions/permissions/workflow
gh api repos/gboddaer/strateu/automated-security-fixes
```

Both rulesets must be active and target `refs/heads/main`. The current owner should
have `current_user_can_bypass: always` on owner-approval and `never` on main-safety.
CODEOWNERS should have no errors. Check Actions results after changing pins.
The classic branch-protection endpoint can return 404 when protection uses rulesets.

Do not test force pushes or deletion against main. Test ordinary contributor
review/CI blocking with a harmless PR; an owner account's bypass is not a test of
an ordinary contributor's access.

## References

- [Rules REST API](https://docs.github.com/en/rest/repos/rules)
- [Actions permissions REST API](https://docs.github.com/en/rest/actions/permissions)
- [Ruleset bypass permissions](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository)
- [Code owners](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)
