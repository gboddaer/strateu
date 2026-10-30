# Protect main after publication

Prepared for `gboddaer/strateu`. These files do not change GitHub settings by themselves.
They are ready-to-submit REST API payloads, not an assertion that protection is active.

## Policy and limits

- `main-safety`: every change needs a PR; block force pushes and branch deletion;
  resolve review conversations. No actor can bypass this ruleset.
- `owner-approval`: one approval, including a code-owner approval; invalidate stale
  approvals and require approval from someone other than the last pusher.
- `.github/CODEOWNERS`: `@gboddaer` owns every path, including CODEOWNERS itself.
  GitHub uses the file on the PR's base branch, so land it on main before activation.
- The approval ruleset permits repository admins (GitHub's built-in role ID 5) to
  bypass **through a PR only**. This lets the sole owner merge their own PRs, since
  authors cannot approve themselves. The separate safety rules still apply.
- This admin exception can bypass approval on ANY PR, not only your own. Use it
  deliberately, normally only for your own PRs. Adding another admin gives them
  this exception too. GitHub cannot make repository rules immutable to an admin
  who can edit or delete them.
- These rules require your approval for ordinary contributors; they do not stop
  other people from leaving non-binding reviews. Keep yourself the only admin
  and do not grant unnecessary write access.
- CI checks are intentionally NOT required yet. After verifying which checks run
  reliably on every PR, add `required_status_checks` to `main-safety`, with their
  exact names and preferably their GitHub App integration IDs. Do not put checks
  in the bypassable approval ruleset. Until then, inspect CI manually before merging.
- Only `main` is protected. Tags, releases, other branches, Actions, secrets and
  publication/history audits are outside this configuration's scope.

## Before publication

1. Review and commit these files, then push them to main while it is still private.
   Do not put credentials in the payloads or this document.
2. Audit repository history and Actions before making anything public. These rules
   do not make untrusted PR workflows safe or remove leaked secrets.
3. Confirm you are still the only admin. If you add another admin, review this policy.
4. Activate immediately after making the repo public, or first upgrade to a plan
   that supports rulesets on this private repo. Publication is a separate manual
   action: none of the commands below changes visibility.

## Apply once, from the repository root

Requires authenticated `gh` with repository administration permission. The current
private-repo plan returned HTTP 403 for rulesets during preparation. There is no
server-side dry-run here; GitHub acceptance is only known when submitted.

First confirm the account, target, visibility, and existing rulesets:

```sh
gh auth status
gh api repos/gboddaer/strateu --jq '{full_name, visibility, default_branch, permissions}'
gh api repos/gboddaer/strateu/rulesets --jq '.[] | {id, name, enforcement}'
```

If either name already exists, use the update commands below instead of creating
another copy. Confirm CODEOWNERS has landed:

```sh
gh api 'repos/gboddaer/strateu/contents/.github/CODEOWNERS?ref=main' --jq .sha
```

Create the approval gate first, then the non-bypassable safety gate. Stop if a
command fails. If only one succeeds, protection is partial: finish the other
before treating the branch as protected.

```sh
gh api --method POST repos/gboddaer/strateu/rulesets \
  --input .github/rulesets/owner-approval.json --jq '{id, name, enforcement}'
gh api --method POST repos/gboddaer/strateu/rulesets \
  --input .github/rulesets/main-safety.json --jq '{id, name, enforcement}'
```

Record both returned IDs. POST creates a new ruleset on every invocation; it is
not an update or an idempotent command.

To update an existing rule, substitute its numeric ID:

```sh
gh api --method PUT repos/gboddaer/strateu/rulesets/APPROVAL_ID \
  --input .github/rulesets/owner-approval.json
gh api --method PUT repos/gboddaer/strateu/rulesets/SAFETY_ID \
  --input .github/rulesets/main-safety.json
```

## Verify after application

Read the stored configuration and the rules effective on main:

```sh
gh api repos/gboddaer/strateu/rulesets/APPROVAL_ID
gh api repos/gboddaer/strateu/rulesets/SAFETY_ID
gh api repos/gboddaer/strateu/rules/branches/main
gh api 'repos/gboddaer/strateu/codeowners/errors?ref=main'
gh api repos/gboddaer/strateu/collaborators --jq '.[] | {login, role_name}'
```

Check that both enforcement values are `active`, both target `refs/heads/main`,
main-safety has an empty bypass list, and owner-approval has exactly the admin
role with `pull_request` bypass. Confirm no CODEOWNERS errors. Unexpected inherited
or additional rules must be reviewed, not blindly removed.

Finally, test a harmless external-contributor PR:

- Changes request `@gboddaer` as code owner; another person's approval cannot
  satisfy your code-owner requirement.
- Without your approval, GitHub shows review requirements as unmet. Your admin
  account may display a bypass option; that is not an ordinary contributor's access.
- Your approval permits merging after conversations are resolved; pushing another
  change invalidates approval.
- An owner-authored PR can be merged using the deliberate approval bypass, but
  still needs resolved conversations and must use a PR.

Do not test force pushes or deletion against main. The API readback verifies the
configured prohibition; behavioral testing belongs on a disposable test repository.
These checks are not claimed as executed during preparation.

## References

- [GitHub rules REST API](https://docs.github.com/en/rest/repos/rules)
- [Creating rulesets and PR-only bypass](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository)
- [Code owners](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)
