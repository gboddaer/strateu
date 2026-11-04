# Hosted Kolibri code-quality check: Tesseracted

## Request and budget

User authorized the same code-generation tests at
https://tesseracted.com/kolibri-1-chat/, with a strict **maximum of 10 shots**.
No account creation, quota bypass, proxy rotation, CAPTCHA bypass or automatic
message retry is permitted or attempted. Initial page inspection showed 10 guest
messages left. Ordinary page/asset loads and analytics are not model submissions.

Plan: the five existing cases once each, fresh conversations, web search off,
extended thinking off; retain up to five additional shots for confirmation.
Every attempted submission and actual chat POST is recorded before it is sent.
The browser intercept blocks duplicate chat POSTs for a case and any request
beyond the ten-shot cap. A failed submission is not automatically retried.

## Method and comparability

Use the normal browser UI and its own public chat flow. Prompts contain only the
public toy coding instructions from `tools/kolibri/code_quality_cases.json`, not
reference answers or oracle source. The original system instruction is prepended
to the user text, because this UI does not expose a custom system-role control.
Thus the **task is the same**, but the provider's hidden system prompt, template,
sampling, output budget and backend are not controlled. Do not call this an
identical-token or engine-fidelity comparison.

The downloaded client labels the default model variant `official` and identifies
responses as `Kolibri · Official FP8`. That is the site's description, not an
independent verification of its weight revision, precision or serving engine.
No alternative quantized variant is selected.

Use the corrected local C++ compile/behavior checks from the earlier
[quality investigation](generated-code-quality.md). The raw-mode check tests
actual Ctrl+S/Q delivery, not just an IXON mask. An uncompiled full notepad is
not assigned a runtime pass rate.

## Artifacts and current state

Artifacts are under `/data4/kolibri-sram/out/tesseracted/`:

- `ledger.json`: submission attempts and actual chat POST records, hard cap 10.
- First-case exact prompt, initial inspection snapshot/screenshot and failure
  accessibility snapshot. No model response or generated code was received.
- Browser automation source and inspection logs under `browser/` and the root.

The recorder redacts Turnstile tokens if a chat request is observed. None was
sent during this attempt. Browser profile data remains local and is not part
of a proposed source commit. No user login is used.

Browser setup initially had local inspection/temporary-directory failures before
any model submission. Those are not counted as generated answers or evidence
about Kolibri.

## Outcome: blocked before inference (4 November 2026)

Attempt 1 submitted the string-insertion prompt through the normal UI with
web search off and extended thinking unchecked. The site's security check did
not release the request. The final page displayed:

> Security check is taking longer than expected. Please try again.

The local observer subsequently timed out waiting for a chat response. The
failure snapshot (`insert-char-failure-snapshot.yml`) still shows **10 free
messages left**. The network ledger contains **zero chat POSTs**, zero blocked
retries and one failed UI-submission attempt. There is no inference result to
compile, score or attribute to the model.

On the user's `resume`, inspected and waited for the existing attempt; did not
resubmit or start another model request. Stopped rather than bypassing the
security check or changing identity/network to evade it. The browser closed.
Human-verified access or the provider's approved API access is needed to proceed
from this environment. Alternatively, the same five prompts can be run manually
and the returned code evaluated with the existing local oracles.

Exact copy/paste prompts and UI settings are prepared in
`/data4/kolibri-sram/out/tesseracted/manual-prompts.md`. No automatic test success/failure rate is reported for the hosted model.

## Manual responses supplied by the user

The user is running the prompts in their browser and pasting responses here.
These are user-supplied hosted responses, not independently captured API output.
They are separate from the failed automated attempt above.

1. **String insertion:** unedited pasted function compiles and passes the
   existing insertion oracle (`manual/insert-char-result.json`).
2. **Line editing:** pasted signatures contain `std::vectorstd::string`, so the
   original angle-bracket formatting is unconfirmed. Preserve that paste as-is;
   do not attribute its compiler diagnostics to the model. A separately labelled
   diagnostic restores only the two type spellings to `std::vector<std::string>`.
   It compiles but fails the join-position assertion (`r==0 && c==2`): the code
   puts the cursor at the combined line's end. Static inspection also finds that
   backspace erases the last character, not the character before the cursor.
   No logic was repaired. Artifacts: `manual/line-edit-response.txt`,
   `manual/line-edit-type-restored.txt`, and `manual/line-edit-result.json`.

3. **Raw mode:** unedited pasted function compiles but fails actual Ctrl+S/Q
   delivery through the PTY. It leaves IXON enabled and assigns VSTART=19
   (Ctrl+S), so Ctrl+S is still consumed as flow control instead of delivered
   to the editor. Artifacts: `manual/raw-mode-response.txt`,
   `manual/raw-mode-result.json`, and `manual/raw-mode-run.log`.

4. **Nonblocking input:** unedited function compiles and passes the existing
   oracle for EAGAIN/would-block, unsigned byte 255 and EOF. EINTR retry is
   present in the source but was not dynamically exercised by this oracle.
   Artifacts: `manual/nonblocking-read-response.txt` and
   `manual/nonblocking-read-result.json`.

5. **Complete notepad:** complete pasted 176-line source fails compilation
   unchanged, with two compiler diagnostic lines:
   - No C++ overload for `std::getline(FILE*, std::string&)`.
   - The noncapturing `atexit` lambda references the automatic local `old_t`.
   Artifacts: `manual/notepad-response.txt`, `manual/notepad.cpp`,
   `manual/notepad-result.json`, and `manual/notepad-compile.log`. No executable
   was produced and no full-editor runtime score was assigned. Static inspection
   also finds incorrect winsize handling, saving without a newline per buffer
   line, and unreachable escape-parser states; those were not runtime-tested.

Five manual responses have now been received: two pass their existing behavior
oracles, two have demonstrated semantic failures (line-edit after the separately
labelled type restoration), and the full editor fails compilation unchanged.
The exact original formatting of Test 2 remains unconfirmed. No pasted logic
was repaired to obtain a pass. No additional automatic model request was sent;
the automated ledger still contains one failed UI attempt and zero chat POSTs.
Those were the first five user-reported manual generations, not an independently
observed count of the site's remaining allowance.

## Sixth manual generation: complete notepad with thinking enabled

At the user's request, repeat Test 5 unchanged in a fresh conversation, with
Web off and Extended thinking checked. The user pasted a long reasoning trace
that stopped mid-plan at `tcsetattr(`, no final code, and the site's notice:

> Output limit reached. No answer was produced within the limit. Try a shorter
> request or turn off extended thinking.

Record this as **no answer within the hosted output budget**, not as a C++
compilation failure. No final program was supplied, so no compiler or PTY test
was run. The exact cap, generated token count and backend settings are unknown.
A result summary and the exact notice are in
`manual/notepad-thinking-result.json`; the complete reasoning was user-pasted in
the conversation, not independently captured by the browser recorder.

Six manual generations have now been reported. No automatic chat POST, retry,
continuation or repair request was sent. This final control shows that simply
enabling thinking did not deliver a program within this demo's budget. It does
**not** establish whether a larger reasoning/output budget would produce correct
code, or whether thinking generally improves or harms the model's coding.

## Finding and limits

The hosted response reproduces the same invalid `std::getline(FILE*, string&)`
choice seen in Strata Q4, llama.cpp Q4 and llama.cpp Q8. The line-join cursor
error also recurs. Together with the local controls, this is further evidence
of model-side C++ reliability limitations, rather than a defect unique to the
Strata engine or solely to the local Q4 quantization.

Do not overstate that result: `Official FP8` is the site's label, the generation
was manually relayed, and its model revision, hidden prompt, sampling and backend
were not independently verified. This does not prove every remaining Strata
numerical path is correct, nor measure general model coding ability. Five
distinct targeted prompts and one thinking-enabled repeat were attempted. The
repeat is inconclusive for final-code correctness because its answer was capped.
No further hosted shots are needed to establish the shared failures above.
