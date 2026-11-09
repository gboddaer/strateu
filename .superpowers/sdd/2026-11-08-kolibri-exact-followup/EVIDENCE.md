# Evidence storage

The full-logit `.bin` dumps are **not stored in Git**. They are large (about 6.9 GiB total; one dump is 62.5 MB) and
GitHub rejects or warns above 50 MB. They were removed from this branch's history on 2026-11-08; the tracked JSON,
logs and `compare.txt` files keep each run's complete SHA-256 and size, so exactness remains auditable.

Raw dumps (and the rest of the run tree) are preserved locally at:

    /home/gbo/kolibri-evidence/2026-11-08-followup-bins

The pre-rewrite branch tip is kept as local backup ref `refs/backup/kolibri-with-bins` (`441e3c75`). Raw dumps are
ignored by Git; do not add them with `-f`. Re-create a dump by rerunning the recorded benchmark command in the
matching `.json`/`.log`; compare its SHA-256 with the stored value.
