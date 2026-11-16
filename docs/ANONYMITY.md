# Anonymous publication

Export a clean review artifact from the working repository:

```bash
python -m reproducibility export --output dist/s2s-anonymous
python -m reproducibility verify dist/s2s-anonymous --strict
```

The allowlisted export includes source/configuration files, tests, the small benchmark snapshot, and archived reference scores. It excludes `.git`, model weights, generated results, credentials, caches, local environments, and editor files. It retains required third-party attributions.

The current working repository may contain author-identifying historical commits. Do not publish that history as an anonymous artifact. Use a separate anonymous account or an appropriate review proxy, and upload the exported copy. The optional `tools/publish_anonymous.sh` initializes fresh anonymous commit metadata only in a verified export; it refuses a directory that already has Git history.

Content scanning is a check for common identity/credential patterns, not a guarantee against account metadata, platform administrators, or code recognition. Inspect the actual public repository and downloaded artifact before submitting the link.
