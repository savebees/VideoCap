# TODO

The following components are intentionally outside the current VideoCap release:

- **VideoEval:** evaluate global captions, event boundaries, event captions, temporal coverage, consistency, and hallucination with explicit reference-based and human-review protocols.
- **VideoQA:** derive grounded question-answer examples and task labels from accepted VideoCap annotations, with provenance back to events and evidence frames.

Before either component is added, define its public schema, acceptance criteria, benchmark splits, and reproducible evaluation protocol. Keep both modules independent from the VideoCap production path so annotation generation remains small and auditable.
