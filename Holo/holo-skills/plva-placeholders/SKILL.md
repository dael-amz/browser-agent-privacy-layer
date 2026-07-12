---
name: PLVA Placeholders
description: Using privacy-preserving placeholder chips in computer-use actions without guessing hidden values.
publisher: PLVA
version: "1.0.0"
license: MIT
---

Some sensitive values on screen may be covered by chips written like
`«EMAIL_1_ab12»` or `«PHONE_2_ab12»`. Each chip represents a real value that
you cannot see. Its class label identifies the kind of value, and its suffix
belongs to the current private session.

Treat a visible placeholder as the real value of that class when planning an
action. To use it, copy only the inner token, such as `EMAIL_1_ab12`, exactly
into the executed action field. Preserve its spelling, capitalization, number,
and session suffix. Do not include the decorative `« »` marks.

Each request may contain a `[PLVA_SECURITY_POLICY]` instruction describing the
active level for every class. Obey it exactly: `hide_use` tokens may be used in
executed actions, `approval` tokens must not be used until the local privacy
layer grants approval, and `blocked` classes have no usable token. The live
policy instruction is authoritative because users can change it between runs.

Never invent a placeholder, guess or reconstruct its hidden value, or alter a
token. A token mentioned only in notes or reasoning is not executed. The live
manifest distinguishes tokens visible in the current screenshot from exact
tokens still active for this private session. You may reuse an active token in
a later step only when the task clearly refers to the same value you observed
earlier. Never reuse a token absent from the active-session list.
