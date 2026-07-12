import assert from "node:assert/strict";
import test from "node:test";
import { createGuard } from "@nationaldesignstudio/rampart";

test("redacts structured PII without loading the model", async () => {
  const guard = await createGuard({ heuristicsOnly: true });
  const result = await guard.protect(
    "Email test.person@example.com; SSN 472-81-0094; card 4111 1111 1111 1111.",
  );

  assert.equal(result.text.includes("test.person@example.com"), false);
  assert.equal(result.text.includes("472-81-0094"), false);
  assert.equal(result.text.includes("4111 1111 1111 1111"), false);
  assert.match(result.text, /\[EMAIL_1\]/);
  assert.match(result.text, /\[SSN_1\]/);
  assert.match(result.text, /\[CREDIT_CARD_1\]/);
});

test("restores placeholders within the same local session", async () => {
  const guard = await createGuard({ heuristicsOnly: true });
  const original = "Write to test.person@example.com.";
  const safe = await guard.protect(original);

  assert.equal(guard.reveal(safe.text), original);
});
