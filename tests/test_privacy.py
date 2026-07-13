from __future__ import annotations

import io
import json
from typing import Any

import pytest
from PIL import Image

from plva_proxy.privacy import (
    DEFAULT_SAFETY_LEVELS,
    PLACEHOLDER_DUPLICATE_WARNING,
    PLACEHOLDER_INSTRUCTIONS,
    PLACEHOLDER_MANIFEST_KEY,
    PLACEHOLDER_MANIFEST_PREFIX,
    PLACEHOLDER_SCHEME,
    PLACEHOLDER_SYSTEM_BEGIN,
    HistoryScrubber,
    PrivacyError,
    SafetyPolicy,
    SessionVault,
    StubRedactor,
    StubSpan,
    VaultRedactor,
    _coalesce_finding_values,
    privacy_request_hook,
    privacy_response_hook,
)


def png_fixture() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (240, 80), "white").save(output, format="PNG")
    return output.getvalue()


def test_vault_assigns_stable_nonce_namespaced_placeholders_and_disposes() -> None:
    vault = SessionVault(nonce="a3f9")

    first = vault.store("EMAIL", " Alice@Example.com ")
    same = vault.store("email", "Alice@Example.com")
    second = vault.store("EMAIL", "bob@example.com")

    assert first == same == "EMAIL_1_a3f9"
    assert vault.nonce == "a3f9"
    assert second == "EMAIL_2_a3f9"
    assert vault.resolve(first) == " Alice@Example.com "
    assert vault.resolve(second) == "bob@example.com"
    vault.dispose()
    with pytest.raises(PrivacyError, match="not issued"):
        vault.resolve(first)


def test_vault_rejects_invalid_nonce_empty_values_and_keeps_credentials_case_sensitive() -> None:
    with pytest.raises(ValueError, match="nonce"):
        SessionVault(nonce="invalid")
    vault = SessionVault(nonce="a3f9")
    with pytest.raises(PrivacyError, match="empty"):
        vault.store("EMAIL", "  ")
    assert vault.store("API_KEY", "Secret") != vault.store("API_KEY", "secret")


def test_plain_scrub_never_rewrites_inside_placeholder_tokens_or_uses_tiny_fragments() -> None:
    vault = SessionVault(nonce="a3f9", policy=SafetyPolicy({"GOV_ID": "hide_use"}))
    token = vault.store("EMAIL", "alice@example.com")
    vault.store("GOVERNMENT_ID", "a3f9")
    vault.store("NAME", "n")

    assert vault.scrub_plain(token) == token
    assert vault.scrub_plain("n") == "n"


def test_adjacent_same_class_ocr_spans_are_coalesced_before_vaulting() -> None:
    values = _coalesce_finding_values(
        {
            "text": "sean_hammond",
            "values": [
                {"label": "GIVEN_NAME", "value": "sean", "start": 0, "end": 4},
                {
                    "label": "GIVEN_NAME",
                    "value": "_hammond",
                    "start": 4,
                    "end": 12,
                },
            ],
        }
    )

    assert [(value["label"], value["value"]) for value in values] == [
        ("GIVEN_NAME", "sean_hammond")
    ]


def test_stub_ocr_finding_drives_both_vault_and_visible_placeholder_chip() -> None:
    source = png_fixture()
    stub = StubRedactor((StubSpan("EMAIL", "alice@example.com", (20, 20, 220, 55)),))
    vault = SessionVault(nonce="a3f9")
    redactor = VaultRedactor(stub, vault)

    painted = redactor(source)
    analysis = redactor.latest_analysis

    assert vault.resolve("EMAIL_1_a3f9") == "alice@example.com"
    assert analysis["findings"][0]["values"][0]["placeholder"] == "EMAIL_1_a3f9"
    assert analysis["findings"][0]["placeholders"] == ["EMAIL_1_a3f9"]
    assert analysis["vault_placeholders"] == 1
    with Image.open(io.BytesIO(source)) as before, Image.open(io.BytesIO(painted)) as after:
        assert after.getpixel((0, 0)) == before.getpixel((0, 0))
        assert after.getpixel((25, 25)) != before.getpixel((25, 25))


def test_stub_detect_nothing_preserves_all_pixels_and_creates_no_vault_entry() -> None:
    source = png_fixture()
    vault = SessionVault(nonce="a3f9")
    redactor = VaultRedactor(StubRedactor(), vault)

    output = redactor(source)

    with Image.open(io.BytesIO(source)) as before, Image.open(io.BytesIO(output)) as after:
        assert after.tobytes() == before.tobytes()
    assert vault.placeholders() == ()


def test_vault_redactor_caches_final_painted_frame_and_manifest() -> None:
    class CountingRedactor:
        def __init__(self) -> None:
            self.calls = 0
            self.inner = StubRedactor((StubSpan("EMAIL", "alice@example.com", (20, 20, 220, 55)),))

        @property
        def latest_analysis(self) -> dict[str, Any]:
            return self.inner.latest_analysis

        def start(self) -> None:
            self.inner.start()

        def close(self) -> None:
            self.inner.close()

        def __call__(self, png: bytes) -> bytes:
            self.calls += 1
            return self.inner(png)

    detector = CountingRedactor()
    redactor = VaultRedactor(detector, SessionVault(nonce="a3f9"))

    first = redactor.redact_with_manifest(png_fixture())
    second = redactor.redact_with_manifest(png_fixture())

    assert first == second
    assert detector.calls == 1


def test_vault_redactor_remasks_clipped_suffix_of_previously_vaulted_value() -> None:
    class ClippedFindingRedactor:
        def __init__(self) -> None:
            self._analysis: dict[str, Any] = {}

        @property
        def latest_analysis(self) -> dict[str, Any]:
            return self._analysis

        def start(self) -> None:
            return

        def close(self) -> None:
            self._analysis = {}

        def __call__(self, png: bytes) -> bytes:
            self._analysis = {
                "backend": "clipped-fixture",
                "findings": [
                    {
                        "x1": 20,
                        "y1": 20,
                        "x2": 220,
                        "y2": 55,
                        "text": "benchmark@example.com",
                        "labels": [],
                        "values": [],
                        "sensitive": False,
                    }
                ],
            }
            return png

    vault = SessionVault(nonce="a3f9")
    token = vault.store("EMAIL", "alice.benchmark@example.com")
    redactor = VaultRedactor(ClippedFindingRedactor(), vault)

    painted, manifest = redactor.redact_with_manifest(png_fixture())

    assert manifest == ({"token": token, "class": "EMAIL", "safety_level": "hide_use"},)
    assert redactor.latest_analysis["findings"][0]["vault_matches"] == [token]
    with Image.open(io.BytesIO(png_fixture())) as before, Image.open(io.BytesIO(painted)) as after:
        assert after.getpixel((25, 25)) != before.getpixel((25, 25))


def test_vault_observed_text_ignores_short_name_fragments_and_substrings() -> None:
    vault = SessionVault(nonce="a3f9")
    for value in ("e", "st", "int", "Alice Smith"):
        vault.store("NAME", value)

    matches = vault.match_observed_text("File Edit View History Profiles Alice Smith")

    assert [match["token"] for match in matches] == ["NAME_4_a3f9"]


def test_vault_name_scrub_never_corrupts_browseruse_protocol_words() -> None:
    vault = SessionVault(nonce="a3f9")
    for value in ("odel", "con", "key"):
        vault.store("NAME", value)

    text = "Model integer continue hotkey_desktop"

    assert vault.scrub_plain(text) == text


def test_vault_observed_text_keeps_structured_clipped_value_guard() -> None:
    vault = SessionVault(nonce="a3f9")
    token = vault.store("EMAIL", "alice.benchmark@example.com")

    assert vault.match_observed_text("benchmark@example.com") == (
        {"token": token, "class": "EMAIL", "safety_level": "hide_use"},
    )


def test_stub_and_vault_redactor_lifecycle_and_invalid_inputs() -> None:
    value = "a@example.com"
    stub = StubRedactor((StubSpan("EMAIL", value, (0, 0, 20, 20)),))
    vault = SessionVault(nonce="a3f9")
    redactor = VaultRedactor(stub, vault)
    redactor.start()
    redactor(png_fixture())
    assert vault.resolve("EMAIL_1_a3f9") == value
    redactor.close()
    assert redactor.latest_analysis == {}
    with pytest.raises(PrivacyError, match="invalid image"):
        StubRedactor()(b"not an image")
    with pytest.raises(PrivacyError, match="outside"):
        StubRedactor((StubSpan("EMAIL", value, (-1, 0, 20, 20)),))(png_fixture())
    with pytest.raises(ValueError, match="cache"):
        VaultRedactor(StubRedactor(), SessionVault(nonce="a3f9"), cache_entries=-1)


def test_history_scrub_plain_match_then_semantic_backstop() -> None:
    vault = SessionVault(nonce="a3f9", policy=SafetyPolicy({"SSN": "hide_use"}))
    vault.store("EMAIL", "alice@example.com")

    def classify(texts: tuple[str, ...]) -> list[dict[str, object]]:
        assert texts[0] == "Already typed " + " " * len("EMAIL_1_a3f9")
        unknown = "472-81-0094"
        start = texts[1].index(unknown)
        return [
            {"sensitive": False, "values": []},
            {
                "sensitive": True,
                "values": [
                    {
                        "label": "SSN",
                        "value": unknown,
                        "start": start,
                        "end": start + len(unknown),
                    }
                ],
            },
        ]

    scrubber = HistoryScrubber(vault, classify)
    scrubbed = scrubber.scrub(("Already typed alice@example.com", "New value 472-81-0094"))

    assert scrubbed == ("Already typed EMAIL_1_a3f9", "New value SSN_1_a3f9")
    assert vault.resolve("SSN_1_a3f9") == "472-81-0094"


def test_history_scrub_opaquely_removes_blocked_values_without_vaulting() -> None:
    secret = "synthetic-secret"

    def classify(texts: tuple[str, ...]) -> list[dict[str, object]]:
        start = texts[0].index(secret)
        return [
            {
                "sensitive": True,
                "values": [
                    {
                        "label": "PASSWORD",
                        "value": secret,
                        "start": start,
                        "end": start + len(secret),
                    }
                ],
            }
        ]

    vault = SessionVault(nonce="a3f9")
    scrubber = HistoryScrubber(vault, classify)

    scrubbed = scrubber.scrub((f"Never repeat {secret}",))

    assert scrubbed == ("Never repeat [PLVA_BLOCKED_PASSWORD]",)
    assert vault.entries() == ()
    assert scrubber.diagnostics()["blocked_hits"] == 1


def test_history_scrub_fails_closed_when_classifier_has_no_exact_value() -> None:
    scrubber = HistoryScrubber(
        SessionVault(nonce="a3f9"),
        lambda texts: [{"sensitive": True, "values": []} for _ in texts],
    )

    with pytest.raises(PrivacyError, match="without an exact"):
        scrubber.scrub(("password field",))


def test_history_scrub_masks_blocked_value_without_vaulting_it() -> None:
    value = "not-a-real-password"

    def classify(texts: tuple[str, ...]) -> list[dict[str, object]]:
        start = texts[0].index(value)
        return [
            {
                "sensitive": True,
                "values": [
                    {
                        "label": "PASSWORD",
                        "value": value,
                        "start": start,
                        "end": start + len(value),
                    }
                ],
            }
        ]

    vault = SessionVault(nonce="a3f9")
    scrubbed = HistoryScrubber(vault, classify).scrub((f"Credential: {value}",))

    assert scrubbed == ("Credential: [PLVA_BLOCKED_PASSWORD]",)
    assert vault.entries() == ()


def test_history_scrub_rejects_malformed_classifier_results() -> None:
    vault = SessionVault(nonce="a3f9")
    with pytest.raises(PrivacyError, match="wrong result count"):
        HistoryScrubber(vault, lambda texts: []).scrub(("text",))
    with pytest.raises(PrivacyError, match="invalid values"):
        HistoryScrubber(vault, lambda texts: [{"values": None}]).scrub(("text",))
    with pytest.raises(PrivacyError, match="did not project"):
        HistoryScrubber(
            vault,
            lambda texts: [
                {"values": [{"label": "EMAIL", "value": "wrong", "start": 0, "end": 5}]}
            ],
        ).scrub(("email",))


def test_request_hook_scrubs_history_and_injects_placeholder_instructions() -> None:
    vault = SessionVault(nonce="a3f9")
    vault.store("EMAIL", "alice@example.com")
    scrubber = HistoryScrubber(
        vault,
        lambda texts: [{"sensitive": False, "values": []} for _ in texts],
    )
    hook = privacy_request_hook(scrubber)

    document, headers = hook(
        {
            "messages": [
                {"role": "system", "content": "Runtime instructions"},
                {"role": "assistant", "content": "Typed alice@example.com"},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "continue"}],
                },
            ]
        },
        {"content-type": "application/json"},
    )

    assert document["messages"][0]["role"] == "system"
    assert document["messages"][0]["content"].startswith("Runtime instructions\n\n")
    assert PLACEHOLDER_INSTRUCTIONS in document["messages"][0]["content"]
    assert document["messages"][0]["content"].count(PLACEHOLDER_SYSTEM_BEGIN) == 1
    assert document["messages"][1]["content"] == "Typed EMAIL_1_a3f9"
    assert headers == {"content-type": "application/json"}

    with pytest.raises(PrivacyError, match="message history"):
        hook({}, {})


def test_request_hook_merges_all_system_messages_for_hcompany_compatibility() -> None:
    scrubber = HistoryScrubber(
        SessionVault(nonce="a3f9"),
        lambda texts: [{"sensitive": False, "values": []} for _ in texts],
    )
    document, _ = privacy_request_hook(scrubber)(
        {
            "messages": [
                {"role": "system", "content": "Holo base prompt"},
                {"role": "system", "content": "Additional policy"},
                {"role": "user", "content": "Observe"},
            ]
        },
        {},
    )

    system_messages = [
        message for message in document["messages"] if message.get("role") == "system"
    ]
    assert len(system_messages) == 1
    assert "Holo base prompt\n\nAdditional policy" in system_messages[0]["content"]
    assert system_messages[0]["content"].count("[PLVA_PLACEHOLDERS_BEGIN]") == 1
    assert PLACEHOLDER_INSTRUCTIONS in system_messages[0]["content"]
    assert document["messages"][0] is system_messages[0]

    repeated, _ = privacy_request_hook(scrubber)(document, {})
    serialized = json.dumps(repeated)
    assert serialized.count("[PLVA_PLACEHOLDERS_BEGIN]") == 1


def test_request_hook_never_semantically_classifies_browseruse_control_plane_text() -> None:
    calls: list[tuple[str, ...]] = []

    def classify(texts: tuple[str, ...]) -> list[dict[str, object]]:
        calls.append(texts)
        return [{"sensitive": False, "values": []} for _ in texts]

    document, _ = privacy_request_hook(HistoryScrubber(SessionVault(nonce="a3f9"), classify))(
        {
            "messages": [
                {"role": "system", "content": "Model integer hotkey_desktop schema"},
                {"role": "assistant", "content": "continue"},
                {"role": "user", "content": "Open the account page"},
                {"role": "tool", "content": "Local result"},
            ]
        },
        {},
    )

    assert calls == [("Open the account page", "Local result")]
    assert document["messages"][0]["content"].startswith("Model integer hotkey_desktop schema")
    assert document["messages"][1]["content"] == "continue"


def test_request_prompt_reports_approval_without_egressing_context_details() -> None:
    vault = SessionVault(nonce="a3f9", policy=SafetyPolicy({"API_KEY": "approval"}))
    token = vault.store("API_KEY", "synthetic-secret-key")
    vault.grant_approval(
        token,
        tool_name="write",
        argument_path="args.content",
        target="https://private-target.test/account/123",
    )
    scrubber = HistoryScrubber(
        vault,
        lambda texts: [{"sensitive": False, "values": []} for _ in texts],
    )

    document, _ = privacy_request_hook(scrubber)(
        {"messages": [{"role": "system", "content": "Runtime instructions"}]}, {}
    )
    serialized = json.dumps(document)

    assert "[PLVA_LOCAL_APPROVALS]" in serialized and token in serialized
    assert "synthetic-secret-key" not in serialized
    assert "private-target" not in serialized
    assert "args.content" not in serialized


def test_request_hook_injects_only_current_manifest_and_removes_stale_teaching() -> None:
    vault = SessionVault(nonce="a3f9")
    assert vault.store("EMAIL", "alice@example.com") == "EMAIL_1_a3f9"
    assert vault.store("PHONE", "+1 415 555 0100") == "PHONE_1_a3f9"
    scrubber = HistoryScrubber(
        vault,
        lambda texts: [{"sensitive": False, "values": []} for _ in texts],
    )
    old_manifest = (
        f"{PLACEHOLDER_MANIFEST_PREFIX} Placeholders visible in the current screenshot: "
        "«EMAIL_9_a3f9» (email)."
    )
    messages = [
        {"role": "system", "content": PLACEHOLDER_INSTRUCTIONS},
        {"role": "user", "content": [{"type": "text", "text": old_manifest}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Fill the current form"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            ],
        },
    ]
    document, _ = privacy_request_hook(scrubber)(
        {
            "messages": messages,
            PLACEHOLDER_MANIFEST_KEY: {
                "message_index": 2,
                "items": [
                    {"token": "EMAIL_1_a3f9", "class": "EMAIL"},
                    {"token": "PHONE_1_a3f9", "class": "PHONE"},
                ],
            },
        },
        {},
    )

    serialized = json.dumps(document)
    assert serialized.count("[PLVA_PLACEHOLDERS]") == 1
    assert "EMAIL_9_a3f9" not in serialized
    assert PLACEHOLDER_MANIFEST_KEY not in document
    current = document["messages"][-1]["content"]
    assert current[-2]["text"] == (
        f"{PLACEHOLDER_MANIFEST_PREFIX} Placeholders visible in the current screenshot: "
        "«EMAIL_1_a3f9» (email · hidden, use allowed), "
        "«PHONE_1_a3f9» (phone · hidden, use allowed). "
        "Use only exact tokens listed in this manifest; never invent one."
    )
    assert current[-1]["type"] == "image_url"


def test_request_hook_emits_explicit_empty_manifest_and_rejects_forgery() -> None:
    scrubber = HistoryScrubber(
        SessionVault(nonce="a3f9"),
        lambda texts: [{"sensitive": False, "values": []} for _ in texts],
    )
    hook = privacy_request_hook(scrubber)
    empty, _ = hook(
        {
            "messages": [{"role": "user", "content": "Observe"}],
            PLACEHOLDER_MANIFEST_KEY: {"message_index": 0, "items": []},
        },
        {},
    )
    assert "visible in the current screenshot: none" in empty["messages"][0]["content"]

    with pytest.raises(PrivacyError, match="token is invalid"):
        hook(
            {
                "messages": [{"role": "user", "content": "Observe"}],
                PLACEHOLDER_MANIFEST_KEY: {
                    "message_index": 0,
                    "items": [{"token": "EMAIL_1_dead", "class": "PHONE"}],
                },
            },
            {},
        )


def test_request_hook_lists_issued_session_token_after_it_leaves_current_frame() -> None:
    vault = SessionVault(nonce="a3f9")
    token = vault.store("EMAIL", "alice@example.com")
    scrubber = HistoryScrubber(
        vault,
        lambda texts: [{"sensitive": False, "values": []} for _ in texts],
    )

    document, _ = privacy_request_hook(scrubber)(
        {
            "messages": [{"role": "user", "content": "Continue to the next form step"}],
            PLACEHOLDER_MANIFEST_KEY: {"message_index": 0, "items": []},
        },
        {},
    )

    manifest = document["messages"][0]["content"]
    assert "visible in the current screenshot: none" in manifest
    assert f"Active private-session tokens from earlier observations: «{token}»" in manifest
    assert "alice@example.com" not in manifest


def test_request_hook_rejects_well_shaped_but_unissued_manifest_token() -> None:
    scrubber = HistoryScrubber(
        SessionVault(nonce="a3f9"),
        lambda texts: [{"sensitive": False, "values": []} for _ in texts],
    )

    with pytest.raises(PrivacyError, match="not issued"):
        privacy_request_hook(scrubber)(
            {
                "messages": [{"role": "user", "content": "Observe"}],
                PLACEHOLDER_MANIFEST_KEY: {
                    "message_index": 0,
                    "items": [{"token": "EMAIL_1_a3f9", "class": "EMAIL"}],
                },
            },
            {},
        )


def test_history_scrubber_classifies_only_new_or_changed_safe_texts() -> None:
    calls: list[tuple[str, ...]] = []

    def classify(texts: tuple[str, ...]) -> list[dict[str, object]]:
        calls.append(texts)
        return [{"sensitive": False, "values": []} for _ in texts]

    scrubber = HistoryScrubber(SessionVault(nonce="a3f9"), classify)
    assert scrubber.scrub(("base prompt", "first step")) == ("base prompt", "first step")
    assert scrubber.scrub(("base prompt", "first step", "second step")) == (
        "base prompt",
        "first step",
        "second step",
    )

    assert calls == [("base prompt", "first step"), ("second step",)]
    assert scrubber.diagnostics()["classification_cache_hits"] == 2


def test_request_features_can_be_disabled_independently_for_diagnostics() -> None:
    calls = 0

    def classify(_: tuple[str, ...]) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        return []

    scrubber = HistoryScrubber(SessionVault(nonce="a3f9"), classify)
    base = {
        "messages": [
            {"role": "system", "content": "Runtime instructions"},
            {"role": "user", "content": "synthetic@example.invalid"},
        ],
        PLACEHOLDER_MANIFEST_KEY: {"message_index": 1, "items": []},
    }
    disabled, _ = privacy_request_hook(
        scrubber,
        history_scrub=False,
        inject_scheme=False,
        inject_duplicate_warning=False,
        inject_manifest=False,
        inject_policy=False,
    )(base, {})

    assert disabled == {
        "messages": [
            {"role": "system", "content": "Runtime instructions"},
            {"role": "user", "content": "synthetic@example.invalid"},
        ]
    }
    assert calls == 0

    scheme, _ = privacy_request_hook(
        scrubber,
        history_scrub=False,
        inject_scheme=True,
        inject_duplicate_warning=False,
        inject_manifest=False,
        inject_policy=False,
    )(base, {})
    duplicate, _ = privacy_request_hook(
        scrubber,
        history_scrub=False,
        inject_scheme=False,
        inject_duplicate_warning=True,
        inject_manifest=False,
        inject_policy=False,
    )(base, {})

    assert PLACEHOLDER_SCHEME in scheme["messages"][0]["content"]
    assert PLACEHOLDER_DUPLICATE_WARNING in duplicate["messages"][0]["content"]
    assert len([m for m in scheme["messages"] if m["role"] == "system"]) == 1
    assert len([m for m in duplicate["messages"] if m["role"] == "system"]) == 1


def test_vault_redactor_manifest_contains_tokens_and_classes_never_values() -> None:
    value = "alice@example.com"
    redactor = VaultRedactor(
        StubRedactor((StubSpan("EMAIL", value, (20, 20, 220, 55)),)),
        SessionVault(nonce="a3f9"),
    )

    _, manifest = redactor.redact_with_manifest(png_fixture())

    assert manifest == (
        {
            "token": "EMAIL_1_a3f9",
            "class": "EMAIL",
            "safety_level": "hide_use",
        },
    )
    assert value not in json.dumps(manifest)


def test_safety_policy_defaults_and_resolution_gates() -> None:
    assert DEFAULT_SAFETY_LEVELS["EMAIL"] == "hide_use"
    assert DEFAULT_SAFETY_LEVELS["API_KEY"] == "approval"
    assert DEFAULT_SAFETY_LEVELS["PASSWORD"] == "blocked"
    policy = SafetyPolicy({"PHONE": "blocked", "SSN": "approval"})
    vault = SessionVault(nonce="a3f9", policy=policy)

    assert vault.store("EMAIL", "alice@example.com") == "EMAIL_1_a3f9"
    approval = vault.store("SSN", "472-81-0094")
    with pytest.raises(PrivacyError, match="approval"):
        vault.resolve(approval)
    with pytest.raises(PrivacyError, match="blocked PII"):
        vault.store("PHONE", "+1 415 555 0136")
    with pytest.raises(PrivacyError, match="blocked PII"):
        vault.store("UNRECOGNIZED_PII", "synthetic")


def test_approval_grant_is_exact_one_use_and_never_exposes_value() -> None:
    policy = SafetyPolicy({"API_KEY": "approval"})
    vault = SessionVault(nonce="a3f9", policy=policy)
    token = vault.store("API_KEY", "synthetic-secret-key")

    grant = vault.grant_approval(
        token,
        tool_name="write",
        argument_path="content",
        target="https://example.test",
        ttl_seconds=30,
    )

    assert "synthetic-secret-key" not in json.dumps(grant)
    resolved = vault.resolve_action(
        ((token, "content"),), tool_name="write", target="https://example.test"
    )
    assert resolved[(token, "content")] == "synthetic-secret-key"
    assert vault.approvals() == ()
    with pytest.raises(PrivacyError, match="matching local approval"):
        vault.resolve_action(
            ((token, "content"),), tool_name="write", target="https://example.test"
        )


def test_approval_rejects_wrong_tool_path_target_and_expiry() -> None:
    now = [10.0]
    vault = SessionVault(
        nonce="a3f9",
        policy=SafetyPolicy({"API_KEY": "approval"}),
        clock=lambda: now[0],
    )
    token = vault.store("API_KEY", "synthetic-secret-key")
    vault.grant_approval(
        token,
        tool_name="write",
        argument_path="content",
        target="https://example.test",
        ttl_seconds=1,
        use_count=2,
    )

    for tool, path, target in (
        ("click", "content", "https://example.test"),
        ("write", "args.content", "https://example.test"),
        ("write", "content", "https://attacker.test"),
    ):
        with pytest.raises(PrivacyError, match="matching local approval"):
            vault.resolve_action(((token, path),), tool_name=tool, target=target)
    assert vault.approvals()[0]["remaining_uses"] == 2
    now[0] = 11.1
    assert vault.approvals() == ()


def test_approval_resolution_is_atomic_and_revocable() -> None:
    vault = SessionVault(nonce="a3f9", policy=SafetyPolicy({"API_KEY": "approval"}))
    token = vault.store("API_KEY", "synthetic-secret-key")
    vault.grant_approval(token, tool_name="write", argument_path="content")

    with pytest.raises(PrivacyError, match="not issued"):
        vault.resolve_action(
            ((token, "content"), ("API_KEY_99_a3f9", "content")),
            tool_name="write",
            target=None,
        )
    assert vault.approvals()[0]["remaining_uses"] == 1
    assert vault.revoke_approval(token, tool_name="write", argument_path="content")
    assert not vault.revoke_approval(token, tool_name="write", argument_path="content")


def test_blocked_finding_is_masked_but_never_vaulted_or_manifested() -> None:
    vault = SessionVault(nonce="a3f9")
    redactor = VaultRedactor(
        StubRedactor((StubSpan("PASSWORD", "not-a-real-secret", (20, 20, 220, 55)),)),
        vault,
    )

    _, manifest = redactor.redact_with_manifest(png_fixture())

    assert manifest == ()
    assert vault.entries() == ()
    assert redactor.latest_analysis["findings"][0]["values"][0]["safety_level"] == "blocked"


def test_policy_prompt_teaches_each_security_level_without_values() -> None:
    policy = SafetyPolicy({"EMAIL": "approval", "API_KEY": "blocked"})
    prompt = policy.prompt()

    assert "hide_use=" in prompt
    assert "approval=" in prompt and "EMAIL" in prompt
    assert "blocked=" in prompt and "API_KEY" in prompt
    assert "alice@example.com" not in prompt


@pytest.mark.parametrize("shape", ["singular", "plural"])
def test_response_resolves_only_executed_fields_not_reasoning(shape: str) -> None:
    vault = SessionVault(nonce="a3f9")
    token = vault.store("EMAIL", "alice@example.com")
    call = {"tool_name": "write", "content": token}
    action: dict[str, Any] = {"note": token, "thought": f"Use {token}"}
    if shape == "singular":
        action["tool_call"] = call
    else:
        action["tool_calls"] = [call]
    document = {"choices": [{"message": {"role": "assistant", "content": json.dumps(action)}}]}

    rewritten = privacy_response_hook(vault)(document)
    result = json.loads(rewritten["choices"][0]["message"]["content"])
    executed = result["tool_call"] if shape == "singular" else result["tool_calls"][0]

    assert executed["content"] == "alice@example.com"
    assert result["note"] == token
    assert result["thought"] == f"Use {token}"


def test_response_fails_closed_on_forged_placeholder_in_action() -> None:
    vault = SessionVault(nonce="a3f9")
    document = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {"tool_call": {"tool_name": "write", "content": "EMAIL_99_a3f9"}}
                    )
                }
            }
        ]
    }

    with pytest.raises(PrivacyError, match="not issued"):
        privacy_response_hook(vault)(document)


def test_response_uses_approval_only_for_the_granted_action_context() -> None:
    vault = SessionVault(nonce="a3f9", policy=SafetyPolicy({"API_KEY": "approval"}))
    token = vault.store("API_KEY", "synthetic-secret-key")
    vault.grant_approval(token, tool_name="write", argument_path="content")
    document = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"tool_call": {"tool_name": "write", "content": token}})
                }
            }
        ]
    }

    result = privacy_response_hook(vault)(document)
    action = json.loads(result["choices"][0]["message"]["content"])
    assert action["tool_call"]["content"] == "synthetic-secret-key"
    assert "synthetic-secret-key" not in json.dumps(document)


def test_response_resolves_decorative_wrapper_without_typing_guillemets() -> None:
    vault = SessionVault(nonce="a3f9")
    token = vault.store("EMAIL", "alice@example.com")
    document = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {"tool_call": {"tool_name": "write", "content": f"« {token} »"}}
                    )
                }
            }
        ]
    }

    result = privacy_response_hook(vault)(document)
    action = json.loads(result["choices"][0]["message"]["content"])
    assert action["tool_call"]["content"] == "alice@example.com"


@pytest.mark.parametrize(
    "action,extract",
    [
        (
            {"action": "type", "text": "EMAIL_1_a3f9", "thought": "EMAIL_1_a3f9"},
            lambda value: value["text"],
        ),
        (
            {"action": {"type": "type", "text": "EMAIL_1_a3f9"}},
            lambda value: value["action"]["text"],
        ),
        (
            {
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "write",
                            "arguments": '{"content":"EMAIL_1_a3f9"}',
                        },
                    }
                ]
            },
            lambda value: value["tool_calls"][0]["function"]["arguments"],
        ),
    ],
)
def test_response_resolves_supported_action_grammars(action: dict[str, Any], extract: Any) -> None:
    vault = SessionVault(nonce="a3f9")
    vault.store("EMAIL", "alice@example.com")
    document = {"choices": [{"message": {"content": json.dumps(action)}}]}

    result = privacy_response_hook(vault)(document)
    rewritten = json.loads(result["choices"][0]["message"]["content"])

    assert "alice@example.com" in extract(rewritten)
    if isinstance(rewritten.get("thought"), str):
        assert rewritten["thought"] == "EMAIL_1_a3f9"


def test_response_rejects_malformed_tool_call_envelopes() -> None:
    vault = SessionVault(nonce="a3f9")
    document = {"choices": [{"message": {"content": json.dumps({"tool_calls": ["not-a-call"]})}}]}

    with pytest.raises(PrivacyError, match="invalid shape"):
        privacy_response_hook(vault)(document)


def test_response_resolves_nested_direct_tool_fields_but_not_answers() -> None:
    vault = SessionVault(nonce="a3f9")
    token = vault.store("EMAIL", "alice@example.com")
    direct = {
        "tool_name": "write",
        "args": {"values": [token], "metadata": {"token": token}},
    }
    answer = {"tool_name": "answer", "content": token}
    document = {
        "choices": [
            {"message": {"content": json.dumps(direct)}},
            {"message": {"content": json.dumps(answer)}},
        ]
    }

    result = privacy_response_hook(vault)(document)
    resolved = json.loads(result["choices"][0]["message"]["content"])
    untouched = json.loads(result["choices"][1]["message"]["content"])

    assert resolved["args"]["values"] == ["alice@example.com"]
    assert resolved["args"]["metadata"]["token"] == "alice@example.com"
    assert untouched["content"] == token
