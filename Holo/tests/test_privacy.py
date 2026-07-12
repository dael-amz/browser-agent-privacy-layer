from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from plva_proxy.privacy import (
    PLACEHOLDER_INSTRUCTIONS,
    PLACEHOLDER_MANIFEST_KEY,
    PLACEHOLDER_MANIFEST_PREFIX,
    HistoryScrubber,
    PrivacyError,
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
    vault = SessionVault(nonce="a3f9")
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


def test_history_scrub_plain_match_then_semantic_backstop() -> None:
    vault = SessionVault(nonce="a3f9")
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


def test_history_scrub_fails_closed_when_classifier_has_no_exact_value() -> None:
    scrubber = HistoryScrubber(
        SessionVault(nonce="a3f9"),
        lambda texts: [{"sensitive": True, "values": []} for _ in texts],
    )

    with pytest.raises(PrivacyError, match="without an exact"):
        scrubber.scrub(("password field",))


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
                {"role": "assistant", "content": "Typed alice@example.com"},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "continue"}],
                },
            ]
        },
        {"content-type": "application/json"},
    )

    assert document["messages"][0] == {"role": "system", "content": PLACEHOLDER_INSTRUCTIONS}
    assert document["messages"][1]["content"] == "Typed EMAIL_1_a3f9"
    assert headers == {"content-type": "application/json"}

    with pytest.raises(PrivacyError, match="message history"):
        hook({}, {})


def test_request_hook_injects_only_current_manifest_and_removes_stale_teaching() -> None:
    scrubber = HistoryScrubber(
        SessionVault(nonce="a3f9"),
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
                    {"token": "PHONE_2_a3f9", "class": "PHONE"},
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
        "«EMAIL_1_a3f9» (email), «PHONE_2_a3f9» (phone). "
        "Use only the exact inner tokens shown here for this step."
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
    assert "visible in the current screenshot: none" in empty["messages"][1]["content"]

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


def test_vault_redactor_manifest_contains_tokens_and_classes_never_values() -> None:
    value = "alice@example.com"
    redactor = VaultRedactor(
        StubRedactor((StubSpan("EMAIL", value, (20, 20, 220, 55)),)),
        SessionVault(nonce="a3f9"),
    )

    _, manifest = redactor.redact_with_manifest(png_fixture())

    assert manifest == ({"token": "EMAIL_1_a3f9", "class": "EMAIL"},)
    assert value not in json.dumps(manifest)


@pytest.mark.parametrize("shape", ["singular", "plural"])
def test_response_resolves_only_executed_fields_not_reasoning(shape: str) -> None:
    vault = SessionVault(nonce="a3f9")
    token = vault.store("EMAIL", "alice@example.com")
    call = {"tool_name": "write", "content": token}
    action = {"note": token, "thought": f"Use {token}"}
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
