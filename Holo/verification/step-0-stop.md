# Step 0 verification: stopped

Date: 2026-07-11

Status: **STOPPED — external contracts do not satisfy the blueprint.**

> Superseded on 2026-07-11: the user explicitly authorized Overshoot's smaller
> 35B model. The selected exact model id is `Hcompany/Holo3-35B-A3B`. The
> historical findings below explain why the original 122B contract was stopped;
> current Step 0 evidence is recorded separately when its gates pass.

The blueprint requires every Step 0 `VERIFY` assumption to pass before later
steps are built. The checks below found blocking mismatches, so the project must
not silently substitute a model or claim end-to-end verification.

## Confirmed

- The current Overshoot base URL is `https://api.overshoot.ai/v1beta`.
- Overshoot documents `GET /models` as the source of truth and does not require
  authentication for that endpoint.
- A live `GET /models` probe returned these H models as ready:
  - `Hcompany/Holo-3.1-35B-A3B-FP8`
  - `Hcompany/Holo3-35B-A3B`
- The local `.env` contains an `API_KEY` value. Its value was never printed,
  copied, or committed.
- HoloDesktop documents `--base-url` and `--model` configuration for an
  OpenAI-compatible endpoint.
- OpenShell supports standalone sandboxes and declarative network policies.

## Blocking mismatches

1. The required `Holo3-122B-A10B` model is absent from Overshoot's live model
   discovery. Guessing an unadvertised model ID would violate hard constraint 8.
2. The installed local `holo` launcher is broken with
   `ModuleNotFoundError: No module named 'holo_desktop'`, so the required
   logging-stub capture has not been run.
3. `openshell` is not installed locally. More importantly, OpenShell policy
   governs sandboxed processes; it does not constrain a host-resident Holo
   runtime that needs macOS Accessibility and Screen Recording permissions.
   The blueprint's egress-isolation topology therefore needs an additional host
   network control or a proven contained-desktop design.
4. HoloDesktop's documented third-party custom-base-URL path does not expose a
   documented third-party API-key flag. The local PLVA proxy can inject the
   Overshoot credential, but that flow still needs the closed-runtime traffic
   capture before it is treated as verified.

## Evidence

- [Overshoot models](https://docs.overshoot.ai/models)
- [Overshoot API introduction](https://docs.overshoot.ai/api-reference/introduction)
- [HoloDesktop CLI reference](https://hub.hcompany.ai/holo-desktop-cli/reference/cli)
- [HoloDesktop architecture](https://hub.hcompany.ai/holo-desktop-cli/architecture)
- [Holo agent-loop request schema](https://hub.hcompany.ai/agent-loop)
- [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell)

## Required decision

Continue only after one of these is supplied and explicitly selected:

- an Overshoot private endpoint/model ID that serves `Holo3-122B-A10B`; or
- authorization to change the fixed model/provider constraint (for example,
  use Overshoot's available 35B Holo model or H Company's hosted 122B model).

After that decision, repair Holo in an isolated project environment, capture
its exact request/response contract with a logging stub using only synthetic
data, and prove the selected OpenShell/host-network topology before allowing the
agent to control the desktop.
