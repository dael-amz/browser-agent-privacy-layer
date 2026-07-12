# Optional voiceover script (~120s, conversational pace)

The video is designed to work silently: every beat carries its own on-screen
copy. If you record VO (or use Gradium TTS for bonus sponsor points), duck the
music −6 dB under it. Times match the timeline in README.md.

**0:00–0:05: Cold open**
> Right now, an AI agent somewhere is screenshotting a screen exactly like yours.

**0:05–0:14: The problem**
> Computer-use agents work by screenshotting your entire screen: every single
> step: and shipping the raw pixels to a cloud model. Everything you can see,
> the provider sees.

**0:14–0:24: The constraint → the hack**
> Here's the catch: the agent runtime is a closed binary. You can't patch it.
> The only thing it lets you configure is the base URL of the model API.
> So… we became the base URL.

**0:24–0:33: Title**
> This is Holo: private computer use. PLVA is a local, fail-closed proxy
> between the closed runtime and H Company's Holo3: and it intercepts both
> directions of traffic.

**0:33–0:47: Request leg**
> Before a single byte leaves the machine: on-device Apple Vision and Core ML
> find the PII. The proxy paints placeholder chips over the real pixels. Real
> values go into a memory-only vault. History is scrubbed: exact match plus a
> semantic backstop: and the model gets a manifest teaching it the tokens.

**0:47–1:00: Response leg**
> Now the twist. Holo3 answers: type EMAIL-one. On the way back, the proxy
> resolves the token against the vault: locally, in transit: and the REAL
> email gets typed into the field. The task works. The model never saw the
> value. The model works with what it can't see.

**1:00–1:14: With vs Without**
> Side by side. Without PLVA, the provider receives your raw screen. With
> PLVA, it receives painted chips. Both tasks complete. Same task. Same
> result. Only one of them leaked.

**1:14–1:28: Live proof**
> Don't trust us: audit it. One environment variable turns it on. The live
> audit viewer shows every redacted frame Holo3 actually received: chips,
> never values. End to end, on H Company's Holo3-35B.

**1:28–1:41: Fail-closed**
> And for the security engineers: fail-closed everywhere: if any stage fails,
> nothing is forwarded. Tokens carry a per-session nonce, so on-screen text
> can't forge one. Streaming-safe. Memory-only. Passwords? Never resolvable.
> Period.

**1:41–1:52: Adoption**
> Adoption cost: one config line. No SDK, no patching the agent, no provider
> cooperation. A local control panel sets policy per class: live.

**1:52–2:00: Endcard**
> Holo. Private computer use. Redact for the model: not for the user.
