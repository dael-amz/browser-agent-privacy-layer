# Packet-filter bootstrap handoff

The live launcher now enforces loopback-only TCP for the complete CUA process
group without elevation. This permits Holo's local CLI↔Agent-API channel and
the PLVA proxy while rejecting external destinations. The macOS packet-filter anchor is
an additional host-level boundary. It cannot be inspected or installed by an
unprivileged process, and this project never invokes `sudo` automatically.

First get a machine-readable, non-mutating report:

```bash
.venv/bin/python -m plva_proxy.egress_preflight
```

On a normal unbootstrapped Mac this reports `ready:false`, a missing
`_plvaproxy` role user, and/or `inspectable_without_elevation:false`. An
administrator must then review and perform the following manually:

```bash
sudo sysadminctl -addUser _plvaproxy -shell /usr/bin/false
sudo pfctl -nf docs/egress/pf-plva.anchor
sudo pfctl -a plva -f docs/egress/pf-plva.anchor
sudo pfctl -e
sudo pfctl -a plva -sr
```

Do not put a provider key directly in the command line or shell history. Give
the role user read/execute access only to the proxy runtime and a protected
credential file, then start the proxy under that role through an
administrator-reviewed service definition. Reload the anchor before each run
because provider hostnames are resolved when the rules are loaded.

The bootstrap is complete only when the administrator-side `pfctl` inspection
shows the expected `plva` rules. A nonprivileged status process may continue to
report inspection as unavailable even when PF is active; it must not claim PF
is enabled without evidence.
