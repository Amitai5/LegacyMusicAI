# Vendor integrations

Upstream model repositories are installed here in isolated environments and are ignored by Git. Installers must pin a reviewed revision, record the upstream license, verify checksums where available, and never update a checkout implicitly during a generation run.

Installed by `scripts/install-models.ps1` at the commits in `config/upstreams.yaml`:

- `ACE-Step-1.5/`
- `SoulX-Singer/`
- `seed-vc/` (optional; install with `-IncludeSeedVC`, then select it explicitly at runtime)

The installer applies only the registered ACE-Step Windows training patch. First-party
SoulX and Seed-VC wrappers live in `scripts/`; vocal-pipeline changes do not require editing
those upstream checkouts. Do not commit vendor code, caches, or downloaded weights.

Upstream source licenses are recorded in `config/upstreams.yaml`; they are separate from
this repository's license and from model/data permissions. In particular, the pinned Seed-VC
source is recorded as GPL-3.0-only and archived. Environment isolation avoids dependency
conflicts; it does not remove upstream licensing obligations.
