# Vendor integrations

Upstream model repositories are installed here in isolated environments and are ignored by Git. Installers must pin a reviewed revision, record the upstream license, verify checksums where available, and never update a checkout implicitly during a generation run.

Installed by `scripts/install-models.ps1` at the commits in `config/upstreams.yaml`:

- `ACE-Step-1.5/`
- `SoulX-Singer/`
- `seed-vc/` (optional and disabled by default)
