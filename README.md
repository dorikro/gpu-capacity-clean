# gpu-capacity

A CLI tool to check real-time Azure GPU VM availability across regions.

## What does this tool do?

This tool answers: **"How many GPU VMs can I deploy right now in a given Azure region?"**

It has three commands:

| Command | What it does | How it works |
|---------|-------------|--------------|
| `probe` | Checks **actual physical GPU capacity** | Runs ARM deployment validation (dry-run) — no resources are created |
| `quota` | Shows vCPU quota limits and usage | Reads Azure Compute SKU and Usage APIs |
| `pricing` | Shows GPU VM pricing | Reads the public Azure Retail Prices API |

### How `probe` works

The tool submits a VMSS deployment template to Azure Resource Manager with `validate` mode. ARM checks quota, policy, SKU availability, and physical capacity — then returns success or a specific error. A binary search finds the maximum deployable count.

**No VMs are created. No resources are deployed. This is a read-only dry-run.**

A temporary empty resource group (`gpu-capacity-probe-rg`) is created to anchor the validation calls. It contains no resources and can be safely deleted.

## Cost

| Command | Azure charges | Authentication required |
|---------|--------------|----------------------|
| `probe` | **None** — ARM validation (dry-run) is free. No resources are deployed. | Yes (`az login`) |
| `quota` | **None** — read-only API calls to Compute SKU and Usage APIs. | Yes (`az login`) |
| `pricing` | **None** — calls the public Azure Retail Prices API. | No |

**This tool incurs zero Azure charges.** It does not create, modify, or delete any billable resources.

## Required permissions

| Command | Azure APIs called | Minimum RBAC role |
|---------|-------------------|-------------------|
| `probe` | `Microsoft.Resources/deployments/validate` (ARM dry-run), `Microsoft.Resources/resourceGroups/write` (one-time RG creation) | **Contributor** on the subscription (or a custom role with deployment validate + resource group write) |
| `quota` | `Microsoft.Compute/skus/read`, `Microsoft.Compute/locations/usages/read` | **Reader** on the subscription |
| `pricing` | Public API (no auth) | None |

The `probe` command creates one empty resource group (`gpu-capacity-probe-rg`) to anchor validation calls. It does not create any other resources. If the resource group already exists, only the `deployments/validate` permission is needed.

## Prerequisites

- Python 3.10+
- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) with an active login (`az login`)

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Probe A100 capacity in European regions
python gpu_capacity.py probe --gpu A100 -r europe

# Probe H100 in US regions
python gpu_capacity.py probe --gpu H100 -r us

# Probe specific regions
python gpu_capacity.py probe --gpu A100 -r uksouth,swedencentral

# Quick quota check (faster, no dry-run)
python gpu_capacity.py quota --gpu T4 -r europe --available

# Pricing table (no auth needed)
python gpu_capacity.py pricing --gpu A100 -r us
```

### Region groups

Use `--region` / `-r` with a region name, comma-separated list, or group alias:

| Group | Alias | Example regions |
|-------|-------|----------------|
| `europe` | `eu` | northeurope, westeurope, uksouth, swedencentral, ... |
| `us` | `na` | eastus, eastus2, westus2, westus3, ... |
| `americas` | | US + canadacentral, brazilsouth, ... |
| `asia` | `apac` | eastasia, southeastasia, japaneast, ... |
| `middleeast` | | uaenorth, qatarcentral, israelcentral |
| `australia` | | australiaeast, australiasoutheast |
| `africa` | | southafricanorth, southafricawest |

## Supported GPUs

| GPU | Series | Use case |
|-----|--------|----------|
| V100 16GB | NCv3 | Training / Inference |
| T4 16GB | NCasT4v3 | Inference |
| A100 80GB | NCA100v4 | Training / Inference |
| A100 40/80GB | NDA100v4 | Training |
| H100 80GB | NDH100v5 | Training |
| A10 24GB | NVA10v5 | Inference / VDI |

## Notes

- `probe` takes ~5-15 seconds per SKU-region combination. Use `--gpu` and `--region` to narrow scope.
- Capacity results are a point-in-time snapshot and may change within minutes.
