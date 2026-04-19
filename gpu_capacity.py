#!/usr/bin/env python3
"""
gpu-capacity — Probe real Azure GPU capacity using ARM deployment validation.

Uses dry-run deployment validation to discover how many GPU VMs you can
actually deploy RIGHT NOW in each region. This goes beyond quota numbers
by checking physical hardware availability.

Usage:
    gpu-capacity probe --gpu A100                           # Probe A100 across all regions
    gpu-capacity probe --gpu A100 -r europe                 # Probe A100 in European regions
    gpu-capacity probe --gpu H100 -r us                     # Probe H100 in US regions
    gpu-capacity probe --gpu A100 -r uksouth,swedencentral  # Specific regions
    gpu-capacity probe --sku Standard_NC24ads_A100_v4 -r swedencentral
    gpu-capacity probe --gpu A100 -r swedencentral --count 32  # Test exact count
    gpu-capacity quota --gpu T4 -r europe --available       # Only show available quota
    gpu-capacity pricing --gpu A100 -r us                   # Show pricing

Region groups: europe (eu), us (na), americas, asia (apac), middleeast, australia, africa
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU SKU definitions
# ---------------------------------------------------------------------------

GPU_SKUS: dict[str, dict] = {
    "Standard_NC6s_v3":            {"gpu": "V100",       "gpus": 1, "vram": 16,  "vcpus": 6,  "series": "NCv3",     "use": "Train/Infer"},
    "Standard_NC12s_v3":           {"gpu": "V100",       "gpus": 2, "vram": 32,  "vcpus": 12, "series": "NCv3",     "use": "Train/Infer"},
    "Standard_NC24s_v3":           {"gpu": "V100",       "gpus": 4, "vram": 64,  "vcpus": 24, "series": "NCv3",     "use": "Train/Infer"},
    "Standard_NC4as_T4_v3":        {"gpu": "T4",         "gpus": 1, "vram": 16,  "vcpus": 4,  "series": "NCasT4v3", "use": "Inference"},
    "Standard_NC8as_T4_v3":        {"gpu": "T4",         "gpus": 1, "vram": 16,  "vcpus": 8,  "series": "NCasT4v3", "use": "Inference"},
    "Standard_NC16as_T4_v3":       {"gpu": "T4",         "gpus": 1, "vram": 16,  "vcpus": 16, "series": "NCasT4v3", "use": "Inference"},
    "Standard_NC64as_T4_v3":       {"gpu": "T4",         "gpus": 4, "vram": 64,  "vcpus": 64, "series": "NCasT4v3", "use": "Inference"},
    "Standard_NC24ads_A100_v4":    {"gpu": "A100 80GB",  "gpus": 1, "vram": 80,  "vcpus": 24, "series": "NCA100v4", "use": "Train/Infer"},
    "Standard_NC48ads_A100_v4":    {"gpu": "A100 80GB",  "gpus": 2, "vram": 160, "vcpus": 48, "series": "NCA100v4", "use": "Train/Infer"},
    "Standard_NC96ads_A100_v4":    {"gpu": "A100 80GB",  "gpus": 4, "vram": 320, "vcpus": 96, "series": "NCA100v4", "use": "Train/Infer"},
    "Standard_ND96asr_v4":         {"gpu": "A100 40GB",  "gpus": 8, "vram": 320, "vcpus": 96, "series": "NDA100v4", "use": "Training"},
    "Standard_ND96amsr_A100_v4":   {"gpu": "A100 80GB",  "gpus": 8, "vram": 640, "vcpus": 96, "series": "NDA100v4", "use": "Training"},
    "Standard_ND96isr_H100_v5":    {"gpu": "H100 80GB",  "gpus": 8, "vram": 640, "vcpus": 96, "series": "NDH100v5", "use": "Training"},
    "Standard_NV6ads_A10_v5":      {"gpu": "A10",        "gpus": 1, "vram": 24,  "vcpus": 6,  "series": "NVA10v5",  "use": "Infer/VDI"},
    "Standard_NV12ads_A10_v5":     {"gpu": "A10",        "gpus": 1, "vram": 24,  "vcpus": 12, "series": "NVA10v5",  "use": "Infer/VDI"},
    "Standard_NV18ads_A10_v5":     {"gpu": "A10",        "gpus": 1, "vram": 24,  "vcpus": 18, "series": "NVA10v5",  "use": "Infer/VDI"},
    "Standard_NV36ads_A10_v5":     {"gpu": "A10",        "gpus": 1, "vram": 24,  "vcpus": 36, "series": "NVA10v5",  "use": "Infer/VDI"},
    "Standard_NV72ads_A10_v5":     {"gpu": "A10",        "gpus": 2, "vram": 48,  "vcpus": 72, "series": "NVA10v5",  "use": "Infer/VDI"},
}

# ---------------------------------------------------------------------------
# Region groups — shortcuts for geographic areas
# ---------------------------------------------------------------------------

REGION_GROUPS: dict[str, list[str]] = {
    "europe": [
        "northeurope", "westeurope", "uksouth", "ukwest", "francecentral",
        "francesouth", "germanywestcentral", "germanynorth", "swedencentral",
        "norwayeast", "norwaywest", "switzerlandnorth", "switzerlandwest",
        "polandcentral", "italynorth", "spaincentral",
    ],
    "us": [
        "eastus", "eastus2", "centralus", "northcentralus", "southcentralus",
        "westus", "westus2", "westus3", "westcentralus",
    ],
    "asia": [
        "eastasia", "southeastasia", "japaneast", "japanwest", "koreacentral",
        "koreasouth", "centralindia", "southindia", "westindia",
    ],
    "americas": [
        "eastus", "eastus2", "centralus", "northcentralus", "southcentralus",
        "westus", "westus2", "westus3", "westcentralus",
        "canadacentral", "canadaeast", "brazilsouth", "brazilsoutheast",
    ],
    "middleeast": [
        "uaenorth", "uaecentral", "qatarcentral", "israelcentral",
    ],
    "africa": [
        "southafricanorth", "southafricawest",
    ],
    "australia": [
        "australiaeast", "australiasoutheast", "australiacentral",
        "australiacentral2",
    ],
}

# Allow some common aliases
REGION_GROUPS["eu"] = REGION_GROUPS["europe"]
REGION_GROUPS["na"] = REGION_GROUPS["us"]
REGION_GROUPS["apac"] = REGION_GROUPS["asia"]


def resolve_region_filter(region_arg: str | None) -> list[str] | None:
    """
    Parse the --region argument into a list of region names.

    Accepts:
      - None → return None (means "all regions")
      - A group name like "europe", "us", "asia"
      - A comma-separated list like "uksouth,swedencentral,westeurope"
      - A single region like "uksouth"
    """
    if not region_arg:
        return None

    region_arg = region_arg.strip().lower()

    # Check if it's a group name
    if region_arg in REGION_GROUPS:
        return REGION_GROUPS[region_arg]

    # Comma-separated list or single region
    regions = [r.strip() for r in region_arg.split(",") if r.strip()]
    return regions


SERIES_TO_QUOTA_FAMILY: dict[str, str] = {
    "NCv3":     "standardNCv3Family",
    "NCasT4v3": "standardNCAST4v3Family",
    "NCA100v4": "standardNCA100v4Family",
    "NDA100v4": "standardNDA100v4Family",
    "NDH100v5": "standardNDH100v5Family",
    "NVA10v5":  "standardNVA10v5Family",
}


# ---------------------------------------------------------------------------
# ARM deployment validation probe
# ---------------------------------------------------------------------------

# ARM template for VMSS dry-run validation
VMSS_PROBE_TEMPLATE = {
    "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
    "contentVersion": "1.0.0.0",
    "parameters": {
        "vmSize": {"type": "string"},
        "location": {"type": "string"},
        "instanceCount": {"type": "int"},
    },
    "resources": [
        {
            "type": "Microsoft.Compute/virtualMachineScaleSets",
            "apiVersion": "2023-09-01",
            "name": "[concat('gpu-probe-', uniqueString(parameters('location'), parameters('vmSize')))]",
            "location": "[parameters('location')]",
            "sku": {
                "name": "[parameters('vmSize')]",
                "tier": "Standard",
                "capacity": "[parameters('instanceCount')]",
            },
            "properties": {
                "overprovision": False,
                "upgradePolicy": {"mode": "Manual"},
                "virtualMachineProfile": {
                    "osProfile": {
                        "computerNamePrefix": "probe",
                        "adminUsername": "probeadmin",
                        "adminPassword": "Pr0b3-N0tR3al!",
                    },
                    "storageProfile": {
                        "imageReference": {
                            "publisher": "Canonical",
                            "offer": "0001-com-ubuntu-server-jammy",
                            "sku": "22_04-lts-gen2",
                            "version": "latest",
                        },
                        "osDisk": {
                            "createOption": "FromImage",
                            "managedDisk": {"storageAccountType": "Standard_LRS"},
                        },
                    },
                    "networkProfile": {
                        "networkInterfaceConfigurations": [
                            {
                                "name": "probe-nic",
                                "properties": {
                                    "primary": True,
                                    "ipConfigurations": [
                                        {
                                            "name": "probe-ip",
                                            "properties": {
                                                "subnet": {
                                                    "id": "[resourceId('Microsoft.Network/virtualNetworks/subnets', 'probe-vnet', 'default')]"
                                                }
                                            },
                                        }
                                    ],
                                },
                            }
                        ]
                    },
                },
            },
        }
    ],
}


@dataclass
class ProbeResult:
    """Result of a capacity probe for one SKU in one region."""
    region: str
    sku: str
    gpu: str
    gpus_per_vm: int
    vram_gb: int
    vcpus_per_vm: int
    max_vms: int           # max VMs validated via dry-run (0 = none available)
    max_gpus: int          # max_vms * gpus_per_vm
    quota_limit_vcpus: int # vCPU quota limit for the family
    quota_used_vcpus: int  # vCPUs currently in use
    status: str            # deployable | quota_only | restricted | policy_blocked | unavailable
    error: str = ""


def _validate_deployment(subscription_id: str, resource_group: str, location: str,
                          vm_size: str, count: int) -> tuple[bool, str]:
    """
    Run ARM deployment validation (dry-run) for a VMSS with the given instance count.

    Returns (success: bool, error_code: str).
    Error codes:
      "" = validation passed
      "SkuNotAvailable" = no physical capacity
      "QuotaExceeded" = quota limit reached
      "OperationNotAllowed" = quota or policy
      "RequestDisallowedByPolicy" = Azure policy blocks this
      other = unexpected error
    """
    import tempfile, os

    template = VMSS_PROBE_TEMPLATE.copy()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(template, f)
        template_path = f.name

    try:
        result = subprocess.run(
            [
                "az", "deployment", "group", "validate",
                "--resource-group", resource_group,
                "--template-file", template_path,
                "--parameters",
                f"vmSize={vm_size}",
                f"location={location}",
                f"instanceCount={count}",
                "--no-prompt",
                "-o", "json",
            ],
            capture_output=True, text=True, timeout=60,
        )
    finally:
        os.unlink(template_path)

    if result.returncode == 0:
        return True, ""

    # Parse error from stderr or stdout
    error_text = result.stderr + result.stdout
    for code in [
        "SkuNotAvailable",
        "OverconstrainedAllocationRequest",
        "QuotaExceeded",
        "OperationNotAllowed",
        "RequestDisallowedByPolicy",
        "InvalidTemplateDeployment",
    ]:
        if code in error_text:
            # Distinguish between quota and capacity errors within OperationNotAllowed
            if code == "OperationNotAllowed" and "quota" in error_text.lower():
                return False, "QuotaExceeded"
            return False, code

    return False, "Unknown"


def probe_capacity(
    subscription_id: str,
    resource_group: str,
    sku: str,
    region: str,
    max_count: int = 100,
    exact_count: int | None = None,
    console=None,
) -> ProbeResult:
    """
    Probe VM capacity using ARM dry-run validation.

    If exact_count is set, tests that specific count only.
    Otherwise, walks through fixed steps (1, 5, 10, 25, 50, 100) up to max_count
    and reports the highest passing count.
    """
    info = GPU_SKUS[sku]

    # Single exact count mode
    if exact_count is not None:
        if console:
            console.print(f"    [dim]Probing {region}/{sku}: trying {exact_count} VMs...[/dim]", end="\r")
        success, error = _validate_deployment(subscription_id, resource_group, region, sku, exact_count)
        if success:
            return ProbeResult(
                region=region, sku=sku, gpu=info["gpu"], gpus_per_vm=info["gpus"],
                vram_gb=info["vram"], vcpus_per_vm=info["vcpus"],
                max_vms=exact_count, max_gpus=exact_count * info["gpus"],
                quota_limit_vcpus=0, quota_used_vcpus=0,
                status="deployable",
            )
        status = "unavailable"
        if error == "RequestDisallowedByPolicy":
            status = "policy_blocked"
        elif error == "QuotaExceeded":
            status = "quota_exceeded"
        elif error in ("SkuNotAvailable", "OverconstrainedAllocationRequest"):
            status = "unavailable"
        return ProbeResult(
            region=region, sku=sku, gpu=info["gpu"], gpus_per_vm=info["gpus"],
            vram_gb=info["vram"], vcpus_per_vm=info["vcpus"],
            max_vms=0, max_gpus=0,
            quota_limit_vcpus=0, quota_used_vcpus=0,
            status=status, error=error,
        )

    # Step-based probing: walk through fixed counts
    steps = [s for s in [1, 5, 10, 25, 50, 100] if s <= max_count]
    if max_count not in steps:
        steps.append(max_count)

    best = 0
    last_error = None

    for count in steps:
        if console:
            console.print(f"    [dim]Probing {region}/{sku}: trying {count} VMs...[/dim]", end="\r")
        success, error = _validate_deployment(subscription_id, resource_group, region, sku, count)
        if success:
            best = count
        else:
            last_error = error
            break

    if best == 0:
        status = "unavailable"
        if last_error == "RequestDisallowedByPolicy":
            status = "policy_blocked"
        elif last_error == "QuotaExceeded":
            status = "quota_exceeded"
        elif last_error in ("SkuNotAvailable", "OverconstrainedAllocationRequest"):
            status = "unavailable"
        return ProbeResult(
            region=region, sku=sku, gpu=info["gpu"], gpus_per_vm=info["gpus"],
            vram_gb=info["vram"], vcpus_per_vm=info["vcpus"],
            max_vms=0, max_gpus=0,
            quota_limit_vcpus=0, quota_used_vcpus=0,
            status=status, error=last_error,
        )

    return ProbeResult(
        region=region, sku=sku, gpu=info["gpu"], gpus_per_vm=info["gpus"],
        vram_gb=info["vram"], vcpus_per_vm=info["vcpus"],
        max_vms=best, max_gpus=best * info["gpus"],
        quota_limit_vcpus=0, quota_used_vcpus=0,
        status="deployable",
    )


# ---------------------------------------------------------------------------
# Quick quota check (no probing)
# ---------------------------------------------------------------------------

@dataclass
class QuotaInfo:
    region: str
    sku: str
    gpu: str
    gpus_per_vm: int
    vram_gb: int
    vcpus_per_vm: int
    quota_limit: int
    quota_used: int
    quota_available: int
    utilization_pct: float
    restriction: str
    status: str


def collect_quota(subscription_id: str, console=None) -> list[QuotaInfo]:
    """Quick quota check using Compute SKU + Usage APIs (no dry-run probing)."""
    from azure.identity import AzureCliCredential
    from azure.mgmt.compute import ComputeManagementClient

    credential = AzureCliCredential()
    compute = ComputeManagementClient(credential, subscription_id)

    if console:
        console.print("  [dim]Fetching SKU availability...[/dim]", end="")
    sku_map: dict[str, str] = {}
    for sku in compute.resource_skus.list():
        if sku.resource_type != "virtualMachines" or sku.name not in GPU_SKUS:
            continue
        for li in (sku.location_info or []):
            loc = li.location
            if loc:
                region = loc.lower()
                restriction = "None"
                for r in (sku.restrictions or []):
                    if r.type and r.type.value == "Location":
                        if region in [v.lower() for v in (r.values or [])]:
                            restriction = r.reason_code.value if r.reason_code else "Unknown"
                sku_map[f"{region}::{sku.name}"] = restriction
    if console:
        console.print(f" {len(sku_map)} entries")

    regions = set(k.split("::")[0] for k in sku_map)
    if console:
        console.print(f"  [dim]Fetching quota for {len(regions)} regions...[/dim]", end="")

    quota_map: dict[str, dict] = {}
    target_families = set(SERIES_TO_QUOTA_FAMILY.values())

    def fetch_quota(region):
        results = {}
        try:
            for u in compute.usage.list(region):
                fname = u.name.value if u.name else ""
                if fname in target_families:
                    results[f"{region}::{fname}"] = {"limit": u.limit, "usage": u.current_value}
        except Exception:
            pass
        return results

    with ThreadPoolExecutor(max_workers=10) as pool:
        for fut in as_completed({pool.submit(fetch_quota, r): r for r in regions}):
            quota_map.update(fut.result())
    if console:
        console.print(" done")

    records = []
    for sku_name, info in GPU_SKUS.items():
        qfamily = SERIES_TO_QUOTA_FAMILY.get(info["series"], "")
        for key, restriction in sku_map.items():
            r, s = key.split("::")
            if s != sku_name:
                continue
            q = quota_map.get(f"{r}::{qfamily}", {"limit": 0, "usage": 0})
            limit, usage = q["limit"], q["usage"]
            avail = max(0, limit - usage)
            util = (usage / limit * 100) if limit > 0 else 0.0

            if restriction != "None":
                status = "restricted"
            elif limit == 0:
                status = "not_offered"
            elif util < 50:
                status = "available"
            elif util < 90:
                status = "limited"
            else:
                status = "exhausted"

            records.append(QuotaInfo(
                region=r, sku=sku_name, gpu=info["gpu"], gpus_per_vm=info["gpus"],
                vram_gb=info["vram"], vcpus_per_vm=info["vcpus"],
                quota_limit=limit, quota_used=usage, quota_available=avail,
                utilization_pct=round(util, 1), restriction=restriction, status=status,
            ))
    return records


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

@dataclass
class GpuPrice:
    region: str
    sku: str
    gpu: str
    ondemand: Optional[float]
    spot: Optional[float]
    reserved_1yr: Optional[float]
    reserved_3yr: Optional[float]
    currency: str


def collect_pricing(console=None) -> list[GpuPrice]:
    """Fetch pricing from the public Azure Retail Prices API."""
    sku_names = list(GPU_SKUS.keys())
    all_items: list[dict] = []

    if console:
        console.print("  [dim]Fetching pricing...[/dim]", end="")
    for i in range(0, len(sku_names), 5):
        batch = sku_names[i : i + 5]
        sku_filters = " or ".join(f"armSkuName eq '{s}'" for s in batch)
        filt = f"serviceFamily eq 'Compute' and ({sku_filters}) and priceType ne 'DevTestConsumption'"
        skip = 0
        while True:
            try:
                url = f"https://prices.azure.com/api/retail/prices?$filter={quote(filt)}&$skip={skip}"
                with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                items = data.get("Items", [])
                if not items:
                    break
                all_items.extend(items)
                if not data.get("NextPageLink"):
                    break
                skip += len(items)
            except Exception:
                break
    if console:
        console.print(f" {len(all_items)} entries")

    grouped: dict[str, dict] = {}
    for item in all_items:
        sku = item.get("armSkuName", "")
        if sku not in GPU_SKUS:
            continue
        region = item.get("armRegionName", "").lower()
        if not region:
            continue
        key = f"{region}::{sku}"
        if key not in grouped:
            grouped[key] = {"region": region, "sku": sku, "gpu": GPU_SKUS[sku]["gpu"],
                            "od": None, "sp": None, "r1": None, "r3": None,
                            "cur": item.get("currencyCode", "USD")}
        price = item.get("retailPrice", 0)
        ptype, rterm = item.get("type", ""), item.get("reservationTerm", "")
        sdesc = item.get("skuName", "").lower()
        if "spot" in sdesc or "low priority" in sdesc:
            grouped[key]["sp"] = price
        elif ptype == "Consumption":
            meter = item.get("meterName", "").lower()
            if "spot" in meter or "low priority" in meter:
                grouped[key]["sp"] = price
            else:
                grouped[key]["od"] = price
        elif ptype == "Reservation":
            if "1 Year" in rterm:
                grouped[key]["r1"] = round(price / (365 * 24), 4)
            elif "3 Year" in rterm:
                grouped[key]["r3"] = round(price / (3 * 365 * 24), 4)

    return [GpuPrice(d["region"], d["sku"], d["gpu"], d["od"], d["sp"], d["r1"], d["r3"], d["cur"])
            for d in grouped.values()]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

STATUS_STYLE = {
    "deployable": ("green", "DEPLOY OK"),
    "available": ("green", "available"),
    "limited": ("yellow", "limited"),
    "quota_exceeded": ("red", "quota full"),
    "exhausted": ("red", "exhausted"),
    "unavailable": ("red", "no capacity"),
    "restricted": ("bright_black", "restricted"),
    "policy_blocked": ("bright_black", "policy deny"),
    "not_offered": ("bright_black", "not offered"),
}


def display_probe_results(results: list[ProbeResult]):
    from rich.console import Console
    from rich.table import Table

    if not results:
        print("No results.")
        return

    results.sort(key=lambda r: (-r.max_gpus, r.region))

    table = Table(title="GPU Capacity Probe Results (ARM dry-run)", padding=(0, 1))
    table.add_column("Region", style="cyan", no_wrap=True)
    table.add_column("GPU", style="bold")
    table.add_column("SKU", no_wrap=True, max_width=28)
    table.add_column("Max VMs", justify="right")
    table.add_column("Max GPUs", justify="right")
    table.add_column("Total VRAM", justify="right")
    table.add_column("Status", no_wrap=True)
    table.add_column("Error", style="dim", max_width=25)

    for r in results:
        color, label = STATUS_STYLE.get(r.status, ("white", r.status))
        table.add_row(
            r.region,
            r.gpu,
            r.sku.replace("Standard_", ""),
            f"[bold green]{r.max_vms}[/bold green]" if r.max_vms > 0 else "0",
            f"[bold green]{r.max_gpus}[/bold green]" if r.max_gpus > 0 else "0",
            f"{r.max_vms * r.vram_gb}GB" if r.max_vms > 0 else "-",
            f"[{color}]{label}[/{color}]",
            r.error if r.error else "",
        )

    console = Console()
    console.print(table)

    deployable = [r for r in results if r.max_vms > 0]
    if deployable:
        best = max(deployable, key=lambda r: r.max_gpus)
        console.print(f"\n[bold green]Best:[/bold green] {best.region} — "
                      f"{best.max_vms} VMs, {best.max_gpus} GPUs, "
                      f"{best.max_vms * best.vram_gb}GB VRAM")
    else:
        console.print("\n[bold red]No deployable capacity found.[/bold red]")


def display_quota(records: list[QuotaInfo], sort_by: str = "status"):
    from rich.console import Console
    from rich.table import Table

    if not records:
        print("No data.")
        return

    order = {"available": 0, "limited": 1, "exhausted": 2, "restricted": 3, "not_offered": 4}
    if sort_by == "status":
        records.sort(key=lambda r: (order.get(r.status, 5), r.gpu, r.region))
    elif sort_by == "region":
        records.sort(key=lambda r: (r.region, r.gpu))
    elif sort_by == "gpu":
        records.sort(key=lambda r: (r.gpu, order.get(r.status, 5), r.region))
    elif sort_by == "available":
        records.sort(key=lambda r: -r.quota_available)

    table = Table(title="GPU Quota (quick view — use 'probe' for real capacity)", padding=(0, 1))
    table.add_column("Region", style="cyan", no_wrap=True)
    table.add_column("GPU", style="bold")
    table.add_column("GPUs/VM", justify="right")
    table.add_column("VRAM", justify="right")
    table.add_column("SKU", no_wrap=True, max_width=28)
    table.add_column("Quota vCPU", justify="right")
    table.add_column("Used", justify="right")
    table.add_column("Free vCPU", justify="right")
    table.add_column("Util%", justify="right")
    table.add_column("Status", no_wrap=True)

    for r in records:
        color, label = STATUS_STYLE.get(r.status, ("white", r.status))
        uc = "green" if r.utilization_pct < 50 else "yellow" if r.utilization_pct < 90 else "red"
        table.add_row(
            r.region, r.gpu, str(r.gpus_per_vm), f"{r.vram_gb}GB",
            r.sku.replace("Standard_", ""),
            str(r.quota_limit), str(r.quota_used),
            f"[bold]{r.quota_available}[/bold]" if r.quota_available > 0 else "0",
            f"[{uc}]{r.utilization_pct}%[/{uc}]" if r.quota_limit > 0 else "-",
            f"[{color}]{label}[/{color}]",
        )

    console = Console()
    console.print(table)
    total = len(records)
    a = sum(1 for r in records if r.status == "available")
    l = sum(1 for r in records if r.status == "limited")
    console.print(f"\n[bold]{total}[/bold] entries: [green]{a} available[/green], [yellow]{l} limited[/yellow]")


def display_pricing(prices: list[GpuPrice], sort_by: str = "price"):
    from rich.console import Console
    from rich.table import Table

    if not prices:
        print("No pricing data.")
        return

    if sort_by == "price":
        prices.sort(key=lambda p: p.ondemand or 9999)
    elif sort_by == "region":
        prices.sort(key=lambda p: (p.region, p.ondemand or 9999))
    elif sort_by == "gpu":
        prices.sort(key=lambda p: (p.gpu, p.ondemand or 9999))

    table = Table(title="Azure GPU Pricing (USD/hr)", padding=(0, 1))
    table.add_column("Region", style="cyan", no_wrap=True)
    table.add_column("GPU", style="bold")
    table.add_column("SKU", no_wrap=True, max_width=28)
    table.add_column("On-Demand", justify="right")
    table.add_column("Spot", justify="right", style="green")
    table.add_column("1yr RI", justify="right", style="yellow")
    table.add_column("3yr RI", justify="right", style="yellow")
    table.add_column("Spot Save%", justify="right")

    for p in prices:
        sp = ""
        if p.ondemand and p.spot and p.ondemand > 0:
            sp = f"[green]{(1 - p.spot / p.ondemand) * 100:.0f}%[/green]"
        table.add_row(
            p.region, p.gpu, p.sku.replace("Standard_", ""),
            f"${p.ondemand:.4f}" if p.ondemand else "-",
            f"${p.spot:.4f}" if p.spot else "-",
            f"${p.reserved_1yr:.4f}" if p.reserved_1yr else "-",
            f"${p.reserved_3yr:.4f}" if p.reserved_3yr else "-",
            sp,
        )
    Console().print(table)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_sub_info() -> tuple[str, str]:
    r = subprocess.run(["az", "account", "show", "--query", "{id:id,name:name}", "-o", "json"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("Error: Run 'az login' first.", file=sys.stderr)
        sys.exit(1)
    d = json.loads(r.stdout)
    return d["id"], d["name"]


def resolve_skus(gpu_filter: str | None, sku_filter: str | None) -> list[str]:
    """Resolve GPU/SKU filter to list of matching SKU names."""
    matches = list(GPU_SKUS.keys())
    if sku_filter:
        matches = [s for s in matches if sku_filter.lower() in s.lower()]
    if gpu_filter:
        matches = [s for s in matches if gpu_filter.lower() in GPU_SKUS[s]["gpu"].lower()]
    return matches


def resolve_regions(region_filter: list[str] | None, subscription_id: str, skus: list[str]) -> list[str]:
    """
    Get list of regions where the given SKUs are offered.

    If region_filter is provided, only return regions from that list that
    actually have the SKUs available. If None, return all regions.
    """
    from azure.identity import AzureCliCredential
    from azure.mgmt.compute import ComputeManagementClient

    credential = AzureCliCredential()
    compute = ComputeManagementClient(credential, subscription_id)
    sku_set = set(skus)
    available_regions = set()

    for sku in compute.resource_skus.list():
        if sku.resource_type != "virtualMachines" or sku.name not in sku_set:
            continue
        for li in (sku.location_info or []):
            if li.location:
                available_regions.add(li.location.lower())

    if region_filter:
        # Intersect requested regions with regions where SKUs exist
        filter_set = set(r.lower() for r in region_filter)
        return sorted(available_regions & filter_set)

    return sorted(available_regions)


def ensure_resource_group(subscription_id: str) -> str:
    """Find or create a temporary resource group for validation probes."""
    rg_name = "gpu-capacity-probe-rg"

    # Check if it exists
    result = subprocess.run(
        ["az", "group", "exists", "--name", rg_name],
        capture_output=True, text=True,
    )
    if result.stdout.strip() == "true":
        return rg_name

    # Create in a neutral region (the VMSS location is parameterized, RG location doesn't matter)
    subprocess.run(
        ["az", "group", "create", "--name", rg_name, "--location", "eastus", "-o", "none"],
        capture_output=True, text=True,
    )
    return rg_name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="gpu-capacity",
        description="Probe real Azure GPU capacity using ARM deployment validation (dry-run).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Region groups:
  europe (eu)     — North/West Europe, UK, France, Germany, Sweden, Norway, etc.
  us (na)         — East US, West US, Central US, etc.
  americas        — US + Canada + Brazil
  asia (apac)     — East/Southeast Asia, Japan, Korea, India
  middleeast      — UAE, Qatar, Israel
  australia       — Australia East/Southeast/Central
  africa          — South Africa

Examples:
  gpu-capacity probe --gpu A100                         # Probe A100 across ALL regions
  gpu-capacity probe --gpu A100 -r europe               # Probe A100 in European regions
  gpu-capacity probe --gpu H100 -r us                   # Probe H100 in US regions
  gpu-capacity probe --gpu A100 -r uksouth,swedencentral  # Specific regions
  gpu-capacity probe --sku Standard_NC24ads_A100_v4 --max 50
  gpu-capacity quota --gpu T4 -r europe --available     # Quick quota view
  gpu-capacity pricing --gpu A100 -r us                 # Pricing table
""",
    )
    subs = parser.add_subparsers(dest="command")

    # probe
    p_probe = subs.add_parser("probe", help="Probe real deployable capacity via ARM dry-run")
    p_probe.add_argument("--gpu", "-g", help="GPU name filter (e.g. A100, H100, T4)")
    p_probe.add_argument("--sku", "-s", help="SKU name filter")
    p_probe.add_argument("--region", "-r", help="Region, group (europe/us/asia), or comma-separated list")
    p_probe.add_argument("--count", "-c", type=int, help="Test a specific VM count (skip step probing)")
    p_probe.add_argument("--max", type=int, default=100, help="Max VMs to probe (default: 100)")
    p_probe.add_argument("--subscription", help="Subscription ID")

    # quota
    p_quota = subs.add_parser("quota", help="Quick quota view (no probing)")
    p_quota.add_argument("--gpu", "-g", help="GPU name filter")
    p_quota.add_argument("--sku", "-s", help="SKU name filter")
    p_quota.add_argument("--region", "-r", help="Region, group (europe/us/asia), or comma-separated list")
    p_quota.add_argument("--available", "-a", action="store_true", help="Only show available")
    p_quota.add_argument("--sort", choices=["status", "region", "gpu", "available"], default="status")
    p_quota.add_argument("--subscription", help="Subscription ID")

    # pricing
    p_price = subs.add_parser("pricing", help="GPU pricing table")
    p_price.add_argument("--gpu", "-g", help="GPU name filter")
    p_price.add_argument("--region", "-r", help="Region, group (europe/us/asia), or comma-separated list")
    p_price.add_argument("--sort", choices=["price", "region", "gpu"], default="price")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    try:
        from rich.console import Console
    except ImportError:
        print("Missing: pip install rich", file=sys.stderr)
        sys.exit(1)

    console = Console()

    if args.command == "probe":
        sub_id = args.subscription or get_sub_info()[0]
        sub_name = get_sub_info()[1] if not args.subscription else sub_id
        console.print(f"[bold]Subscription:[/bold] {sub_name}")

        skus = resolve_skus(args.gpu, args.sku)
        if not skus:
            console.print("[red]No matching GPU SKUs found.[/red]")
            sys.exit(1)
        console.print(f"[bold]SKUs to probe:[/bold] {len(skus)}")

        region_filter = resolve_region_filter(args.region)
        console.print("[dim]Resolving regions...[/dim]")
        regions = resolve_regions(region_filter, sub_id, skus)
        if not regions:
            console.print("[red]No matching regions found.[/red]")
            if region_filter:
                console.print(f"[dim]Requested: {', '.join(region_filter)}[/dim]")
                console.print("[dim]Available groups: europe, us, asia, americas, middleeast, australia, africa[/dim]")
            sys.exit(1)
        console.print(f"[bold]Regions:[/bold] {len(regions)} — {', '.join(regions)}")

        rg = ensure_resource_group(sub_id)
        console.print(f"[dim]Using resource group: {rg}[/dim]")

        total_probes = len(skus) * len(regions)
        console.print(f"\n[bold]Probing {total_probes} SKU-region combinations...[/bold]\n")

        results: list[ProbeResult] = []
        done = 0
        for sku in skus:
            for region in regions:
                done += 1
                sku_short = sku.replace("Standard_", "")
                console.print(f"  [{done}/{total_probes}] {region} / {sku_short}...", end=" ")
                result = probe_capacity(sub_id, rg, sku, region, max_count=args.max, exact_count=args.count, console=console)
                if result.max_vms > 0:
                    console.print(f"[green]{result.max_vms} VMs ({result.max_gpus} GPUs)[/green]")
                else:
                    color, label = STATUS_STYLE.get(result.status, ("red", result.status))
                    console.print(f"[{color}]{label}[/{color}]")
                results.append(result)

        console.print()
        display_probe_results(results)

    elif args.command == "quota":
        sub_id = args.subscription or get_sub_info()[0]
        sub_name = get_sub_info()[1] if not args.subscription else sub_id
        console.print(f"[bold]Subscription:[/bold] {sub_name}")

        records = collect_quota(sub_id, console=console)
        if args.gpu:
            records = [r for r in records if args.gpu.lower() in r.gpu.lower()]
        if args.sku:
            records = [r for r in records if args.sku.lower() in r.sku.lower()]
        if args.region:
            region_filter = resolve_region_filter(args.region)
            if region_filter:
                filter_set = set(r.lower() for r in region_filter)
                records = [r for r in records if r.region.lower() in filter_set]
        if args.available:
            records = [r for r in records if r.status == "available"]
        display_quota(records, sort_by=args.sort)

    elif args.command == "pricing":
        console.print("[bold]Fetching pricing...[/bold]")
        prices = collect_pricing(console=console)
        if args.gpu:
            prices = [p for p in prices if args.gpu.lower() in p.gpu.lower()]
        if args.region:
            region_filter = resolve_region_filter(args.region)
            if region_filter:
                filter_set = set(r.lower() for r in region_filter)
                prices = [p for p in prices if p.region.lower() in filter_set]
        display_pricing(prices, sort_by=args.sort)


if __name__ == "__main__":
    main()
