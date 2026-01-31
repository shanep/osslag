from __future__ import annotations

import os

import pandas as pd
import requests

FEDORA_DISTGIT = os.environ.get("FEDORA_DISTGIT", "https://src.fedoraproject.org")


def list_rpms(
    namespace: str = "rpms", page: int = 1, per_page: int = 100
) -> list[dict]:
    # Pagure API: https://src.fedoraproject.org/api/0/projects?namespace=rpms
    url = f"{FEDORA_DISTGIT}/api/0/projects?namespace={namespace}&page={page}&per_page={per_page}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json().get("projects", [])


def fetch_all_rpms(max_pages: int = 200) -> pd.DataFrame:
    rows = []
    for p in range(1, max_pages + 1):
        items = list_rpms(page=p)
        if not items:
            break
        for prj in items:
            rows.append(
                {
                    "pkg_name": prj.get("name"),
                    "fullname": prj.get("fullname"),
                    "url": prj.get("url"),
                    "summary": prj.get("summary"),
                    "upstream_url": prj.get("upstream_url"),
                    "namespace": prj.get("namespace"),
                }
            )
    return pd.DataFrame(rows)
