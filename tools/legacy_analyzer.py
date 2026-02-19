#!/usr/bin/env python3
"""
Legacy project analyzer for AI context generation.

Outputs a single JSON document with:
- project/build metadata
- dependency inventory
- Java package/import dependency graph
- Spring/MyBatis configuration summary
- optional LSP-derived symbol summary
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


SKIP_DIRS = {
    ".git",
    ".idea",
    ".svn",
    "node_modules",
    "target",
    ".gradle",
    "build",
}


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def iter_files(root: Path, suffixes: Optional[Set[str]] = None) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        base = Path(dirpath)
        for name in filenames:
            p = base / name
            if suffixes is None:
                yield p
            elif p.suffix.lower() in suffixes:
                yield p


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def parse_pom(pom_path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "exists": pom_path.exists(),
        "coordinates": {},
        "properties": {},
        "repositories": [],
        "dependencies": [],
        "profiles": [],
        "plugins": [],
        "stats": {},
    }
    if not pom_path.exists():
        return result

    tree = ET.parse(pom_path)
    root = tree.getroot()

    children = list(root)
    child_by_name = {local_name(c.tag): c for c in children}
    for key in ("groupId", "artifactId", "version", "packaging", "name"):
        elem = child_by_name.get(key)
        result["coordinates"][key] = elem.text.strip() if elem is not None and elem.text else None

    props = child_by_name.get("properties")
    if props is not None:
        for c in list(props):
            k = local_name(c.tag)
            v = (c.text or "").strip()
            result["properties"][k] = v

    repos = child_by_name.get("repositories")
    if repos is not None:
        for repo in repos:
            if local_name(repo.tag) != "repository":
                continue
            item: Dict[str, Any] = {}
            for c in repo:
                k = local_name(c.tag)
                if k in ("id", "url"):
                    item[k] = (c.text or "").strip()
            if item:
                result["repositories"].append(item)

    deps_parent = child_by_name.get("dependencies")
    if deps_parent is not None:
        for dep in deps_parent:
            if local_name(dep.tag) != "dependency":
                continue
            item: Dict[str, Any] = {
                "groupId": None,
                "artifactId": None,
                "version": None,
                "scope": None,
                "type": None,
                "systemPath": None,
            }
            for c in dep:
                k = local_name(c.tag)
                if k in item:
                    item[k] = (c.text or "").strip() or None
            if item["groupId"] and item["artifactId"]:
                result["dependencies"].append(item)

    build = child_by_name.get("build")
    if build is not None:
        for c in build:
            if local_name(c.tag) == "plugins":
                for plugin in c:
                    if local_name(plugin.tag) != "plugin":
                        continue
                    pitem = {"groupId": None, "artifactId": None, "version": None}
                    for p in plugin:
                        k = local_name(p.tag)
                        if k in pitem:
                            pitem[k] = (p.text or "").strip() or None
                    if pitem["artifactId"]:
                        result["plugins"].append(pitem)

    profiles = child_by_name.get("profiles")
    if profiles is not None:
        for profile in profiles:
            if local_name(profile.tag) != "profile":
                continue
            pitem: Dict[str, Any] = {"id": None, "properties": {}}
            for c in profile:
                k = local_name(c.tag)
                if k == "id":
                    pitem["id"] = (c.text or "").strip()
                elif k == "properties":
                    for pc in c:
                        pitem["properties"][local_name(pc.tag)] = (pc.text or "").strip()
            result["profiles"].append(pitem)

    seen = set()
    for d in result["dependencies"]:
        seen.add((d["groupId"], d["artifactId"], d["scope"]))

    result["stats"] = {
        "dependency_count": len(result["dependencies"]),
        "unique_dependency_count": len(
            {(d["groupId"], d["artifactId"]) for d in result["dependencies"]}
        ),
        "system_scope_dependency_count": sum(1 for d in result["dependencies"] if d["scope"] == "system"),
        "plugin_count": len(result["plugins"]),
        "profile_count": len(result["profiles"]),
        "repository_count": len(result["repositories"]),
    }
    return result


JAVA_PACKAGE_RE = re.compile(r"^\s*package\s+([a-zA-Z0-9_.]+)\s*;", re.MULTILINE)
JAVA_IMPORT_RE = re.compile(r"^\s*import\s+(?:static\s+)?([a-zA-Z0-9_.*]+)\s*;", re.MULTILINE)
JAVA_ANNOT_RE = re.compile(r"@([A-Za-z_][A-Za-z0-9_]*)")
JAVA_TYPE_RE = re.compile(r"\b(class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)")


def infer_layer(package_name: str, class_name: str, annotations: Set[str], path: Path) -> str:
    lower_pkg = package_name.lower()
    lower_name = class_name.lower()
    ann = {a.lower() for a in annotations}
    spath = str(path).lower()

    if "controller" in ann or "restcontroller" in ann or lower_name.endswith("controller"):
        return "web.controller"
    if "service" in ann or lower_name.endswith("service") or lower_name.endswith("serviceimpl"):
        return "service"
    if "repository" in ann or lower_name.endswith("dao") or lower_name.endswith("mapper"):
        return "data"
    if ".web." in lower_pkg or "/web/" in spath:
        return "web"
    if ".service." in lower_pkg or "/service/" in spath:
        return "service"
    if ".mapper." in lower_pkg or ".dao." in lower_pkg:
        return "data"
    if ".util." in lower_pkg:
        return "util"
    if ".common." in lower_pkg:
        return "common"
    return "other"


def package_to_module(package_name: str) -> str:
    if not package_name:
        return "default"
    parts = package_name.split(".")
    if package_name.startswith("kr.go.culture.inmun360."):
        return ".".join(parts[:6]) if len(parts) >= 6 else package_name
    if package_name.startswith("kr.co.contentwise."):
        return ".".join(parts[:4]) if len(parts) >= 4 else package_name
    if package_name.startswith("homepage."):
        return ".".join(parts[:2]) if len(parts) >= 2 else package_name
    return ".".join(parts[:3]) if len(parts) >= 3 else package_name


def analyze_java(root: Path) -> Dict[str, Any]:
    java_files = [p for p in iter_files(root, suffixes={".java"})]
    file_records: List[Dict[str, Any]] = []
    package_counter: Counter[str] = Counter()
    annotation_counter: Counter[str] = Counter()
    layer_counter: Counter[str] = Counter()
    package_set: Set[str] = set()

    for p in java_files:
        text = read_text(p)
        package_match = JAVA_PACKAGE_RE.search(text)
        package_name = package_match.group(1) if package_match else ""
        package_set.add(package_name)
        package_counter[package_name] += 1
        imports = JAVA_IMPORT_RE.findall(text)
        annotations = set(JAVA_ANNOT_RE.findall(text))
        for a in annotations:
            annotation_counter[a] += 1
        types = JAVA_TYPE_RE.findall(text)
        type_names = [name for _, name in types]
        class_name = type_names[0] if type_names else p.stem
        layer = infer_layer(package_name, class_name, annotations, p)
        layer_counter[layer] += 1
        file_records.append(
            {
                "path": rel(p, root),
                "package": package_name,
                "imports": imports,
                "types": type_names,
                "annotations": sorted(annotations),
                "layer": layer,
            }
        )

    internal_prefixes = set()
    for pkg in package_set:
        if not pkg:
            continue
        internal_prefixes.add(pkg.split(".")[0])

    package_edges: Counter[Tuple[str, str]] = Counter()
    module_edges: Counter[Tuple[str, str]] = Counter()
    for rec in file_records:
        src_pkg = rec["package"]
        src_module = package_to_module(src_pkg)
        for imp in rec["imports"]:
            imp_root = imp.split(".")[0]
            if imp_root not in internal_prefixes:
                continue
            target_pkg = imp[:-2] if imp.endswith(".*") else ".".join(imp.split(".")[:-1])
            if not target_pkg or target_pkg == src_pkg:
                continue
            package_edges[(src_pkg, target_pkg)] += 1
            dst_module = package_to_module(target_pkg)
            if src_module != dst_module:
                module_edges[(src_module, dst_module)] += 1

    top_packages = [{"package": k, "fileCount": v} for k, v in package_counter.most_common(40)]
    top_annotations = [{"annotation": k, "count": v} for k, v in annotation_counter.most_common(40)]

    top_package_edges = [
        {"from": a, "to": b, "weight": w}
        for (a, b), w in package_edges.most_common(200)
    ]
    top_module_edges = [
        {"from": a, "to": b, "weight": w}
        for (a, b), w in module_edges.most_common(200)
    ]

    return {
        "fileCount": len(java_files),
        "packageCount": len([p for p in package_set if p]),
        "topPackages": top_packages,
        "topAnnotations": top_annotations,
        "layerSummary": dict(layer_counter),
        "packageDependencyGraph": {
            "nodeCount": len([p for p in package_set if p]),
            "edgeCount": len(package_edges),
            "topEdges": top_package_edges,
        },
        "moduleDependencyGraph": {
            "nodeCount": len({package_to_module(p) for p in package_set if p}),
            "edgeCount": len(module_edges),
            "topEdges": top_module_edges,
        },
    }


def summarize_files(root: Path) -> Dict[str, Any]:
    counts = Counter()
    for p in iter_files(root):
        ext = p.suffix.lower() or "<noext>"
        counts[ext] += 1
    return {
        "totalFiles": sum(counts.values()),
        "topExtensions": [{"ext": k, "count": v} for k, v in counts.most_common(40)],
    }


def parse_mybatis_mapper_xml(path: Path) -> Dict[str, Any]:
    info = {"path": str(path), "namespace": None, "statementCounts": {}}
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        if local_name(root.tag) != "mapper":
            return info
        info["namespace"] = root.attrib.get("namespace")
        cnt = Counter()
        for elem in root.iter():
            tag = local_name(elem.tag)
            if tag in ("select", "insert", "update", "delete"):
                cnt[tag] += 1
        info["statementCounts"] = dict(cnt)
    except Exception:
        pass
    return info


def analyze_mybatis(root: Path) -> Dict[str, Any]:
    mapper_java = [
        p for p in iter_files(root, suffixes={".java"})
        if "/database/mapper/" in str(p).replace("\\", "/") and p.name.endswith("Mapper.java")
    ]
    mapper_xml = [
        p for p in iter_files(root, suffixes={".xml"})
        if "/database/mapper/" in str(p).replace("\\", "/")
    ]
    legacy_sqlmap_xml = [
        p for p in iter_files(root, suffixes={".xml"})
        if "/sqlmap/mapping/" in str(p).replace("\\", "/")
    ]

    details = []
    namespace_counter = Counter()
    stmt_counter = Counter()
    for p in mapper_xml:
        info = parse_mybatis_mapper_xml(p)
        if info["namespace"]:
            namespace_counter[info["namespace"]] += 1
        for k, v in info["statementCounts"].items():
            stmt_counter[k] += v
        details.append(
            {
                "path": rel(p, root),
                "namespace": info["namespace"],
                "statementCounts": info["statementCounts"],
            }
        )

    return {
        "mapperJavaCount": len(mapper_java),
        "mapperXmlCount": len(mapper_xml),
        "legacySqlMapXmlCount": len(legacy_sqlmap_xml),
        "statementSummary": dict(stmt_counter),
        "namespaces": [k for k, _ in namespace_counter.most_common(200)],
        "mapperXmlDetails": details[:300],
    }


def parse_spring_web_xml(web_xml: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"exists": web_xml.exists(), "servletMappings": [], "filters": [], "listeners": []}
    if not web_xml.exists():
        return info
    try:
        tree = ET.parse(web_xml)
        root = tree.getroot()
        for elem in root:
            tag = local_name(elem.tag)
            if tag == "servlet-mapping":
                item = {}
                for c in elem:
                    ctag = local_name(c.tag)
                    if ctag in ("servlet-name", "url-pattern"):
                        item[ctag] = (c.text or "").strip()
                if item:
                    info["servletMappings"].append(item)
            elif tag == "filter":
                item = {}
                for c in elem:
                    ctag = local_name(c.tag)
                    if ctag in ("filter-name", "filter-class"):
                        item[ctag] = (c.text or "").strip()
                if item:
                    info["filters"].append(item)
            elif tag == "listener":
                for c in elem:
                    if local_name(c.tag) == "listener-class":
                        info["listeners"].append((c.text or "").strip())
    except Exception as exc:
        info["error"] = str(exc)
    return info


def parse_spring_context_files(root: Path) -> Dict[str, Any]:
    ctx_files = sorted(
        p
        for p in iter_files(root, suffixes={".xml"})
        if "/web-inf/config/common/" in str(p).lower().replace("\\", "/")
    )
    imports: Dict[str, List[str]] = {}
    bean_counts: Dict[str, int] = {}
    for f in ctx_files:
        try:
            tree = ET.parse(f)
            r = tree.getroot()
            bean_count = 0
            imp: List[str] = []
            for e in r.iter():
                tag = local_name(e.tag)
                if tag == "bean":
                    bean_count += 1
                elif tag == "import":
                    res = e.attrib.get("resource")
                    if res:
                        imp.append(res)
            imports[rel(f, root)] = imp
            bean_counts[rel(f, root)] = bean_count
        except Exception:
            imports[rel(f, root)] = []
            bean_counts[rel(f, root)] = 0

    return {
        "contextFileCount": len(ctx_files),
        "importsByFile": imports,
        "beanCountByFile": bean_counts,
    }


RISK_PATTERNS = [
    re.compile(r"(password|passwd|secret|api[_.-]?key|clientSecret)\s*[=:]\s*.+", re.IGNORECASE),
    re.compile(r"<entry\s+key=\"[^\"]*(password|passwd|secret)[^\"]*\">.+</entry>", re.IGNORECASE),
]


def scan_risks(root: Path) -> Dict[str, Any]:
    findings = []
    candidates = [
        p for p in iter_files(root, suffixes={".properties", ".xml", ".yml", ".yaml"})
        if "target/" not in str(p).replace("\\", "/")
    ]
    for p in candidates:
        text = read_text(p)
        for i, line in enumerate(text.splitlines(), 1):
            for pattern in RISK_PATTERNS:
                if pattern.search(line):
                    findings.append(
                        {
                            "path": rel(p, root),
                            "line": i,
                            "hint": "possible secret/property credential",
                        }
                    )
                    break
    return {
        "findingCount": len(findings),
        "findings": findings[:200],
    }


def project_structure(root: Path) -> Dict[str, Any]:
    result = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if entry.name in SKIP_DIRS:
            continue
        if entry.is_dir():
            file_count = 0
            for _ in iter_files(entry):
                file_count += 1
            result.append({"name": entry.name, "type": "dir", "fileCount": file_count})
        else:
            result.append({"name": entry.name, "type": "file", "size": entry.stat().st_size})
    return {"topLevel": result}


class LspClient:
    def __init__(self, command: List[str], cwd: Path) -> None:
        self.command = command
        self.cwd = cwd
        self.proc: Optional[subprocess.Popen[bytes]] = None
        self.recv_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.stop_event = threading.Event()
        self.reader_thread: Optional[threading.Thread] = None
        self.next_id = 1

    def start(self) -> None:
        self.proc = subprocess.Popen(
            self.command,
            cwd=str(self.cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()

    def _reader_loop(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        out = self.proc.stdout
        while not self.stop_event.is_set():
            try:
                headers = {}
                line = out.readline()
                if not line:
                    break
                while line and line.strip():
                    raw = line.decode("utf-8", errors="replace").strip()
                    if ":" in raw:
                        k, v = raw.split(":", 1)
                        headers[k.strip().lower()] = v.strip()
                    line = out.readline()
                length = int(headers.get("content-length", "0"))
                if length <= 0:
                    continue
                payload = out.read(length)
                if not payload:
                    break
                msg = json.loads(payload.decode("utf-8", errors="replace"))
                self.recv_queue.put(msg)
            except Exception:
                break

    def _send(self, msg: Dict[str, Any]) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("LSP process not started")
        body = json.dumps(msg, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        try:
            self.proc.stdin.write(header)
            self.proc.stdin.write(body)
            self.proc.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError("LSP stdin is closed (broken pipe)") from exc

    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        msg = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        self._send(msg)

    def request(self, method: str, params: Optional[Dict[str, Any]] = None, timeout: float = 10.0) -> Dict[str, Any]:
        req_id = self.next_id
        self.next_id += 1
        self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}})

        deadline = time.time() + timeout
        buffered = []
        while time.time() < deadline:
            remaining = max(0.05, deadline - time.time())
            try:
                msg = self.recv_queue.get(timeout=remaining)
            except queue.Empty:
                continue
            if msg.get("id") == req_id:
                return msg
            buffered.append(msg)
        for m in buffered:
            self.recv_queue.put(m)
        raise TimeoutError(f"LSP request timeout: {method}")

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            self.request("shutdown", timeout=5.0)
        except Exception:
            pass
        try:
            self.notify("exit")
        except Exception:
            pass
        self.stop_event.set()
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=2.0)
        except Exception:
            pass


def run_lsp_analysis(root: Path, include_lsp: bool, lsp_command: Optional[str], sample_size: int) -> Dict[str, Any]:
    if not include_lsp:
        return {"status": "skipped", "reason": "--include-lsp not enabled"}

    cmd: List[str]
    tmp_data_dir = None
    if lsp_command:
        cmd = shlex.split(lsp_command)
    else:
        jdtls = shutil.which("jdtls")
        if not jdtls:
            return {"status": "unavailable", "reason": "jdtls not found in PATH"}
        tmp_data_dir = tempfile.mkdtemp(prefix="legacy-analyzer-jdtls-")
        cmd = [jdtls, "-data", tmp_data_dir]

    result: Dict[str, Any] = {
        "status": "started",
        "command": cmd,
        "symbolQueries": {},
        "errors": [],
    }
    client = LspClient(cmd, cwd=root)

    try:
        client.start()
        init_params = {
            "processId": os.getpid(),
            "rootUri": root.resolve().as_uri(),
            "capabilities": {},
            "workspaceFolders": [{"uri": root.resolve().as_uri(), "name": root.name}],
        }
        init_resp = client.request("initialize", init_params, timeout=30.0)
        if "error" in init_resp:
            result["status"] = "error"
            result["errors"].append({"stage": "initialize", "error": init_resp["error"]})
            return result

        client.notify("initialized", {})
        for q in ("Controller", "Service", "Mapper", "DAO"):
            try:
                resp = client.request("workspace/symbol", {"query": q}, timeout=20.0)
                items = resp.get("result") or []
                result["symbolQueries"][q] = {
                    "count": len(items),
                    "sample": items[:sample_size],
                }
            except Exception as exc:
                result["errors"].append({"stage": f"workspace/symbol:{q}", "error": str(exc)})

        result["status"] = "ok" if not result["errors"] else "partial"
        return result
    except Exception as exc:
        result["status"] = "error"
        result["errors"].append(
            {
                "stage": "runtime",
                "error": str(exc),
                "hint": "Try --lsp-command with explicit jdtls startup args for your environment.",
            }
        )
        return result
    finally:
        client.close()
        if tmp_data_dir:
            try:
                shutil.rmtree(tmp_data_dir, ignore_errors=True)
            except Exception:
                pass


def make_ai_context(summary: Dict[str, Any]) -> Dict[str, Any]:
    pom = summary.get("build", {}).get("maven", {})
    java = summary.get("java", {})
    mybatis = summary.get("mybatis", {})
    risks = summary.get("risks", {})
    spring = summary.get("spring", {})

    key_points = [
        f"java_files={java.get('fileCount', 0)}",
        f"maven_deps={pom.get('stats', {}).get('dependency_count', 0)}",
        f"system_scope_deps={pom.get('stats', {}).get('system_scope_dependency_count', 0)}",
        f"mapper_xml={mybatis.get('mapperXmlCount', 0)}",
        f"legacy_sqlmap_xml={mybatis.get('legacySqlMapXmlCount', 0)}",
        f"risk_findings={risks.get('findingCount', 0)}",
        f"spring_context_files={spring.get('context', {}).get('contextFileCount', 0)}",
    ]

    migration_signals = {
        "hasLegacySqlMap": (mybatis.get("legacySqlMapXmlCount", 0) > 0),
        "hasSystemScopeDependencies": (pom.get("stats", {}).get("system_scope_dependency_count", 0) > 0),
        "hasHeavyWebXmlConfig": (len(spring.get("webXml", {}).get("filters", [])) > 0),
    }

    return {
        "keyPoints": key_points,
        "migrationSignals": migration_signals,
        "recommendedNextActions": [
            "Split mandatory vs optional runtime dependencies.",
            "Migrate remaining sqlMap resources to Mapper interface + Mapper XML.",
            "Externalize secrets from XML/properties into environment or secret manager.",
            "Use module dependency graph to define migration batches.",
        ],
    }


def analyze(root: Path, include_lsp: bool, lsp_command: Optional[str], sample_size: int) -> Dict[str, Any]:
    pom = parse_pom(root / "pom.xml")
    spring_web = parse_spring_web_xml(root / "src" / "main" / "webapp" / "WEB-INF" / "web.xml")
    spring_ctx = parse_spring_context_files(root)

    result: Dict[str, Any] = {
        "metadata": {
            "tool": "legacy-analyzer",
            "version": "0.1.0",
            "generatedAtUtc": utc_now_iso(),
            "rootPath": str(root.resolve()),
        },
        "project": {
            "name": root.name,
            "structure": project_structure(root),
            "files": summarize_files(root),
        },
        "build": {"maven": pom},
        "java": analyze_java(root),
        "mybatis": analyze_mybatis(root),
        "spring": {"webXml": spring_web, "context": spring_ctx},
        "risks": scan_risks(root),
    }
    result["lsp"] = run_lsp_analysis(root, include_lsp, lsp_command, sample_size)
    result["aiContext"] = make_ai_context(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Legacy Java project analyzer for AI context JSON.")
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    parser.add_argument("--include-lsp", action="store_true", help="Enable LSP-based symbol analysis")
    parser.add_argument(
        "--lsp-command",
        default=None,
        help='LSP server command, e.g. \'jdtls -data .jdtls-workspace\'',
    )
    parser.add_argument("--sample-size", type=int, default=20, help="Max LSP sample items per symbol query")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    data = analyze(root, args.include_lsp, args.lsp_command, args.sample_size)
    with output.open("w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
