#!/usr/bin/env python3
"""Prepare the pilot, or every cataloged dictionary with prepared scans, locally."""

from collections import Counter
from contextlib import nullcontext
from functools import lru_cache
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import unicodedata

from PIL import Image, __version__ as pillow_version


ROOT = Path(__file__).resolve().parent
SCANS = ROOT.parent / "LigurianScans"
PILOT = {
    "frisoni-1910": ("frisoni", [
        ("genovese_italiano", "guardia", 1),
        ("genovese_italiano", "Guardia", 1),
        ("genovese_italiano", "abboccamento", 1),
        ("italiano_genovese", "abboccamento", 1),
    ]),
    "casaccia-1876": ("casaccia", [
        ("main_dictionary", "abbiscâ", 1),
        ("main_dictionary", "vivûssâ", 1),
        ("main_dictionary", "abdicazion", 1),
    ]),
    "paganini-1857": ("paganini", [
        ("main_dictionary", "cuxinha", 1),
        ("main_dictionary", "fornelli", 1),
        ("main_dictionary", "banderolla", 1),
    ]),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepared_books(catalog, scans, all_books):
    require(catalog.get("format") == "liguriankb.dictionary-jsonl" and catalog.get("schema_version") == 1,
            "Unsupported dictionary export catalog")
    registrations = {}
    for entry in catalog["dictionaries"]:
        document = entry["document_id"]
        require(document not in registrations, "Duplicate catalog document: " + document)
        require(document and document not in (".", "..") and not any(c in document for c in "/\\\0"),
                "Invalid document id: " + document)
        require(Path(entry["file"]).name == entry["file"] and entry["file"].endswith(".jsonl")
                and not any(c in entry["file"] for c in "\\\0"), "Invalid export filename")
        registrations[document] = entry
    wanted = set(scans) if all_books else set(PILOT)
    require(wanted <= set(registrations), "Prepared scans or pilot have no export registration")
    return {document: entry for document, entry in registrations.items() if document in wanted}


def atomic_json(path, value):
    staged = path.with_suffix(path.suffix + ".tmp")
    staged.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    staged.replace(path)


def extra_label(record):
    if record.get("illustration_tables"):
        return "tavola-" + str(min(record["illustration_tables"]))
    return Path(record["pdf"]).stem + "-" + str(record["pdf_page"])


def entry_key(entry, source):
    return (entry["section"], entry["content"][source]["lemmas"][0], entry.get("lemma_index", 1))


def entry_ids(entries, source):
    names = [unicodedata.normalize("NFC", entry_key(e, source)[1].lower()) + "." + str(e.get("lemma_index", 1)) for e in entries]
    counts = Counter((e["section"], name.casefold()) for e, name in zip(entries, names))
    used = set()
    for i, (entry, name) in enumerate(zip(entries, names)):
        lemma = entry_key(entry, source)[1]
        if counts[entry["section"], name.casefold()] > 1 and lemma != lemma.lower():
            name += "-"
        for component in (entry["section"], name):
            assert component not in ("", ".", "..") and not any(c in component for c in "/\\\0")
            assert not any(ord(c) < 32 for c in component) and len(component.encode("utf-8")) <= 255
        key = (entry["section"].casefold(), name.casefold())
        assert key not in used, (source, key)
        used.add(key)
        names[i] = name
    return names


def pixel_box(rectangle, frame, dimensions):
    assert len(rectangle) == len(frame) == 4
    assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in rectangle + frame)
    x0, y0, x1, y1 = rectangle
    assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
    left, top, right, bottom = frame
    assert left < right and top < bottom
    box = (math.floor(left + x0 * (right - left)), math.floor(top + y0 * (bottom - top)),
           math.ceil(left + x1 * (right - left)), math.ceil(top + y1 * (bottom - top)))
    assert 0 <= box[0] < box[2] <= dimensions[0] and 0 <= box[1] < box[3] <= dimensions[1], box
    return box


def locations_in(value, pointer=""):
    if isinstance(value, dict):
        if "image_coordinates" in value:
            yield pointer, value
        for key, child in value.items():
            yield from locations_in(child, pointer + "/" + key.replace("~", "~0").replace("/", "~1"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from locations_in(child, pointer + "/" + str(index))


def self_check():
    examples = [{"section": "main", "content": {"example": {"lemmas": [word]}}} for word in ("guardia", "Guardia", "Ä")]
    assert entry_ids(examples, "example") == ["guardia.1", "guardia.1-", "ä.1"]
    assert pixel_box([.25, .25, .75, .75], [2, 3, 6, 7], (10, 10)) == (3, 4, 5, 6)
    assert pixel_box([0, 0, 1, 1], [.2, .3, 5.2, 6.3], (10, 10)) == (0, 0, 6, 7)
    assert list(locations_in({"errata_source_locations": {"0": {"image_coordinates": []}}}))[0][0] == "/errata_source_locations/0"
    catalog = {"format": "liguriankb.dictionary-jsonl", "schema_version": 1, "dictionaries": [
        {"document_id": "pfb-1873", "content_key": "pfb", "file": "pfb.jsonl"},
        {"document_id": "vpl-ac", "content_key": "vpl", "file": "vpl-ac.jsonl"}]}
    assert list(prepared_books(catalog, {"pfb-1873": {}}, True)) == ["pfb-1873"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Prepare every cataloged entry and retained page with prepared scans")
    args = parser.parse_args()
    self_check()
    inputs = {}
    preparation = {"complete": False, "dpi": 300, "inputs": inputs, "books": {}}
    staged_mappings = []
    export_folder = ROOT / "exports"
    def read_input(relative):
        raw = (ROOT.parent / relative).read_bytes()
        inputs[relative] = hashlib.sha256(raw).hexdigest()
        return raw

    manifest = json.loads(read_input("LigurianScans/manifest.json"))
    assert manifest["complete"] is True and manifest["dpi"] == 300
    export_catalog = json.loads(read_input("LigurianKB/exports/manifest.json"))
    books = prepared_books(export_catalog, manifest["books"], args.all)
    catalog = json.loads(read_input("LigurianKB/processing/paganini/illustrations.json"))
    assets = json.loads(read_input("LigurianKB/processing/paganini/scan-assets.json"))
    assert catalog["complete"] is True and assets["complete"] is True and assets["dpi"] == 300
    assert catalog["scan_assets_sha256"] == inputs["LigurianKB/processing/paganini/scan-assets.json"]
    tables = {table["table"]: table for table in catalog["tables"]}
    if args.all:
        export_folder.mkdir(exist_ok=True)
        atomic_json(export_folder / "manifest.json", preparation)
    pages = {}
    crop_paths = set()
    output = {"dpi": 300, "pillow": pillow_version, "inputs": inputs, "pages": pages, "entries": []}

    @lru_cache(maxsize=3)
    def raster(sha, width, height):
        source_path = SCANS / "public/pages" / sha[:2] / (sha + ".png")
        assert digest(source_path) == sha
        with Image.open(source_path) as opened:
            opened.load()
            assert opened.size == (width, height) and opened.mode in ("1", "L", "RGB")
            return opened.copy()

    def page_image(document, record, label, load=True):
        sha = record["sha256"]
        source_path = SCANS / "public/pages" / sha[:2] / (sha + ".png")
        require(digest(source_path) == sha, "Prepared source PNG hash differs: " + sha)
        relative = Path(document) / "pagine" / (str(label) + ".png")
        destination = ROOT / relative
        if relative.as_posix() not in pages:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_file() or digest(destination) != sha:
                shutil.copyfile(source_path, destination)
            assert digest(destination) == sha
            pages[relative.as_posix()] = {"sha256": sha, "pdf": record.get("pdf", "paganini-1857.pdf"),
                                         "pdf_page": record["pdf_page"]}
        else:
            assert pages[relative.as_posix()]["sha256"] == sha
        image = raster(sha, record["width"], record["height"]) if load else None
        return image, relative.as_posix()

    for document, registration in books.items():
        source = registration["content_key"]
        selectors = PILOT.get(document, (None, []))[1]
        export_path = "LigurianKB/exports/" + registration["file"]
        export_bytes = read_input(export_path)
        require(inputs[export_path] == registration["sha256"], "Export catalog SHA-256 differs: " + export_path)
        if source == "paganini":
            require(catalog["public_dictionary_sha256"] == inputs[export_path], "Paganini public catalog/export snapshot differs")
        entries = [json.loads(line) for line in export_bytes.splitlines() if line.strip()]
        require(len(entries) == registration["entries"], "Export catalog entry count differs: " + export_path)
        require(all(set(entry["content"]) == {source} for entry in entries), "Export content key differs: " + export_path)
        identifiers = entry_ids(entries, source)
        by_key = {entry_key(entry, source): (entry, identifier) for entry, identifier in zip(entries, identifiers)}
        assert len(by_key) == len(entries)
        if not args.all:
            assert all(selector in by_key for selector in selectors)
        selected = list(zip(entries, identifiers)) if args.all else [by_key[selector] for selector in selectors]
        staged = export_folder / (document + ".images.jsonl.tmp")
        page_count_before, crop_count_before = len(pages), len(crop_paths)
        print(f"{document}: preparing {len(selected):,} entries", flush=True)
        with (staged.open("w", encoding="utf-8") if args.all else nullcontext(None)) as mapping:
            for entry, identifier in selected:
                content = entry["content"][source]
                folder = Path(document) / entry["section"] / identifier
                folder_path = ROOT / folder
                folder_path.mkdir(parents=True, exist_ok=True)
                result = {"document": document, "section": entry["section"], "entry_id": identifier,
                          "headwords": content["lemmas"], "lemma_index": entry.get("lemma_index", 1),
                          "locations": {}, "images": []}
                if "section" in content:
                    result["thematic_section"] = content["section"]
                seen = {}

                def save_crop(image, record, page_path, box):
                    key = (record["sha256"], tuple(box))
                    if key in seen:
                        return seen[key]
                    index = len(result["images"]) + 1
                    relative = folder / (str(index) + ".png")
                    destination = ROOT / relative
                    crop = image.crop(box)
                    valid = False
                    if destination.is_file():
                        try:
                            with Image.open(destination) as saved:
                                valid = (saved.format == "PNG" and saved.mode == crop.mode and saved.size == crop.size
                                         and saved.tobytes() == crop.tobytes()
                                         and all(abs(dpi - 300) < .01 for dpi in saved.info.get("dpi", (0, 0))))
                        except (OSError, ValueError):
                            pass
                    if not valid:
                        crop.save(destination, optimize=True, dpi=(300, 300))
                    with Image.open(destination) as saved:
                        assert saved.mode == image.mode and saved.size == (box[2] - box[0], box[3] - box[1])
                        assert saved.tobytes() == crop.tobytes()
                        assert all(abs(dpi - 300) < .01 for dpi in saved.info["dpi"])
                    crop_paths.add(relative.as_posix())
                    result["images"].append({"path": relative.as_posix(), "page_image": page_path,
                                             "box": list(box), "sha256": digest(destination)})
                    seen[key] = index
                    return index

                location_records = [("/source_locations", entry["source_locations"])]
                location_records.extend((pointer, loc) for pointer, loc in locations_in(entry) if pointer != "/source_locations")
                for pointer, locations in location_records:
                    mapped = {key: value for key, value in locations.items() if key != "image_coordinates"}
                    mapped["images"] = []
                    for region in locations["image_coordinates"]:
                        record = manifest["books"][document]["pages"][str(region["page"])]
                        image, page_path = page_image(document, record, region["page"])
                        index = save_crop(image, record, page_path, pixel_box(region["rectangle"], record["frame"], image.size))
                        mapped["images"].append({"index": index, **{key: value for key, value in region.items() if key != "rectangle"}})
                    result["locations"][pointer] = mapped
                for reference_index, reference in enumerate(content.get("illustrations", [])):
                    bindings = [b for b in catalog["entry_bindings"][entry["id"]]
                                if all(b.get(key) == reference.get(key) for key in ("table", "number", "part"))]
                    assert len(bindings) == 1
                    table = tables[reference["table"]]
                    record = assets["pdf_pages"][str(table["pdf_page"])].copy()
                    record["pdf"] = "paganini-1857.pdf"
                    assert record["sha256"] == table["image_sha256"]
                    image, page_path = page_image(document, record, extra_label(record) if args.all else "tavola-" + str(table["table"]))
                    assert image.mode == table["image_mode"]
                    mapped = {**reference, "images": []}
                    for figure in table["figures"]:
                        for obj in figure["objects"]:
                            if obj["object_id"] not in bindings[0]["object_ids"]:
                                continue
                            for region in obj["regions"]:
                                box = region["delivery_rectangle"]
                                assert all(isinstance(v, int) for v in box)
                                assert 0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height
                                index = save_crop(image, record, page_path, box)
                                mapped["images"].append({"index": index, "table": table["table"], "number": figure["number"]})
                    assert mapped["images"]
                    result["locations"][f"/content/{source}/illustrations/{reference_index}"] = mapped
                result["image_count"] = len(result["images"])
                assert [r["index"] for r in result["locations"]["/source_locations"]["images"]] == list(range(1, len(entry["source_locations"]["image_coordinates"]) + 1))
                if args.all:
                    result["source_id"] = entry["id"]
                    mapping.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                else:
                    output["entries"].append(result)
        if args.all:
            for page, record in manifest["books"][document]["pages"].items():
                page_image(document, record, page, load=False)
            for record in manifest["books"][document]["extras"]:
                page_image(document, record, extra_label(record), load=False)
            preparation["books"][document] = {"source": source, "entries": len(selected),
                                              "pages": len(pages) - page_count_before,
                                              "crops": len(crop_paths) - crop_count_before,
                                              "mapping_sha256": digest(staged)}
            staged_mappings.append((staged, export_folder / (document + ".images.jsonl")))
            print(f"{document}: verified {len(selected):,} entries, {len(crop_paths) - crop_count_before:,} crops and {len(pages) - page_count_before:,} unchanged pages", flush=True)
        raster.cache_clear()
    paths = list(pages) + sorted(crop_paths)
    assert len(paths) == len(set(unicodedata.normalize("NFC", path).casefold() for path in paths))
    actual_paths = {p.relative_to(ROOT).as_posix() for document in books for p in (ROOT / document).rglob("*.png")}
    assert set(paths) <= actual_paths
    if args.all:
        for relative in actual_paths - set(paths):
            (ROOT / relative).unlink()
        for staged, destination in staged_mappings:
            staged.replace(destination)
        preparation["complete"] = True
        atomic_json(export_folder / "manifest.json", preparation)
        total_entries = sum(book["entries"] for book in preparation["books"].values())
    else:
        atomic_json(ROOT / "pilot.json", output)
        total_entries = len(output["entries"])
    print(f"Verified {total_entries:,} entries, {len(crop_paths):,} crops, {len(pages):,} unchanged full pages, {sum((ROOT / p).stat().st_size for p in paths):,} PNG bytes.", flush=True)


if __name__ == "__main__":
    main()
