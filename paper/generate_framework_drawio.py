from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import uuid
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "systematic_framework.drawio"

CANVAS_W = 1408
CANVAS_H = 768


class DrawioBuilder:
    def __init__(self) -> None:
        modified = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        self.mxfile = ET.Element(
            "mxfile",
            {
                "host": "app.diagrams.net",
                "modified": modified,
                "agent": "Codex",
                "version": "24.7.17",
                "type": "device",
                "compressed": "false",
            },
        )
        self.diagram = ET.SubElement(
            self.mxfile,
            "diagram",
            {
                "id": uuid.uuid4().hex[:12],
                "name": "Page-1",
            },
        )
        self.model = ET.SubElement(
            self.diagram,
            "mxGraphModel",
            {
                "dx": "1662",
                "dy": "910",
                "grid": "1",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": str(CANVAS_W),
                "pageHeight": str(CANVAS_H),
                "math": "0",
                "shadow": "0",
            },
        )
        self.root = ET.SubElement(self.model, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})
        self.next_id = 2

    def _id(self) -> str:
        cell_id = str(self.next_id)
        self.next_id += 1
        return cell_id

    def vertex(
        self,
        value: str,
        x: float,
        y: float,
        w: float,
        h: float,
        style: str,
        parent: str = "1",
    ) -> str:
        cell_id = self._id()
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": cell_id,
                "value": value,
                "style": style,
                "vertex": "1",
                "parent": parent,
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": f"{x:.1f}",
                "y": f"{y:.1f}",
                "width": f"{w:.1f}",
                "height": f"{h:.1f}",
                "as": "geometry",
            },
        )
        return cell_id

    def edge(
        self,
        style: str,
        source: str | None = None,
        target: str | None = None,
        points: list[tuple[float, float]] | None = None,
        source_point: tuple[float, float] | None = None,
        target_point: tuple[float, float] | None = None,
        value: str = "",
        parent: str = "1",
    ) -> str:
        cell_id = self._id()
        attrs = {
            "id": cell_id,
            "value": value,
            "style": style,
            "edge": "1",
            "parent": parent,
        }
        if source:
            attrs["source"] = source
        if target:
            attrs["target"] = target
        cell = ET.SubElement(self.root, "mxCell", attrs)
        geom = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        if source_point:
            ET.SubElement(geom, "mxPoint", {"x": f"{source_point[0]:.1f}", "y": f"{source_point[1]:.1f}", "as": "sourcePoint"})
        if target_point:
            ET.SubElement(geom, "mxPoint", {"x": f"{target_point[0]:.1f}", "y": f"{target_point[1]:.1f}", "as": "targetPoint"})
        if points:
            arr = ET.SubElement(geom, "Array", {"as": "points"})
            for x, y in points:
                ET.SubElement(arr, "mxPoint", {"x": f"{x:.1f}", "y": f"{y:.1f}"})
        return cell_id

    def save(self, path: Path) -> None:
        ET.indent(self.mxfile, space="  ")
        ET.ElementTree(self.mxfile).write(path, encoding="utf-8", xml_declaration=True)


def text_style(font_size: int, bold: bool = False, align: str = "center", color: str = "#111111") -> str:
    style = (
        "text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;overflow=hidden;"
        f"fontFamily=Helvetica;fontSize={font_size};fontColor={color};align={align};verticalAlign=middle;"
    )
    if bold:
        style += "fontStyle=1;"
    return style


def box_style(fill: str, stroke: str, rounded: bool = True, extra: str = "") -> str:
    rounded_flag = "1" if rounded else "0"
    return (
        f"rounded={rounded_flag};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
        "strokeWidth=2;arcSize=16;"
        + extra
    )


def icon_box(fill: str = "#FFFFFF", stroke: str = "#222222", rounded: bool = True, extra: str = "") -> str:
    return box_style(fill, stroke, rounded=rounded, extra="shadow=0;" + extra)


def ellipse_style(fill: str, stroke: str, extra: str = "") -> str:
    return (
        f"ellipse;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=2;"
        + extra
    )


def edge_style(color: str = "#222222", arrow: str = "block", width: int = 2, extra: str = "") -> str:
    return (
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
        f"html=1;strokeColor={color};strokeWidth={width};endArrow={arrow};endFill=1;"
        + extra
    )


def curve_style(color: str, width: int = 3) -> str:
    return f"curved=1;html=1;strokeColor={color};strokeWidth={width};endArrow=none;rounded=0;"


def panel(builder: DrawioBuilder, x: float, y: float, w: float, h: float, fill: str, header_fill: str, stroke: str, title: str) -> None:
    builder.vertex("", x, y, w, h, box_style(fill, stroke, extra="shadow=0;"))
    builder.vertex(
        title,
        x,
        y,
        w,
        72,
        box_style(header_fill, "none", extra="fontFamily=Helvetica;fontSize=16;fontStyle=1;align=center;verticalAlign=middle;"),
    )


def add_graph(builder: DrawioBuilder, nodes: list[tuple[float, float, str]], edges: list[tuple[int, int]], radius: float = 7) -> None:
    for i, j in edges:
        x1, y1, _ = nodes[i]
        x2, y2, _ = nodes[j]
        builder.edge(
            "html=1;strokeColor=#333333;strokeWidth=2;endArrow=none;rounded=0;",
            source_point=(x1, y1),
            target_point=(x2, y2),
        )
    for x, y, color in nodes:
        builder.vertex("", x - radius, y - radius, radius * 2, radius * 2, ellipse_style(color, "#333333"))


def add_curve(builder: DrawioBuilder, points: list[tuple[float, float]], color: str, width: int = 3) -> None:
    builder.edge(
        curve_style(color, width=width),
        source_point=points[0],
        target_point=points[-1],
        points=points[1:-1],
    )


def add_stack(builder: DrawioBuilder, x: float, y: float) -> None:
    layers = ["#CFE9E3", "#73C5C1", "#9CD7E8", "#6DA4D8", "#4B5FBC"]
    for idx, color in enumerate(layers):
        builder.vertex(
            "",
            x,
            y + idx * 16,
            88,
            22,
            "shape=parallelogram;perimeter=parallelogramPerimeter;whiteSpace=wrap;html=1;"
            f"fillColor={color};strokeColor=#1F2833;strokeWidth=2;",
        )


def add_laptop(builder: DrawioBuilder, x: float, y: float) -> None:
    builder.vertex("", x, y, 165, 120, icon_box("#B7BCCB", "#111111"))
    builder.vertex("", x + 9, y + 8, 147, 99, icon_box("#FFFFFF", "#333333", rounded=False))
    builder.vertex("", x - 14, y + 118, 193, 14, icon_box("#C0D8D8", "#111111"))
    builder.edge("html=1;strokeColor=#111111;strokeWidth=2;endArrow=none;", source_point=(x + 45, y + 132), target_point=(x + 120, y + 132))
    builder.vertex("Screenshot App", x + 28, y + 18, 100, 12, text_style(7, bold=True, align="left", color="#555555"))
    for row, color in enumerate(["#1E2430", "#252F45", "#1B202B"]):
        builder.vertex("", x + 18, y + 30 + row * 17, 34, 13, box_style(color, color, rounded=False))
    builder.vertex("", x + 55, y + 30, 83, 60, box_style("#F4F4F4", "#E3E3E3", rounded=False))
    builder.vertex("", x + 55, y + 94, 75, 7, box_style("#EFEFEF", "#EFEFEF", rounded=False))


def add_servers(builder: DrawioBuilder, x: float, y: float) -> None:
    for row in range(3):
        yy = y + row * 26
        builder.vertex("", x, yy, 90, 22, icon_box("#7288A2", "#111111"))
        for col in range(3):
            builder.vertex("", x + 56 + col * 10, yy + 6, 4, 4, ellipse_style("#111111", "#111111", extra="strokeWidth=1;"))
        builder.vertex("", x + 10, yy + 6, 8, 8, ellipse_style("#C7E27A" if row < 2 else "#E88B74", "#111111", extra="strokeWidth=1;"))


def add_document(builder: DrawioBuilder, x: float, y: float, w: float, h: float, fill: str = "#FFFFFF") -> str:
    return builder.vertex(
        "",
        x,
        y,
        w,
        h,
        f"shape=document;whiteSpace=wrap;html=1;boundedLbl=1;fillColor={fill};strokeColor=#222222;strokeWidth=2;",
    )


def add_robot(builder: DrawioBuilder, x: float, y: float) -> None:
    builder.edge("html=1;strokeColor=#111111;strokeWidth=2;endArrow=none;", source_point=(x + 32, y - 6), target_point=(x + 32, y + 6))
    builder.vertex("", x + 28, y - 14, 8, 8, ellipse_style("#FFFFFF", "#111111"))
    builder.vertex("", x, y, 64, 42, icon_box("#7B8FA1", "#111111"))
    builder.vertex("", x + 8, y + 10, 48, 20, icon_box("#95D4EF", "#436179"))
    builder.vertex("", x + 16, y + 15, 8, 8, ellipse_style("#FFFFFF", "#3A5A6E", extra="strokeWidth=1;"))
    builder.vertex("", x + 40, y + 15, 8, 8, ellipse_style("#FFFFFF", "#3A5A6E", extra="strokeWidth=1;"))
    builder.edge("html=1;strokeColor=#3A5A6E;strokeWidth=2;endArrow=none;", source_point=(x + 24, y + 28), target_point=(x + 40, y + 28))
    builder.vertex("", x + 16, y + 42, 32, 34, ellipse_style("#D6EEF8", "#111111"))
    builder.edge("html=1;strokeColor=#111111;strokeWidth=2;endArrow=none;", source_point=(x, y + 18), target_point=(x - 14, y + 6))
    builder.edge("html=1;strokeColor=#111111;strokeWidth=2;endArrow=none;", source_point=(x - 14, y + 6), target_point=(x - 22, y + 14))
    builder.edge("html=1;strokeColor=#111111;strokeWidth=2;endArrow=none;", source_point=(x + 64, y + 18), target_point=(x + 78, y + 6))
    builder.edge("html=1;strokeColor=#111111;strokeWidth=2;endArrow=none;", source_point=(x + 78, y + 6), target_point=(x + 86, y + 14))
    builder.edge("html=1;strokeColor=#111111;strokeWidth=2;endArrow=none;", source_point=(x + 22, y + 76), target_point=(x + 12, y + 98))
    builder.edge("html=1;strokeColor=#111111;strokeWidth=2;endArrow=none;", source_point=(x + 42, y + 76), target_point=(x + 50, y + 98))
    builder.vertex("", x - 28, y + 32, 12, 28, "shape=ext;double=1;whiteSpace=wrap;html=1;fillColor=#D7E2EA;strokeColor=#111111;strokeWidth=2;rotation=-35;")
    builder.vertex("", x + 74, y + 34, 12, 28, "shape=ext;double=1;whiteSpace=wrap;html=1;fillColor=#D7E2EA;strokeColor=#111111;strokeWidth=2;rotation=35;")


def add_magnifier(builder: DrawioBuilder, x: float, y: float) -> None:
    add_graph(
        builder,
        [
            (x + 10, y + 22, "#F1A071"),
            (x + 36, y + 22, "#E9CAA2"),
            (x + 70, y + 22, "#80B889"),
            (x + 70, y + 50, "#7CC0B5"),
        ],
        [(0, 1), (1, 2), (2, 3)],
        radius=6,
    )
    builder.vertex("", x + 72, y - 8, 56, 56, ellipse_style("#D8D7E4", "#111111"))
    builder.vertex("", x + 82, y + 2, 36, 36, ellipse_style("#CDE2F0", "#444444"))
    builder.edge("html=1;strokeColor=#444444;strokeWidth=3;endArrow=none;", source_point=(x + 109, y + 38), target_point=(x + 142, y + 71))
    builder.vertex("", x + 136, y + 65, 8, 8, ellipse_style("#8C684A", "#8C684A", extra="strokeWidth=1;"))
    builder.edge("html=1;strokeColor=#6692B5;strokeWidth=2;endArrow=none;", source_point=(x + 90, y + 21), target_point=(x + 108, y + 21))
    builder.edge("html=1;strokeColor=#6692B5;strokeWidth=2;endArrow=none;", source_point=(x + 90, y + 28), target_point=(x + 101, y + 28))


def main() -> None:
    builder = DrawioBuilder()

    builder.vertex("", 0, 0, CANVAS_W, 70, box_style("#DCEBF3", "none", rounded=False))
    builder.vertex(
        "SYSTEMATIC FRAMEWORK FOR SCIENTIFIC LITERATURE MAPPING & HYPOTHESIS GENERATION",
        24,
        16,
        1360,
        40,
        text_style(26, bold=True),
    )

    panels = [
        (15, 84, 210, 581, "#DCEAF6", "#BBD3E8", "#4F84A8", "1. DATA INGESTION<br>& PROCESSING"),
        (249, 84, 210, 581, "#DDEFEA", "#B9DED3", "#74A78F", "2. INTERACTIVE<br>EXPLORATION"),
        (485, 84, 205, 581, "#F9E9D4", "#F0D1A7", "#D59A4D", "3. NOVELTY<br>DETECTION &<br>GAP ANALYSIS"),
        (717, 84, 205, 581, "#DFF0E3", "#B7D9BF", "#77B488", "4. EVIDENCE<br>PACKAGING &<br>BACKEND"),
        (949, 84, 215, 581, "#EAE0EF", "#D4C1E0", "#9D70B1", "5. AGENT<br>WORKFLOW<br>& GENERATION"),
        (1184, 84, 210, 581, "#F0E6F1", "#DDCAE4", "#9E72AC", "6. RETROSPECTIVE<br>EVALUATION"),
    ]
    for args in panels:
        panel(builder, *args)

    builder.vertex(
        "",
        15,
        684,
        1378,
        72,
        box_style("#DCE8F1", "#2E6F91", extra="shadow=0;"),
    )
    builder.vertex(
        "CORE CAPABILITY: Map a scientific literature space, surface sparse/novel regions, and use evidence-constrained<br>LLM workflows to propose and evaluate research directions",
        70,
        697,
        1275,
        44,
        text_style(20),
    )

    # Panel 1
    add_stack(builder, 76, 168)
    builder.vertex("PubMed<br>Paper Corpus", 44, 266, 150, 62, text_style(18, bold=True))
    prep_box = builder.vertex("Preprocessing", 36, 348, 166, 34, box_style("#D8E7F2", "#5D8DB2", extra="fontStyle=1;fontSize=14;"))
    embed_box = builder.vertex("", 35, 412, 168, 117, box_style("#D4E6F3", "#5D8DB2"))
    builder.vertex("Embedding<br>Generation", 56, 422, 126, 48, text_style(16, bold=True))
    builder.vertex("", 79, 468, 24, 24, box_style("#F0C15D", "#222222", rounded=False))
    builder.vertex("", 126, 464, 28, 28, ellipse_style("#F4F4FB", "#5C6CB1"))
    builder.edge("html=1;strokeColor=#5C6CB1;strokeWidth=2;endArrow=none;", source_point=(132, 478), target_point=(148, 478))
    builder.edge("html=1;strokeColor=#5C6CB1;strokeWidth=2;endArrow=none;", source_point=(140, 470), target_point=(140, 486))
    builder.vertex("BERT/Qwen", 50, 494, 138, 24, text_style(14))
    builder.vertex("Paper<br>Embeddings", 50, 548, 130, 56, text_style(18, bold=True))
    builder.edge(edge_style(), source_point=(120, 328), target_point=(120, 348))
    builder.edge(edge_style(), source_point=(120, 382), target_point=(120, 412))
    builder.edge(edge_style(), source_point=(120, 529), target_point=(120, 548))

    # Panel 2
    add_laptop(builder, 271, 177)
    builder.vertex("Streamlit Application", 261, 320, 190, 26, text_style(16, bold=True))
    for dx, dy, color in [
        (288, 367, "#B98E58"),
        (297, 371, "#7BB7A3"),
        (305, 363, "#CAB97B"),
        (309, 378, "#7FBAA8"),
        (317, 368, "#B9D8B2"),
        (324, 373, "#8FC8AE"),
        (294, 383, "#9F8365"),
    ]:
        builder.vertex("", dx, dy, 5, 5, ellipse_style(color, "#333333", extra="strokeWidth=1;"))
    add_graph(
        builder,
        [
            (390, 367, "#8AB1D9"),
            (405, 353, "#7FC68F"),
            (420, 367, "#E4A35F"),
            (390, 384, "#7DA6D2"),
            (420, 384, "#7BC79F"),
        ],
        [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)],
        radius=6,
    )
    builder.vertex("PCA/UMAP Clustering<br>Views", 257, 396, 118, 56, text_style(13))
    builder.vertex("Semantic<br>Search", 263, 468, 76, 54, text_style(14))
    builder.vertex("Entity<br>Filtering", 372, 468, 76, 54, text_style(14))
    builder.vertex("", 288, 455, 20, 20, ellipse_style("#DDE8ED", "#222222"))
    builder.edge("html=1;strokeColor=#222222;strokeWidth=2;endArrow=none;", source_point=(302, 469), target_point=(316, 483))
    builder.vertex(
        "",
        401,
        453,
        22,
        28,
        "shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;html=1;direction=south;fillColor=#8E9ECB;strokeColor=#222222;strokeWidth=2;",
    )
    add_document(builder, 286, 545, 34, 48, fill="#E5F2E3")
    add_document(builder, 382, 545, 34, 48, fill="#EEF4F8")
    builder.edge("html=1;strokeColor=#2C8C54;strokeWidth=2;endArrow=block;", source_point=(401, 574), target_point=(425, 563))
    builder.vertex("Paper<br>Inspection", 261, 599, 86, 44, text_style(14))
    builder.vertex("Result<br>Export", 369, 599, 80, 44, text_style(14))

    # Panel 3
    add_graph(
        builder,
        [
            (541, 201, "#7FBC9A"),
            (581, 186, "#7AAED3"),
            (616, 208, "#7AAED3"),
            (647, 190, "#E2A06A"),
            (648, 227, "#84C5A7"),
            (620, 259, "#E59A48"),
            (592, 266, "#8AC69B"),
            (548, 253, "#6F9FD4"),
            (558, 223, "#5C8FC8"),
        ],
        [(0, 1), (0, 7), (0, 8), (1, 2), (1, 8), (1, 3), (2, 3), (2, 4), (2, 5), (2, 6), (2, 8), (3, 5), (4, 5), (5, 6), (6, 7), (6, 8), (7, 8)],
        radius=7,
    )
    builder.vertex("k-NN Graph over<br>Embeddings", 512, 278, 150, 50, text_style(16, bold=True))
    builder.edge("html=1;strokeColor=#222222;strokeWidth=1.5;endArrow=none;", source_point=(512, 421), target_point=(660, 421))
    add_curve(builder, [(512, 421), (532, 386), (548, 343), (565, 387), (586, 414), (602, 374), (618, 361), (636, 402), (660, 421)], "#517AA7", 3)
    add_curve(builder, [(514, 421), (535, 413), (549, 405), (564, 395), (581, 413), (602, 409), (621, 382), (638, 388), (659, 421)], "#D5C58D", 2)
    add_curve(builder, [(518, 421), (538, 418), (555, 416), (573, 410), (590, 399), (609, 404), (626, 390), (642, 382), (658, 421)], "#E7A160", 2)
    builder.vertex("Density-Based<br>Gap Scores", 523, 426, 130, 50, text_style(16, bold=True))
    builder.edge(edge_style(), source_point=(588, 478), target_point=(588, 500))
    builder.edge("html=1;strokeColor=#222222;strokeWidth=2;endArrow=block;", source_point=(588, 486), target_point=(545, 501))
    builder.edge("html=1;strokeColor=#222222;strokeWidth=2;endArrow=block;", source_point=(588, 486), target_point=(635, 501))
    add_graph(
        builder,
        [
            (520, 514, "#A4BFCF"),
            (540, 500, "#8AC09A"),
            (562, 514, "#EDA45A"),
            (541, 534, "#DDD39B"),
        ],
        [(0, 1), (1, 2), (1, 3)],
        radius=7,
    )
    add_graph(
        builder,
        [
            (627, 532, "#EC8C62"),
            (655, 514, "#E7BB89"),
            (661, 542, "#D7A96A"),
            (688, 532, "#8CBB99"),
        ],
        [(0, 1), (0, 2), (2, 3)],
        radius=7,
    )
    builder.vertex("Sparse<br>Gap<br>Regions", 516, 548, 70, 72, text_style(13))
    builder.vertex("Cluster-<br>Bridge<br>Targets", 610, 548, 74, 72, text_style(13))

    # Panel 4
    add_servers(builder, 775, 181)
    builder.vertex("SQLite/FastAPI<br>Backend", 744, 278, 150, 48, text_style(17, bold=True))
    builder.edge(edge_style(), source_point=(819, 328), target_point=(819, 392))
    builder.vertex("Frozen<br>Snapshot", 833, 333, 74, 56, text_style(12, align="left"))
    doc_id = add_document(builder, 787, 399, 52, 70, fill="#D5E2F3")
    add_graph(
        builder,
        [
            (816, 430, "#E2A06A"),
            (831, 418, "#A2C1DE"),
            (846, 431, "#8FC298"),
            (816, 450, "#94C2C4"),
            (838, 448, "#D7CB8C"),
        ],
        [(0, 1), (1, 2), (1, 3), (3, 4)],
        radius=4,
    )
    builder.vertex("Structured<br>Evidence Packs", 738, 478, 150, 48, text_style(17, bold=True))
    builder.vertex("surrounding<br>gaps/clusters", 760, 532, 110, 42, text_style(12))

    # Panel 5
    add_robot(builder, 1020, 189)
    builder.vertex("Explain<br>Gap", 959, 282, 88, 52, text_style(15, bold=True))
    builder.edge(
        edge_style(),
        source_point=(839, 438),
        target_point=(1012, 300),
        points=[(935, 438), (935, 300)],
    )
    builder.edge("html=1;strokeColor=#222222;strokeWidth=2;endArrow=none;", source_point=(1031, 300), target_point=(1031, 356))
    builder.edge("html=1;strokeColor=#222222;strokeWidth=2;endArrow=block;", source_point=(1031, 316), target_point=(995, 316))
    builder.edge("html=1;strokeColor=#222222;strokeWidth=2;endArrow=none;", source_point=(988, 332), target_point=(988, 500))
    builder.edge("html=1;strokeColor=#222222;strokeWidth=2;endArrow=block;", source_point=(988, 363), target_point=(1034, 363))
    builder.edge("html=1;strokeColor=#222222;strokeWidth=2;endArrow=block;", source_point=(988, 431), target_point=(1030, 431))
    builder.edge("html=1;strokeColor=#222222;strokeWidth=2;endArrow=block;", source_point=(988, 499), target_point=(1026, 499))
    builder.vertex("Audit<br>Unsupported<br>Claims", 1035, 334, 95, 64, text_style(13, bold=True))
    builder.vertex("Retrieve<br>Evidence<br>(Optional)", 1030, 404, 100, 64, text_style(13, bold=False))
    builder.vertex("Generate<br>Grounded<br>Hypotheses", 1022, 473, 112, 64, text_style(13, bold=True))
    builder.edge(edge_style(), source_point=(1056, 537), target_point=(1056, 561))
    add_document(builder, 1030, 560, 52, 66, fill="#A9BCE8")
    builder.vertex("", 1040, 572, 8, 8, ellipse_style("#F4DA6A", "#FFFFFF", extra="strokeWidth=2;"))
    builder.edge("html=1;strokeColor=#FFFFFF;strokeWidth=2;endArrow=none;", source_point=(1044, 580), target_point=(1044, 598))
    builder.edge("html=1;strokeColor=#FFFFFF;strokeWidth=2;endArrow=none;", source_point=(1044, 598), target_point=(1058, 590))
    for offset in [0, 6, 12, 18]:
        builder.edge("html=1;strokeColor=#FFFFFF;strokeWidth=2;endArrow=none;", source_point=(1053, 578 + offset), target_point=(1070, 578 + offset))
    builder.vertex("Research Blueprint<br>Artifact", 968, 621, 176, 36, text_style(15, bold=True))

    # Panel 6
    add_magnifier(builder, 1205, 186)
    builder.vertex(
        'Predictive Validity<br>Check: "If we used<br>literature up to date<br>X, do ideas appear<br>in future papers?"',
        1204,
        278,
        170,
        126,
        text_style(14, bold=False),
    )
    add_graph(
        builder,
        [
            (1248, 416, "#E7A160"),
            (1298, 416, "#A38CCC"),
            (1330, 417, "#7FBD89"),
            (1298, 460, "#A38CCC"),
            (1329, 438, "#79C3A7"),
        ],
        [(0, 1), (1, 2), (1, 3), (3, 4)],
        radius=7,
    )
    builder.vertex("Multiple Baselines", 1210, 455, 160, 34, text_style(13))
    builder.edge(edge_style(), source_point=(1288, 489), target_point=(1288, 524))
    add_document(builder, 1262, 525, 52, 68, fill="#E5EBF4")
    for y in [539, 547, 555, 563]:
        builder.edge("html=1;strokeColor=#5E646B;strokeWidth=2;endArrow=none;", source_point=(1274, y), target_point=(1300, y))
    builder.vertex("Manual Review<br>Packets", 1214, 598, 150, 42, text_style(14))

    # Section flow arrows
    builder.edge(edge_style(), source_point=(190, 581), target_point=(249, 581))
    builder.edge(edge_style(), source_point=(459, 337), target_point=(485, 337))
    builder.edge(edge_style(), source_point=(690, 528), target_point=(717, 528))
    builder.edge(edge_style(), source_point=(1082, 584), target_point=(1184, 584))
    builder.edge(edge_style(), source_point=(1164, 220), target_point=(1184, 220))

    builder.save(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
