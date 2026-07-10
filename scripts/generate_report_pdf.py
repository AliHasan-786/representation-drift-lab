from __future__ import annotations

import json
import shutil
from pathlib import Path

from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    HRFlowable,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/pdf/representation-drift-lab-report.pdf"
PUBLIC_OUTPUT = ROOT / "public/report/representation-drift-lab-report.pdf"

INK = colors.HexColor("#102019")
MUTED = colors.HexColor("#53645b")
GROUND = colors.HexColor("#071612")
PANEL = colors.HexColor("#10251e")
ACID = colors.HexColor("#c5f36b")
ORANGE = colors.HexColor("#ff8a4c")
CYAN = colors.HexColor("#49aa9f")
PAPER = colors.HexColor("#f2f5ef")
LINE = colors.HexColor("#cbd5cc")


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def clean(value: object) -> str:
    return (
        str(value)
        .replace("→", "to")
        .replace("·", "-")
        .replace("α", "alpha")
        .replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .replace("‑", "-")
    )


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def interval(value: dict, *, percent: bool = True) -> str:
    if percent:
        return f"{pct(value['mean'])} [{pct(value['ci_low'])}, {pct(value['ci_high'])}]"
    return f"{value['mean']:.4f} [{value['ci_low']:.4f}, {value['ci_high']:.4f}]"


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=33,
            leading=34,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=16,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=13,
            leading=19,
            textColor=MUTED,
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=27,
            textColor=INK,
            spaceBefore=8,
            spaceAfter=13,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=INK,
            spaceBefore=12,
            spaceAfter=7,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.3,
            leading=14,
            textColor=INK,
            spaceAfter=8,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.4,
            leading=10.5,
            textColor=MUTED,
        ),
        "eyebrow": ParagraphStyle(
            "Eyebrow",
            parent=base["BodyText"],
            fontName="Courier-Bold",
            fontSize=7.3,
            leading=10,
            textColor=colors.HexColor("#386c2c"),
            spaceAfter=8,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=18,
            textColor=INK,
            borderColor=ACID,
            borderWidth=1,
            borderPadding=12,
            backColor=colors.HexColor("#edf7dc"),
            spaceBefore=8,
            spaceAfter=12,
        ),
        "quote": ParagraphStyle(
            "Quote",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=16,
            textColor=INK,
            leftIndent=12,
            borderColor=ORANGE,
            borderWidth=0,
            borderLeft=3,
            borderPadding=9,
            spaceAfter=10,
        ),
        "toc": ParagraphStyle(
            "TOC",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=18,
            textColor=INK,
        ),
    }


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(clean(text), style)


def table_style(header: bool = True, font_size: float = 7.2) -> TableStyle:
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 3),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, PAPER]),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PANEL),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    return TableStyle(commands)


def metric_cards(items: list[tuple[str, str, str]]) -> Table:
    cells = []
    for label, value, note in items:
        cells.append(
            [
                paragraph(label.upper(), STYLES["small"]),
                Paragraph(value, ParagraphStyle("Metric", parent=STYLES["h1"], fontSize=20, leading=22, textColor=INK)),
                paragraph(note, STYLES["small"]),
            ]
        )
    table = Table([cells], colWidths=[2.15 * inch] * len(cells))
    table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.6, LINE),
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return table


def pareto_chart(methods: list[dict]) -> Drawing:
    width, height = 470, 270
    left, bottom, right, top = 55, 45, 450, 245
    drawing = Drawing(width, height)
    drawing.add(Rect(0, 0, width, height, fillColor=colors.white, strokeColor=LINE))
    drawing.add(Line(left, bottom, right, bottom, strokeColor=MUTED, strokeWidth=0.8))
    drawing.add(Line(left, bottom, left, top, strokeColor=MUTED, strokeWidth=0.8))
    drawing.add(String((left + right) / 2, 16, "Retained accuracy (higher is better)", textAnchor="middle", fontName="Helvetica", fontSize=8, fillColor=MUTED))
    drawing.add(String(12, (bottom + top) / 2, "Adapted accuracy", textAnchor="middle", fontName="Helvetica", fontSize=8, fillColor=MUTED, angle=90))
    palette = [CYAN, ORANGE, colors.HexColor("#5a8fd4"), colors.HexColor("#8f66b3"), colors.HexColor("#d2a72c")]

    def x(value: float) -> float:
        return left + (value - 0.55) / 0.25 * (right - left)

    def y(value: float) -> float:
        return bottom + (value - 0.72) / 0.30 * (top - bottom)

    for index, method in enumerate(methods):
        retained = method["metrics"]["final_retained_accuracy"]["mean"]
        adapted = method["metrics"]["final_adaptation_accuracy"]["mean"]
        color = palette[index % len(palette)]
        drawing.add(Circle(x(retained), y(adapted), 6.5, fillColor=color, strokeColor=INK, strokeWidth=0.5))
        drawing.add(String(x(retained), y(adapted) - 2.5, str(index + 1), textAnchor="middle", fontName="Helvetica-Bold", fontSize=6, fillColor=INK))
    return drawing


def domain_change_chart(scenarios: list[dict]) -> Drawing:
    width, height = 470, 210
    drawing = Drawing(width, height)
    drawing.add(Rect(0, 0, width, height, fillColor=colors.white, strokeColor=LINE))
    zero = 215
    drawing.add(Line(zero, 25, zero, 190, strokeColor=MUTED, strokeWidth=0.8))
    scale = 490
    for index, scenario in enumerate(scenarios):
        y = 158 - index * 52
        adaptation = scenario["metrics"]["adaptation_accuracy_change"]["mean"]
        retained = scenario["metrics"]["retained_accuracy_change"]["mean"]
        drawing.add(String(10, y + 12, clean(scenario["label"]), fontName="Helvetica-Bold", fontSize=8, fillColor=INK))
        drawing.add(Rect(zero, y, adaptation * scale, 10, fillColor=CYAN, strokeColor=None))
        retained_width = retained * scale
        drawing.add(Rect(zero if retained_width >= 0 else zero + retained_width, y - 14, abs(retained_width), 10, fillColor=ORANGE, strokeColor=None))
        drawing.add(String(zero + adaptation * scale + 4, y + 1, f"adapt {pct(adaptation, 1)}", fontName="Helvetica", fontSize=7, fillColor=MUTED))
        anchor = zero + retained_width + (4 if retained_width >= 0 else -4)
        drawing.add(String(anchor, y - 13, f"retain {pct(retained, 1)}", textAnchor="start" if retained_width >= 0 else "end", fontName="Helvetica", fontSize=7, fillColor=MUTED))
    drawing.add(String(10, 12, "Teal: adaptation change. Orange: retained-task change. Values are means across 3 seeds.", fontName="Helvetica", fontSize=7, fillColor=MUTED))
    return drawing


def on_page(canvas, document) -> None:
    page = canvas.getPageNumber()
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(0.65 * inch, 0.55 * inch, 7.85 * inch, 0.55 * inch)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.65 * inch, 0.35 * inch, "REPRESENTATION DRIFT LAB - ALI HASAN")
    canvas.drawRightString(7.85 * inch, 0.35 * inch, f"{page:02d}")
    canvas.restoreState()


def build_pdf() -> None:
    global STYLES
    STYLES = build_styles()
    method_data = read_json("public/data/method-comparison-local.json")
    domain_data = read_json("public/data/domain-comparison-local.json")
    interpolation = read_json("public/data/interpolation-local.json")
    warning = read_json("public/data/early-warning-methodology.json")
    benchmark = read_json("public/data/benchmark-local.json")
    methods = method_data["methods"]
    scenarios = domain_data["scenarios"]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=LETTER,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.72 * inch,
        title="Representation Drift Lab",
        author="Ali Hasan",
        subject="A reproducible study of adaptation, representation drift, and forgetting in vision-language models",
    )
    story = []

    # Cover
    story.extend(
        [
            Spacer(1, 0.55 * inch),
            paragraph("INDEPENDENT POST-COURSE RESEARCH EXTENSION", STYLES["eyebrow"]),
            paragraph("Representation Drift Lab", STYLES["title"]),
            paragraph(
                "When a model learns something new, what does it forget - and can changes inside the model warn us before capabilities are lost?",
                STYLES["subtitle"],
            ),
            Spacer(1, 0.18 * inch),
            HRFlowable(width="100%", thickness=2, color=ACID, spaceBefore=8, spaceAfter=18),
            metric_cards(
                [
                    ("Measured methods", "9", "Each run across 3 independent seeds"),
                    ("Domain scenarios", "3", "Food, satellite, pet, object, and digit data"),
                    ("Automated checks", "41", "Research, artifact, and web tests"),
                ]
            ),
            Spacer(1, 0.35 * inch),
            paragraph(
                "A portfolio research project by <b>Ali Hasan</b><br/>Original 2025 course project with Sahil Mhatre and Corey Chen<br/>Independent extension: Ali Hasan, 2026",
                STYLES["body"],
            ),
            Spacer(1, 0.35 * inch),
            paragraph(
                "Evidence status: LOCAL MULTI-SEED PRELIMINARY. The system and methods are real; the local samples are intentionally small and do not establish universal model rankings.",
                STYLES["callout"],
            ),
            Spacer(1, 0.2 * inch),
            paragraph(
                f"Artifact ID: {method_data['run_id']}<br/>Configuration: {method_data['config_hash']}<br/>Generated from checksummed public artifacts.",
                STYLES["small"],
            ),
            PageBreak(),
        ]
    )

    # Executive summary and TOC
    story.extend(
        [
            paragraph("01 / START HERE", STYLES["eyebrow"]),
            paragraph("The project in plain language", STYLES["h1"]),
            paragraph(
                "Modern AI models can be adapted to new jobs without being rebuilt from scratch. The risk is that learning the new job can quietly damage abilities the model already had. This project treats that risk as something we can measure, visualize, test, and potentially control.",
                STYLES["body"],
            ),
            paragraph(
                "I adapted CLIP - a model that connects images and language - to recognize new image domains. At multiple points during learning, I measured both visible behavior (accuracy) and internal behavior (how image representations moved across every vision-transformer layer).",
                STYLES["body"],
            ),
            paragraph(
                "The central finding is not that drift is always bad. Across the completed scenarios, drift sometimes accompanied forgetting, sometimes coexisted with stable accuracy, and once accompanied positive transfer. Geometry is useful evidence, but it is not a universal substitute for testing the capability itself.",
                STYLES["callout"],
            ),
            paragraph("How to read the technical terms", STYLES["h2"]),
            LongTable(
                [
                    ["Term", "Meaning in this project"],
                    ["Adaptation", "Teaching an existing model a new image-recognition job."],
                    ["Retention", "Checking whether an older capability still works afterward."],
                    ["Representation", "The numeric internal map the model builds for an image."],
                    ["Drift", "How much that internal map changes during adaptation."],
                    ["CKA", "A similarity score for two sets of internal representations. 1 means highly similar."],
                    ["LoRA", "A small trainable adapter that changes a model with far fewer parameters."],
                    ["Checkpoint", "A saved snapshot of the model during training."],
                ],
                colWidths=[1.15 * inch, 5.75 * inch],
                repeatRows=1,
                style=table_style(font_size=8),
            ),
            Spacer(1, 0.18 * inch),
            paragraph("Report map", STYLES["h2"]),
            paragraph(
                "02 - Question, origin, and code audit<br/>03 - Experimental design<br/>04 - Nine-method benchmark<br/>05 - Three-domain stress test<br/>06 - Diagnostics and failure cases<br/>07 - Reproducible system design<br/>08 - Limitations and next research<br/>09 - Reproduction and sources",
                STYLES["toc"],
            ),
            PageBreak(),
        ]
    )

    # Origin
    story.extend(
        [
            paragraph("02 / QUESTION AND ORIGIN", STYLES["eyebrow"]),
            paragraph("From a course observation to a research system", STYLES["h1"]),
            paragraph(
                "The original Cornell Tech CS 5787 group project studied catastrophic forgetting while adapting CLIP from general zero-shot recognition toward Food-101. It found a visible stability-plasticity tension: the target task improved while CIFAR-10 performance declined.",
                STYLES["body"],
            ),
            paragraph(
                "The archived result was interesting but difficult to extend safely. Some saved outputs did not have their producing code, two experiment families used different schedules, and an early drift forecast failed badly. The extension therefore began by separating historical evidence from newly reproduced evidence.",
                STYLES["body"],
            ),
            paragraph(
                "Historical negative result: a step-8,000 forecast predicted final cosine drift of 0.6647, while the saved run ended at 0.3055 - an absolute error of 0.3592.",
                STYLES["quote"],
            ),
            paragraph("A newly recovered source scaffold", STYLES["h2"]),
            paragraph(
                "A later-supplied folder named <b>DL Final</b> adds 2025 CLIP/LoRA source, tests, scripts, and two notebook templates. It adds lineage, not performance evidence: it contains no datasets, checkpoints, embeddings, results, logs, or executed notebook outputs. Static audit found seven explicit placeholders, incompatible pipeline options, missing test imports, incomplete LoRA checkpoint reconstruction, duplicate-caption false negatives, and future-information leakage in its early forecast path. It also evaluates COCO rather than the report's CIFAR-10 path, so it is not treated as the missing producer for the submitted results.",
                STYLES["body"],
            ),
            paragraph("What the extension adds", STYLES["h2"]),
            LongTable(
                [
                    ["Layer", "Extension"],
                    ["Research", paragraph("Multiple methods, seeds, domains, uncertainty intervals, held-out prediction protocol, and negative cases.", STYLES["small"])],
                    ["Diagnostics", paragraph("Task metrics, calibration, fixed projections, CKA, Frechet distance, effective rank, neighborhoods, classes, layers, and cross-modal alignment.", STYLES["small"])],
                    ["Engineering", paragraph("Configuration identity, safe checkpoints, cached immutable selections, environment capture, deterministic regeneration, schemas, and checksums.", STYLES["small"])],
                    ["Product", paragraph("An accessible React case study with synchronized controls, Pareto comparisons, class inspection, recovery simulation, and source-level provenance.", STYLES["small"])],
                ],
                colWidths=[1.15 * inch, 5.75 * inch],
                repeatRows=1,
                style=table_style(font_size=8),
            ),
            Spacer(1, 0.18 * inch),
            paragraph(
                "Attribution: Sahil Mhatre, Ali Hasan, and Corey Chen completed the original 2025 group project. The rebuilt repository, corrected pipeline, new experiments, diagnostics, benchmark system, report, and interactive application are Ali Hasan's independent post-course extension.",
                STYLES["small"],
            ),
            PageBreak(),
        ]
    )

    # Design
    story.extend(
        [
            paragraph("03 / EXPERIMENTAL DESIGN", STYLES["eyebrow"]),
            paragraph("Measure learning and forgetting at the same time", STYLES["h1"]),
            paragraph(
                "Every experiment has an adaptation domain and a retained domain. The model trains only for the adaptation objective. At saved checkpoints, both domains are evaluated, and the same retained images are passed through the frozen baseline and current model for paired geometry measurements.",
                STYLES["body"],
            ),
            metric_cards(
                [
                    ("Seeds", "41 / 42 / 43", "Independent model and data selections"),
                    ("Base model", "CLIP ViT-B/32", "Pinned model revision"),
                    ("Primary adapter", "294,912", "Trainable LoRA parameters"),
                ]
            ),
            paragraph("Scientific controls", STYLES["h2"]),
            paragraph(
                "All same-class captions in a batch are positives, avoiding false negatives from repeated labels. Dimensionality reduction is fit once on baseline embeddings and reused, so plotted lines are real paired movement rather than independent t-SNE layouts. Frechet-style distance uses baseline-fixed PCA and covariance regularization. Undefined statistics, such as CKA on constant text prototypes, are published as null with an explicit definition flag.",
                STYLES["body"],
            ),
            paragraph("The evidence ladder", STYLES["h2"]),
            LongTable(
                [
                    ["Tier", "Purpose", "Publication treatment"],
                    ["Historical", "Preserve original course artifacts and claims", "Clearly labeled, not merged with reproduced results"],
                    ["Smoke", "Test the complete CPU pipeline quickly", "Engineering evidence only"],
                    ["Local multi-seed", "Validate methods, statistics, and interactions", "Preliminary, with intervals and caveats"],
                    ["Portfolio / research", "Larger model and domain coverage", "Required before strong scientific rankings"],
                ],
                colWidths=[1.15 * inch, 2.65 * inch, 3.1 * inch],
                repeatRows=1,
                style=table_style(font_size=7.7),
            ),
            PageBreak(),
        ]
    )

    # Methods
    story.extend(
        [
            paragraph("04 / NINE-METHOD BENCHMARK", STYLES["eyebrow"]),
            paragraph("Different ways to adapt - and recover - the model", STYLES["h1"]),
            paragraph(
                "The method benchmark covers frozen probing, full-parameter tuning, initialization-aware tuning, parameter-efficient adapters, retention distillation, gradient projection, and weight-space recovery. Paper-derived methods are labeled as adapted or inspired when the local protocol is not an exact reproduction.",
                STYLES["body"],
            ),
            pareto_chart(methods),
            Spacer(1, 0.08 * inch),
        ]
    )
    legend_rows = [["#", "Method", "Adapted", "Retained", "1 - CKA", "Parameters"]]
    for index, method in enumerate(methods):
        metrics = method["metrics"]
        legend_rows.append(
            [
                str(index + 1),
                clean(method["label"]),
                pct(metrics["final_adaptation_accuracy"]["mean"]),
                pct(metrics["final_retained_accuracy"]["mean"]),
                f"{metrics['retained_cka_loss']['mean']:.4f}",
                f"{int(metrics['trainable_parameters']['mean']):,}",
            ]
        )
    story.extend(
        [
            LongTable(
                legend_rows,
                colWidths=[0.3 * inch, 2.25 * inch, 0.85 * inch, 0.85 * inch, 0.75 * inch, 1.15 * inch],
                repeatRows=1,
                style=table_style(font_size=6.8),
            ),
            paragraph(
                "Reading the chart: upper-right is better. The frozen probe and LP-FT reach a tiny-subset ceiling. Full fine-tuning from a random head underperforms the zero-shot adaptation baseline. WiSE-FT recovers retention from its compatible fine-tuned source. These are intervention diagnostics, not universal rankings.",
                STYLES["small"],
            ),
            PageBreak(),
        ]
    )

    # Domain
    story.extend(
        [
            paragraph("05 / THREE-DOMAIN STRESS TEST", STYLES["eyebrow"]),
            paragraph("Drift does not mean the same thing everywhere", STYLES["h1"]),
            paragraph(
                "The same rank-8 LoRA method was run on three adaptation and three retained domains. This tests whether the Food-101 story generalizes even before adding new model families.",
                STYLES["body"],
            ),
            domain_change_chart(scenarios),
            Spacer(1, 0.1 * inch),
        ]
    )
    domain_rows = [["Scenario", "Adaptation change", "Retention change", "Retained 1 - CKA", "Samples / seed"]]
    for scenario in scenarios:
        metrics = scenario["metrics"]
        counts = scenario["sample_counts_per_seed"]
        domain_rows.append(
            [
                clean(scenario["label"]),
                interval(metrics["adaptation_accuracy_change"]),
                interval(metrics["retained_accuracy_change"]),
                interval(metrics["retained_cka_loss"], percent=False),
                f"{counts['adaptation_eval']} / {counts['retained_eval']}",
            ]
        )
    story.extend(
        [
            LongTable(
                domain_rows,
                colWidths=[1.35 * inch, 1.55 * inch, 1.55 * inch, 1.6 * inch, 0.85 * inch],
                repeatRows=1,
                style=table_style(font_size=6.7),
            ),
            paragraph("Food-101 to CIFAR-10", STYLES["h2"]),
            paragraph("Adaptation improved by 6.9 points while retention fell by 3.3 points. Mean CKA loss was small (0.0126).", STYLES["body"]),
            paragraph("EuroSAT to CIFAR-100", STYLES["h2"]),
            paragraph("Adaptation improved by 31.9 points and retention fell by 1.7 points, while CKA loss was much larger (0.1558). Large drift did not imply proportionally large forgetting.", STYLES["body"]),
            paragraph("Oxford Pets to MNIST", STYLES["h2"]),
            paragraph("Adaptation improved by 8.3 points and retention improved by 10.8 points while CKA loss reached 0.0293. Drift accompanied positive transfer, not forgetting.", STYLES["body"]),
            paragraph("Excluded exploratory pair", STYLES["h2"]),
            paragraph("A separate Pets-to-EuroSAT artifact began at 0% retained accuracy in every seed. A floor cannot reveal additional forgetting, so the artifact is preserved as a failed validity check and excluded from this three-scenario comparison.", STYLES["small"]),
            PageBreak(),
        ]
    )

    # Diagnostics
    association = method_data["analysis"]["retained_cka_loss_vs_mean_forgetting"]
    warning_model = warning["evaluation"]["test_metrics"]["early_warning_model"]
    warning_baseline = warning["evaluation"]["test_metrics"]["train_mean_baseline"]
    story.extend(
        [
            paragraph("06 / DIAGNOSTICS AND FAILURE CASES", STYLES["eyebrow"]),
            paragraph("The most useful result is where simple stories fail", STYLES["h1"]),
            paragraph(
                f"Across nine intervention means, retained CKA loss and forgetting had Pearson correlation {association['pearson']:.3f}, Spearman correlation {association['spearman']:.3f}, and a bootstrap Pearson interval [{association['pearson_ci_low']:.3f}, {association['pearson_ci_high']:.3f}]. The interval is too wide for a reliable method-ranking claim.",
                STYLES["callout"],
            ),
            paragraph("Failure case 1 - drift without forgetting", STYLES["h2"]),
            paragraph("Retention distillation returned mean CIFAR-10 accuracy to baseline while internal representations still moved. Stable accuracy does not imply an unchanged model.", STYLES["body"]),
            paragraph("Failure case 2 - lower drift with worse retention", STYLES["h2"]),
            paragraph("The gradient null-space baseline had lower mean CKA loss than standard LoRA but worse retained accuracy. A geometry score alone ranked these interventions incorrectly.", STYLES["body"]),
            paragraph("Failure case 3 - drift with positive transfer", STYLES["h2"]),
            paragraph("Pets adaptation improved MNIST accuracy even while MNIST representations moved. Drift measures change; they do not determine whether the change helps or harms a task.", STYLES["body"]),
            paragraph("Early-warning protocol", STYLES["h2"]),
            paragraph(
                f"A held-out train/validation/test evaluator was implemented with naive baselines, calibration, prediction intervals, and error metrics. On synthetic methodology data, model RMSE was {warning_model['rmse']:.3f} versus {warning_baseline['rmse']:.3f} for the train-mean baseline. This validates the evaluation machinery only - it is not evidence that real CLIP forgetting is predictable.",
                STYLES["body"],
            ),
            paragraph("Layer and class analysis", STYLES["h2"]),
            paragraph("Every CLIP vision block is measured against baseline. The application also exposes per-class accuracy, confusion counts, centroid movement, fixed-PCA sample paths, image-space drift, text-space drift, and image-text alignment. Undefined statistics remain explicit rather than being coerced into numbers.", STYLES["body"]),
            PageBreak(),
        ]
    )

    # Engineering
    story.extend(
        [
            paragraph("07 / REPRODUCIBLE SYSTEM DESIGN", STYLES["eyebrow"]),
            paragraph("The research pipeline is part of the result", STYLES["h1"]),
            paragraph(
                "This project is structured as a reusable system rather than a single notebook. Experiments are CLI-driven and configuration-defined. Large checkpoints and embeddings stay outside Git; compact public derivatives point to checksummed manifests.",
                STYLES["body"],
            ),
            LongTable(
                [
                    ["Capability", "Implementation"],
                    ["Identity", "Run IDs and configuration hashes separate scientific settings from storage paths."],
                    ["Data", "Resolved repository revisions, deterministic selections, row indices, fingerprints, and revision-aware caches."],
                    ["Model", "Resolved model commit, trainable-parameter invariants, strategy fidelity labels, and environment versions."],
                    ["Resume", "Checkpointed LoRA state, optimizer state, and random-generator state; complete classification runs are idempotent."],
                    ["Regeneration", "Metrics and web derivatives rebuild deterministically from saved outputs without retraining."],
                    ["Validation", "41 Python checks, web interaction tests, TypeScript, manifest checks, responsive QA, and production budgets."],
                    ["Deployment", "Core charts use precomputed artifacts; no GPU or inference service is required for the public experience."],
                ],
                colWidths=[1.2 * inch, 5.7 * inch],
                repeatRows=1,
                style=table_style(font_size=7.6),
            ),
            Spacer(1, 0.15 * inch),
            paragraph("Web performance budgets", STYLES["h2"]),
            metric_cards(
                [
                    ("JavaScript", "~68 KB", "Gzip production bundle"),
                    ("Core benchmark", "~199 KB", "Initial multi-seed evidence"),
                    ("Detailed microscope", "~846 KB", "Loaded only on request"),
                ]
            ),
            paragraph(
                "The visual experience was checked at mobile and wide-desktop widths. A real-browser review caught overlapping Pareto labels after the method expansion; the chart was redesigned with numbered markers and a structured legend.",
                STYLES["small"],
            ),
            PageBreak(),
        ]
    )

    # Limitations
    story.extend(
        [
            paragraph("08 / LIMITATIONS AND NEXT RESEARCH", STYLES["eyebrow"]),
            paragraph("What the current evidence cannot support", STYLES["h1"]),
            LongTable(
                [
                    ["Limitation", "Why it matters"],
                    ["Small local samples", "Confidence intervals are wide and some tasks hit ceiling effects."],
                    ["Three seeds", "This is the minimum uncertainty gate, not a large statistical sample."],
                    ["One foundation backbone", "Nine methods and three domains still use CLIP ViT-B/32."],
                    ["Short local schedules", "Method outcomes may change under realistic compute and hyperparameter selection."],
                    ["Reference-data asymmetry", "Distillation sees retained examples; plain LoRA does not. That is a resource trade-off."],
                    ["Exploratory selection", "LoRA and WiSE interpolation coefficients are inspected on local evaluation data."],
                    ["Association is not mechanism", "Correlation between geometry and accuracy does not establish causation."],
                ],
                colWidths=[1.55 * inch, 5.35 * inch],
                repeatRows=1,
                style=table_style(font_size=7.7),
            ),
            paragraph("Next research gates", STYLES["h2"]),
            paragraph(
                "1. Add at least two model/control families, beginning with a modern vision-language model and a visual-only control.<br/>2. Run larger fixed samples and realistic schedules on the three domain scenarios.<br/>3. Add adaptive-rank LoRA and MergeTune-style recovery.<br/>4. Train the early-warning model only after enough independent methods and scenarios exist for held-out evaluation.<br/>5. Repeat the strongest comparisons on external GPU compute and publish compute-normalized cost.",
                STYLES["body"],
            ),
            Spacer(1, 0.12 * inch),
            paragraph(
                "Until those gates complete, the public experience presents measured results as preliminary protocol evidence and emphasizes the engineering, methodology, and negative findings.",
                STYLES["callout"],
            ),
            PageBreak(),
        ]
    )

    # Reproduce and sources
    story.extend(
        [
            paragraph("09 / REPRODUCTION AND SOURCES", STYLES["eyebrow"]),
            paragraph("How to reproduce and verify the work", STYLES["h1"]),
            paragraph("Core commands", STYLES["h2"]),
            Paragraph(
                "<font name='Courier' size='7'>make test<br/>make benchmark-local<br/>make methods-local<br/>PYTHONPATH=src .venv/bin/python -m driftlab aggregate-domains<br/>cd apps/web &amp;&amp; npm test &amp;&amp; npm run build</font>",
                STYLES["body"],
            ),
            paragraph("Primary research sources", STYLES["h2"]),
            LongTable(
                [
                    ["Topic", "Primary source"],
                    ["CLIP", "Radford et al. (2021), https://arxiv.org/abs/2103.00020"],
                    ["LoRA", "Hu et al. (2021), https://arxiv.org/abs/2106.09685"],
                    ["CKA", "Kornblith et al. (2019), https://proceedings.mlr.press/v97/kornblith19a"],
                    ["LP-FT", "Kumar et al. (2022), https://arxiv.org/abs/2202.10054"],
                    ["WiSE-FT", "Wortsman et al. (2022), CVPR open-access paper"],
                    ["ZSCL", "Zheng et al. (2023), ICCV open-access paper"],
                ],
                colWidths=[1.1 * inch, 5.8 * inch],
                repeatRows=1,
                style=table_style(font_size=7.4),
            ),
            paragraph("Artifact verification", STYLES["h2"]),
            paragraph(
                f"Method comparison run: {method_data['run_id']}<br/>Method config hash: {method_data['config_hash']}<br/>Method manifest SHA-256: {method_data['source_manifest']['sha256']}<br/><br/>Domain comparison run: {domain_data['run_id']}<br/>Domain config hash: {domain_data['config_hash']}<br/>Domain manifest SHA-256: {domain_data['source_manifest']['sha256']}<br/><br/>Baseline benchmark run: {benchmark['run_id']}<br/>Baseline config hash: {benchmark['config_hash']}<br/><br/>Historical code archive SHA-256: 957682e7d364010b8d05f452eca1e141967daf00b53eb58c2946ca3bdb5ff645",
                STYLES["small"],
            ),
            Spacer(1, 0.18 * inch),
            paragraph(
                "This report is generated from the same public JSON artifacts used by the portfolio application. If a source manifest checksum changes, the production web build rejects the mismatch.",
                STYLES["callout"],
            ),
        ]
    )

    document.build(story, onFirstPage=on_page, onLaterPages=on_page)
    PUBLIC_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT, PUBLIC_OUTPUT)
    print(OUTPUT)
    print(PUBLIC_OUTPUT)


if __name__ == "__main__":
    build_pdf()
