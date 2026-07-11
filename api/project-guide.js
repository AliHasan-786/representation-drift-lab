const PROJECT_CONTEXT = `
Representation Drift Lab is Ali Hasan's independent extension of a 2025 Cornell Tech CS 5787 group project with Sahil Mhatre and Corey Chen.

Ground truth:
- CLIP means Contrastive Language-Image Pre-training. The project uses the already-pretrained openai/clip-vit-base-patch32 model; it did not train CLIP from scratch.
- CLIP has image and text encoders that place matching images and descriptions in a shared representation space.
- LoRA means Low-Rank Adaptation. It freezes the original CLIP parameters and trains small adapters in q_proj and v_proj attention projections.
- The main LoRA setup trains 294,912 parameters, about 0.34% of an approximately 88-million-parameter model.
- A dataset is a labeled collection of examples. A training or adaptation dataset changes the model; a retained dataset is evaluated without contributing training updates.
- Three small local dataset pairs were run: Food-101 to CIFAR-10, EuroSAT to CIFAR-100, and Oxford-IIIT Pet to MNIST.
- A fourth exploratory Oxford Pets to EuroSAT artifact is preserved but excluded: retained EuroSAT accuracy was 0% at baseline in every seed, so the run could not measure additional forgetting and supports no scientific comparison.
- Food-101 contains dishes; CIFAR-10 contains tiny everyday object and animal images; EuroSAT contains satellite land-use images; CIFAR-100 contains 100 tiny object categories; Oxford Pets contains 37 cat and dog breeds; MNIST contains handwritten digits.
- A checkpoint is a saved state during training. The primary local LoRA runs expose steps 0, 5, 10, and 20.
- Accuracy is the share of held-out images assigned the correct class. Adaptation accuracy checks the new task. Retained accuracy checks the older capability.
- A representation is the model's internal numeric description of an input. CKA compares sets of representations. CKA near 1 means similar; 1 minus CKA is shown as drift, where 0 means no measured change.
- Drift means internal change. Forgetting means reduced task performance. The project found they are not interchangeable.
- In Food-101 to CIFAR-10, adaptation improved 6.9 percentage points and retention fell 3.3 points with mean CKA loss 0.0126.
- In EuroSAT to CIFAR-100, adaptation improved 31.9 points and retention fell 1.7 points with much larger mean CKA loss 0.1558.
- In Pets to MNIST, adaptation improved 8.3 points and retained MNIST improved 10.8 points while CKA loss was 0.0293: drift accompanied positive transfer.
- Nine intervention methods and three independent seeds (41, 42, 43) are included, but samples and schedules are intentionally small. Results are preliminary protocol evidence, not universal model rankings.
- The original submitted report is preserved separately. The extension corrected and rebuilt the pipeline, added provenance, tests, seeds, methods, domains, uncertainty, failure cases, and interactive explanations.
- A newly supplied historical folder named DL Final contains a source-code scaffold: 32 Python files, three shell scripts, two unexecuted notebook templates, and no datasets, checkpoints, embeddings, results, or logs. It adds code-lineage evidence, not new performance evidence.
- The historical scaffold planned CLIP ViT-B/32, rank-8/alpha-16 vision LoRA, Food-101 training, primarily COCO evaluation, multimodal drift metrics, semantic clusters, and interactive plots. It cannot be assumed to have produced the CIFAR-10 report path.
- Its audit found explicit placeholder modules, incompatible pipeline CLI options, missing test imports, incomplete adapter checkpoint restoration, duplicate-caption false negatives, performance/geometry conflation, and future-information leakage in early forecasting. The independent extension addresses these issues.
- If a requested fact is not in this context, say that the current project evidence does not establish it. Never invent a number or universal claim.
`;

const RATE_LIMIT_WINDOW_MS = 10 * 60 * 1000;
const RATE_LIMIT_MAX = 10;
const rateBuckets = new Map();
const SENSITIVE_INPUT_PATTERNS = [
  ["an email address", /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/i],
  ["a Social Security number", /\b\d{3}-\d{2}-\d{4}\b/],
  ["a payment-card number", /\b(?:\d[ -]?){13,19}\b/],
  ["a phone number", /(?:\+?\d[\s().-]*){10,15}/],
  ["an access credential", /\b(?:sk|rk|pk)[_-][A-Za-z0-9_-]{16,}\b/],
];

export function sensitiveInputReason(text) {
  const value = String(text ?? "");
  return SENSITIVE_INPUT_PATTERNS.find(([, pattern]) => pattern.test(value))?.[0] ?? null;
}

function requestIdentity(request) {
  const forwarded = request.headers["x-forwarded-for"];
  return String(Array.isArray(forwarded) ? forwarded[0] : forwarded || request.socket?.remoteAddress || "unknown")
    .split(",")[0]
    .trim();
}

function consumeRateLimit(request, now = Date.now()) {
  const identity = requestIdentity(request);
  const current = rateBuckets.get(identity);
  const bucket = !current || current.resetAt <= now
    ? { count: 0, resetAt: now + RATE_LIMIT_WINDOW_MS }
    : current;
  bucket.count += 1;
  rateBuckets.set(identity, bucket);
  if (rateBuckets.size > 500) {
    for (const [key, value] of rateBuckets) if (value.resetAt <= now) rateBuckets.delete(key);
  }
  return { allowed: bucket.count <= RATE_LIMIT_MAX, remaining: Math.max(0, RATE_LIMIT_MAX - bucket.count), resetAt: bucket.resetAt };
}

function hasValidOrigin(request) {
  const origin = request.headers.origin;
  if (!origin) return true;
  const forwardedHost = request.headers["x-forwarded-host"];
  const host = String(Array.isArray(forwardedHost) ? forwardedHost[0] : forwardedHost || request.headers.host || "");
  try {
    return new URL(origin).host === host;
  } catch {
    return false;
  }
}

function textFromResponse(payload) {
  if (typeof payload.output_text === "string") return payload.output_text;
  return (payload.output ?? [])
    .flatMap((item) => item.content ?? [])
    .filter((item) => item.type === "output_text")
    .map((item) => item.text)
    .join("\n")
    .trim();
}

export default async function handler(request, response) {
  response.setHeader("Cache-Control", "no-store");
  if (request.method !== "POST") {
    response.setHeader("Allow", "POST");
    return response.status(405).json({ error: "Use POST." });
  }
  if (!hasValidOrigin(request)) {
    return response.status(403).json({ error: "Cross-origin requests are not allowed." });
  }
  const rateLimit = consumeRateLimit(request);
  response.setHeader("X-RateLimit-Remaining", String(rateLimit.remaining));
  if (!rateLimit.allowed) {
    response.setHeader("Retry-After", String(Math.ceil((rateLimit.resetAt - Date.now()) / 1000)));
    return response.status(429).json({ error: "Too many questions. Please try again later." });
  }
  const question = typeof request.body?.question === "string" ? request.body.question.trim().slice(0, 800) : "";
  if (!question) return response.status(400).json({ error: "A question is required." });
  const sensitiveReason = sensitiveInputReason(question);
  if (sensitiveReason) {
    return response.status(400).json({
      error: `For privacy, do not enter ${sensitiveReason}. Ask the project question without personal or secret information.`,
    });
  }
  if (!process.env.OPENAI_API_KEY) {
    return response.status(503).json({ error: "The grounded GenAI guide is not configured; use the offline guide." });
  }
  try {
    const apiResponse = await fetch("https://api.openai.com/v1/responses", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: process.env.OPENAI_MODEL || "gpt-5.4-mini",
        instructions: `You are the project guide for a public portfolio. Explain from absolute zero in 2-4 short paragraphs. Define every technical term before using it. Answer only from the supplied project context, distinguish the original report from the independent extension, preserve caveats, and never imply the visitor should already know something.\n\n${PROJECT_CONTEXT}`,
        input: question,
        max_output_tokens: 500,
        store: false,
      }),
    });
    if (!apiResponse.ok) return response.status(502).json({ error: "The model provider did not return an answer." });
    const payload = await apiResponse.json();
    const answer = textFromResponse(payload);
    if (!answer) return response.status(502).json({ error: "The model returned no readable answer." });
    return response.status(200).json({ answer });
  } catch {
    return response.status(502).json({ error: "The grounded guide is temporarily unavailable." });
  }
}
