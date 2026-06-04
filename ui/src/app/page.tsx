"use client";

import { FormEvent, useMemo, useState } from "react";

type LabelScore = {
  label: string;
  confidence: number;
  explanation: string;
  low_confidence: boolean;
};

type Action = {
  label: string;
  decision_source: string;
};

type ExplainMeta = {
  explain_source?: string;
  error_code?: string | null;
};

type Classification = {
  sentiment: LabelScore;
  topic: LabelScore;
  intent: LabelScore;
  action: Action;
};

type ApiResponse = Classification & {
  // /explain-classification wraps the classification under `classification`
  classification?: Classification;
  explanation?: {
    summary?: string;
    rationale?: string;
    limitations?: string;
  } | null;
  explain_meta?: ExplainMeta;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  result?: ApiResponse;
  isError?: boolean;
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

const EXAMPLES = [
  "التطبيق يعلق عند الدفع ومحدش يرد عليّ من خدمة العملاء",
  "حولت مبلغ ومارجع لي وخدمة العملاء ما ردوا علي",
  "الواجهة الجديدة جميلة جداً، شكراً للفريق على التطوير",
  "أحتاج تعديل بياناتي في الحساب من فضلكم",
];

const BENCHMARK = {
  accuracy: "68.3%",
  f1: "0.674",
  latencyP50: "96 ms",
  latencyP95: "253 ms",
  samples: "500",
  dataset: "arbml/Arabic_Sentiment_Twitter_Corpus",
};

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [useExplanation, setUseExplanation] = useState(true);

  const canSend = useMemo(
    () => input.trim().length > 0 && !isLoading,
    [input, isLoading],
  );

  const submit = async (text: string) => {
    if (!text.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      text: text.trim(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const endpoint = useExplanation
        ? `${API_BASE_URL}/explain-classification`
        : `${API_BASE_URL}/predict`;

      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim() }),
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      const raw = (await response.json()) as ApiResponse;
      // /explain-classification nests the labels under `classification`;
      // /predict returns them at the top level. Normalize to a single shape.
      const cls: Classification = raw.classification ?? {
        sentiment: raw.sentiment,
        topic: raw.topic,
        intent: raw.intent,
        action: raw.action,
      };
      const normalized: ApiResponse = {
        ...cls,
        explanation: raw.explanation ?? null,
        explain_meta: raw.explain_meta,
      };
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: "Classified.",
          result: normalized,
        },
      ]);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unexpected request failure";
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: `Request failed: ${message}`,
          isError: true,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    await submit(input);
  };

  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
      <Nav />

      <main className="mx-auto w-full max-w-6xl px-4 py-10 md:py-16">
        <Hero />

        <BenchmarkStrip />

        <HowItWorks />

        <section
          id="demo"
          className="mt-16 scroll-mt-24 rounded-3xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900 md:p-8"
        >
          <div className="mb-6 flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
            <div>
              <h2 className="text-2xl font-semibold">Try it</h2>
              <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
                Paste an Arabic complaint or pick an example. Calls{" "}
                <code className="rounded bg-zinc-100 px-1 py-0.5 text-xs dark:bg-zinc-800">
                  {API_BASE_URL}
                </code>
                .
              </p>
            </div>
            <label className="flex cursor-pointer items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={useExplanation}
                onChange={(e) => setUseExplanation(e.target.checked)}
              />
              Include LLM explanation
            </label>
          </div>

          <div className="mb-4 flex flex-wrap gap-2">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                type="button"
                onClick={() => submit(ex)}
                disabled={isLoading}
                dir="rtl"
                className="rounded-full border border-zinc-200 bg-zinc-50 px-3 py-1 text-xs text-zinc-700 transition hover:border-blue-300 hover:bg-blue-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-200 dark:hover:border-blue-700 dark:hover:bg-blue-950/40"
              >
                {ex}
              </button>
            ))}
          </div>

          <form onSubmit={onSubmit} className="space-y-3">
            <textarea
              id="complaint-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="اكتب الشكوى هنا..."
              className="min-h-28 w-full rounded-xl border border-zinc-300 bg-white p-3 text-sm outline-none ring-blue-500 focus:ring-2 dark:border-zinc-700 dark:bg-zinc-950"
              dir="rtl"
            />
            <div className="flex items-center justify-between">
              <p className="text-xs text-zinc-500 dark:text-zinc-400">
                Arabic complaints. Deterministic routing. Fallback if LLM fails.
              </p>
              <button
                type="submit"
                disabled={!canSend}
                className="rounded-xl bg-blue-600 px-5 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-blue-400"
              >
                {isLoading ? "Analyzing..." : "Analyze"}
              </button>
            </div>
          </form>

          <div className="mt-6 space-y-3">
            {messages.length === 0 ? (
              <p className="rounded-xl border border-dashed border-zinc-200 p-4 text-center text-sm text-zinc-500 dark:border-zinc-700 dark:text-zinc-400">
                Submit a complaint to see sentiment, topic, intent, the
                deterministic action, and (optionally) an LLM explanation.
              </p>
            ) : null}

            {messages.map((message) => (
              <article
                key={message.id}
                className={`rounded-2xl border p-4 ${
                  message.role === "user"
                    ? "ml-auto max-w-2xl border-blue-200 bg-blue-50 dark:border-blue-900 dark:bg-blue-950/40"
                    : "mr-auto max-w-4xl border-zinc-200 bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-800/60"
                }`}
                dir={message.role === "user" ? "rtl" : "ltr"}
              >
                <p className="mb-3 text-sm">{message.text}</p>

                {message.result ? (
                  <div className="grid gap-3 md:grid-cols-2">
                    <ResultCard
                      title="Sentiment"
                      score={message.result.sentiment}
                    />
                    <ResultCard title="Topic" score={message.result.topic} />
                    <ResultCard title="Intent" score={message.result.intent} />
                    <div className="rounded-xl border border-zinc-200 bg-white p-3 dark:border-zinc-700 dark:bg-zinc-900">
                      <h3 className="text-sm font-semibold">Action</h3>
                      <p className="mt-1 text-sm font-medium text-blue-700 dark:text-blue-300">
                        {message.result.action.label}
                      </p>
                      <p className="text-xs text-zinc-500 dark:text-zinc-400">
                        source: {message.result.action.decision_source}
                      </p>
                    </div>

                    {message.result.explanation ? (
                      <div className="rounded-xl border border-zinc-200 bg-white p-3 md:col-span-2 dark:border-zinc-700 dark:bg-zinc-900">
                        <h3 className="text-sm font-semibold">
                          LLM Explanation
                        </h3>
                        <p className="mt-1 text-sm">
                          {message.result.explanation.summary ?? "No summary"}
                        </p>
                        <p className="mt-1 text-xs text-zinc-600 dark:text-zinc-300">
                          {message.result.explanation.rationale ??
                            "No rationale"}
                        </p>
                      </div>
                    ) : null}

                    {message.result.explain_meta ? (
                      <div
                        className={`rounded-xl border p-3 text-xs md:col-span-2 ${
                          message.result.explain_meta.error_code == null
                            ? "border-green-200 bg-green-50 dark:border-green-900 dark:bg-green-950/30"
                            : "border-amber-200 bg-amber-50 dark:border-amber-900 dark:bg-amber-950/30"
                        }`}
                      >
                        explain_source:{" "}
                        {message.result.explain_meta.explain_source ??
                          "unknown"}
                        {" | "}
                        {message.result.explain_meta.error_code == null ? (
                          <span>status: ok</span>
                        ) : (
                          <span>
                            error_code:{" "}
                            {message.result.explain_meta.error_code ?? "none"}
                          </span>
                        )}
                      </div>
                    ) : null}
                  </div>
                ) : null}

                {message.isError ? (
                  <p className="text-xs text-red-600 dark:text-red-400">
                    Check API logs and CORS settings if this persists.
                  </p>
                ) : null}
              </article>
            ))}

            {isLoading ? (
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                Analyzing complaint...
              </p>
            ) : null}
          </div>
        </section>

        <Footer />
      </main>
    </div>
  );
}

function Nav() {
  return (
    <nav className="sticky top-0 z-20 border-b border-zinc-200 bg-white/80 backdrop-blur dark:border-zinc-800 dark:bg-zinc-950/80">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-lg bg-linear-to-br from-blue-600 to-purple-600" />
          <span className="font-semibold tracking-tight">Complaint Analyst</span>
          <span className="hidden text-xs text-zinc-500 md:inline">
            Arabic complaint classifier
          </span>
        </div>
        <div className="flex items-center gap-4 text-sm">
          <a
            href="#how"
            className="text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
          >
            How it works
          </a>
          <a
            href="#demo"
            className="text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
          >
            Demo
          </a>
          <a
            href="https://github.com"
            className="rounded-lg border border-zinc-200 px-3 py-1 text-zinc-700 transition hover:border-zinc-300 hover:bg-zinc-50 dark:border-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-800"
          >
            GitHub
          </a>
        </div>
      </div>
    </nav>
  );
}

function Hero() {
  return (
    <section className="mb-12 md:mb-16">
      <div className="inline-flex items-center gap-2 rounded-full border border-zinc-200 bg-white px-3 py-1 text-xs text-zinc-600 shadow-sm dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-400">
        <span className="inline-block h-1.5 w-1.5 rounded-full bg-emerald-500" />
        Live demo · CPU inference · ~100 ms p50
      </div>
      <h1 className="mt-4 text-3xl font-bold leading-tight tracking-tight md:text-5xl">
        Production-grade
        <br className="hidden md:block" /> Arabic complaint classification.
      </h1>
      <p className="mt-4 max-w-2xl text-base text-zinc-600 dark:text-zinc-400 md:text-lg">
        Three MARBERT classifiers (sentiment, topic, intent) feed a
        deterministic rule engine. An optional LLM explains the routing
        decision in structured JSON. Failures route to manual review instead
        of being silently swallowed.
      </p>
      <div className="mt-6 flex flex-wrap gap-3">
        <a
          href="#demo"
          className="rounded-xl bg-zinc-900 px-5 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-zinc-800 dark:bg-white dark:text-zinc-900 dark:hover:bg-zinc-200"
        >
          Try the demo
        </a>
        <a
          href="#how"
          className="rounded-xl border border-zinc-200 bg-white px-5 py-2 text-sm font-medium text-zinc-800 transition hover:border-zinc-300 hover:bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200 dark:hover:bg-zinc-800"
        >
          How it works
        </a>
      </div>
    </section>
  );
}

function BenchmarkStrip() {
  return (
    <section className="mb-16 grid grid-cols-2 gap-3 rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm md:grid-cols-4 dark:border-zinc-800 dark:bg-zinc-900 md:p-6">
      <Stat label="OOD accuracy" value={BENCHMARK.accuracy} />
      <Stat label="F1 (positive)" value={BENCHMARK.f1} />
      <Stat label="p50 latency" value={BENCHMARK.latencyP50} />
      <Stat label="p95 latency" value={BENCHMARK.latencyP95} />
      <p className="col-span-2 mt-2 text-xs text-zinc-500 md:col-span-4 dark:text-zinc-400">
        Out-of-distribution evaluation on {BENCHMARK.samples} stratified
        samples from{" "}
        <code className="rounded bg-zinc-100 px-1 py-0.5 text-xs dark:bg-zinc-800">
          {BENCHMARK.dataset}
        </code>
        . Sentiment model was trained on Saudi government complaints — the
        benchmark tests how it generalizes to general Arabic tweets, which is
        the realistic production case. Full report:{" "}
        <a
          href="https://github.com"
          className="underline underline-offset-2 hover:text-blue-600"
        >
          benchmarks/REPORT.md
        </a>
        .
      </p>
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-1 rounded-xl bg-zinc-50 p-4 dark:bg-zinc-950">
      <span className="text-xs uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
        {label}
      </span>
      <span className="text-2xl font-semibold tabular-nums">{value}</span>
    </div>
  );
}

function HowItWorks() {
  const steps = [
    {
      n: "01",
      title: "Validate input",
      body: "Pydantic schema rejects empty, oversized, or malformed payloads with a stable error envelope and a request_id.",
    },
    {
      n: "02",
      title: "Run three classifiers",
      body: "MARBERT-v2 fine-tunes for sentiment, topic, and intent run in sequence. Each output is mapped to a typed enum — unknown labels raise PredictionError, never silently default.",
    },
    {
      n: "03",
      title: "Rule engine decides",
      body: "A deterministic rule table combines the three labels into one action (e.g., FINANCIAL_ESCALATION, BLOCK_AND_REVIEW). The decision source is recorded in the response.",
    },
    {
      n: "04",
      title: "Confidence guard",
      body: "If any classifier scores below threshold, action is overridden to MANUAL_REVIEW. Low-confidence routing is treated as a feature, not a bug.",
    },
    {
      n: "05",
      title: "Optional LLM explanation",
      body: "On /explain-classification the LLM is called with a JSON schema and a strict timeout. On timeout / non-2xx / invalid JSON, classification is still returned and explain_meta records the failure.",
    },
    {
      n: "06",
      title: "Observability",
      body: "Every step logs through structlog as JSON with the same request_id. The same id is returned in x-request-id so client logs and server logs can be correlated.",
    },
  ];

  return (
    <section id="how" className="mb-16 scroll-mt-24">
      <h2 className="mb-2 text-2xl font-semibold">How it works</h2>
      <p className="mb-6 max-w-3xl text-sm text-zinc-600 dark:text-zinc-400">
        The pipeline keeps the deterministic ML decision separate from the LLM
        explanation. The LLM never changes routing — it only describes what
        already happened.
      </p>
      <ol className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
        {steps.map((s) => (
          <li
            key={s.n}
            className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900"
          >
            <div className="text-xs font-mono text-zinc-400">{s.n}</div>
            <h3 className="mt-1 text-base font-semibold">{s.title}</h3>
            <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
              {s.body}
            </p>
          </li>
        ))}
      </ol>
    </section>
  );
}

function Footer() {
  return (
    <footer className="mt-16 border-t border-zinc-200 pt-8 text-sm text-zinc-500 dark:border-zinc-800 dark:text-zinc-400">
      <p>
        Demo UI for the Arabic complaint classification API (FastAPI backend,
        Next.js frontend).
      </p>
    </footer>
  );
}

function ResultCard({ title, score }: { title: string; score: LabelScore }) {
  return (
    <div className="rounded-xl border border-zinc-200 bg-white p-3 dark:border-zinc-700 dark:bg-zinc-900">
      <h3 className="text-sm font-semibold">{title}</h3>
      <p className="mt-1 text-sm">{score.label}</p>
      <p className="text-xs text-zinc-500 dark:text-zinc-400">
        confidence: {score.confidence.toFixed(2)}
      </p>
      {score.low_confidence ? (
        <p className="mt-1 text-xs text-amber-600 dark:text-amber-400">
          Low confidence — routed to manual review.
        </p>
      ) : null}
    </div>
  );
}
