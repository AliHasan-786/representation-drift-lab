import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import benchmark from "../../../public/data/benchmark-local.json";
import detail from "../../../public/data/reproduction-local.json";
import warning from "../../../public/data/early-warning-methodology.json";
import methods from "../../../public/data/method-comparison-local.json";
import interpolation from "../../../public/data/interpolation-local.json";
import domains from "../../../public/data/domain-comparison-local.json";
import gallery from "../../../public/data/dataset-gallery.json";
import App from "./App";

const response = (payload: unknown) =>
  Promise.resolve({ ok: true, json: () => Promise.resolve(payload) } as Response);

describe("portfolio visitor flow", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("benchmark-local")) return response(benchmark);
      if (url.includes("early-warning")) return response(warning);
      if (url.includes("method-comparison")) return response(methods);
      if (url.includes("interpolation-local")) return response(interpolation);
      if (url.includes("domain-comparison")) return response(domains);
      if (url.includes("dataset-gallery")) return response(gallery);
      if (url.includes("reproduction-local")) return response(detail);
      return Promise.resolve({ ok: false } as Response);
    }));
  });

  it("loads evidence, synchronizes checkpoint controls, and opens details on demand", async () => {
    render(<App />);
    expect(await screen.findByRole("heading", { name: /Can teaching an AI/i })).toBeInTheDocument();
    const checkpoint = screen.getByLabelText(/Training checkpoint/i);
    fireEvent.change(checkpoint, { target: { value: "3" } });
    expect(screen.getByText("step 20", { selector: "strong" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Load microscope/i }));
    expect(await screen.findByRole("heading", { name: "Class microscope" })).toBeInTheDocument();
    expect(screen.getByText(/Error anatomy/i)).toBeInTheDocument();
  });

  it("teaches one progressive story and links the complete report", async () => {
    render(<App />);
    expect(await screen.findByText(/Assume nothing/i)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /CLIP connects pictures with words/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /What are Food-101, CIFAR/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Does the same pattern hold/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Technical depth/i })).not.toBeInTheDocument();
    const reportLinks = screen.getAllByRole("link", { name: /report/i });
    expect(reportLinks.some((link) => link.getAttribute("href") === "/report/representation-drift-lab-report.pdf")).toBe(true);
    expect(reportLinks.some((link) => link.getAttribute("href") === "/report/original-course-report.pdf")).toBe(true);
  });

  it("shows real dataset examples and answers beginner questions offline", async () => {
    render(<App />);
    expect((await screen.findAllByAltText(/real Food-101 example/i)).length).toBeGreaterThanOrEqual(4);
    const question = screen.getByLabelText("Your question");
    fireEvent.change(question, { target: { value: "What is LoRA?" } });
    fireEvent.click(screen.getByRole("button", { name: "Ask the project" }));
    expect((await screen.findAllByText(/about 0.34% of the model/i)).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/works offline/i)).toBeInTheDocument();
  });

  it("explains the recovered source scaffold without presenting it as new results", async () => {
    render(<App />);
    expect(await screen.findByRole("heading", { name: /newly recovered source folder/i })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: /What did not run/i }));
    expect(screen.getByRole("heading", { name: /could not complete as written/i })).toBeInTheDocument();
    expect(screen.getByText(/7 explicit placeholders/i)).toBeInTheDocument();
  });

  it("supports keyboard navigation across teaching tabs", async () => {
    render(<App />);
    await screen.findByRole("heading", { name: /Can teaching an AI/i });
    const accuracyTab = screen.getByRole("tab", { name: "Accuracy lines" });
    fireEvent.keyDown(accuracyTab, { key: "ArrowRight" });
    expect(screen.getByRole("tab", { name: "Layer map" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("heading", { name: /Where inside the 12-block/i })).toBeInTheDocument();
  });

  it("shows a recoverable artifact failure state", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new Error("offline"))));
    render(<App />);
    expect(await screen.findByRole("heading", { name: /could not load/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
    await waitFor(() => expect(fetch).toHaveBeenCalled());
  });
});
