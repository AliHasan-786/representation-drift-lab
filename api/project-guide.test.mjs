import assert from "node:assert/strict";
import test from "node:test";

import handler, { sensitiveInputReason } from "./project-guide.js";

function responseDouble() {
  return {
    headers: {},
    statusCode: null,
    body: null,
    setHeader(name, value) { this.headers[name] = value; },
    status(code) { this.statusCode = code; return this; },
    json(payload) { this.body = payload; return payload; },
  };
}

test("recognizes personal and secret-like input before provider use", () => {
  assert.equal(sensitiveInputReason("My email is ali@example.com"), "an email address");
  assert.equal(sensitiveInputReason("Call me at +1 (212) 555-0199"), "a phone number");
  assert.equal(sensitiveInputReason("My SSN is 123-45-6789"), "a Social Security number");
  assert.equal(sensitiveInputReason("My token is sk-abcdefghijklmnopqrstuvwxyz1234"), "an access credential");
  assert.equal(sensitiveInputReason("What does LoRA mean?"), null);
});

test("rejects sensitive input before checking a model-provider key", async () => {
  const response = responseDouble();
  await handler(
    {
      method: "POST",
      headers: { origin: "https://example.test", host: "example.test" },
      socket: { remoteAddress: "127.0.0.1" },
      body: { question: "My email is ali@example.com. What is CKA?" },
    },
    response,
  );
  assert.equal(response.statusCode, 400);
  assert.match(response.body.error, /do not enter an email address/i);
});
