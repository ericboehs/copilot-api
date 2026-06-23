import { describe, test, expect } from "bun:test"

import { HTTPError } from "../src/lib/error"
import { readHttpErrorBody } from "../src/routes/messages/handler"

// Build an HTTPError wrapping a real Response so the body-reading path is
// exercised exactly as it is in production.
function httpError(status: number, body?: string): HTTPError {
  return new HTTPError("upstream error", new Response(body, { status }))
}

describe("readHttpErrorBody", () => {
  test("unwraps Copilot's nested error.message", async () => {
    const err = httpError(
      400,
      JSON.stringify({ error: { message: "model is overloaded" } }),
    )
    expect(await readHttpErrorBody(err)).toBe(
      "Copilot request failed (HTTP 400): model is overloaded",
    )
  })

  test("returns a bare text body verbatim", async () => {
    const err = httpError(429, "Too Many Requests")
    expect(await readHttpErrorBody(err)).toBe(
      "Copilot request failed (HTTP 429): Too Many Requests",
    )
  })

  test("falls back to status alone when the body is empty", async () => {
    const err = httpError(400, "")
    expect(await readHttpErrorBody(err)).toBe(
      "Copilot request failed (HTTP 400)",
    )
  })

  test("falls back to status alone when the body is whitespace only", async () => {
    const err = httpError(400, "   \n  ")
    expect(await readHttpErrorBody(err)).toBe(
      "Copilot request failed (HTTP 400)",
    )
  })

  test("returns JSON verbatim when it lacks an error.message", async () => {
    const err = httpError(500, JSON.stringify({ detail: "boom" }))
    expect(await readHttpErrorBody(err)).toBe(
      'Copilot request failed (HTTP 500): {"detail":"boom"}',
    )
  })

  test("returns status alone when reading the body throws", async () => {
    // A Response whose body has already been consumed throws on a second read.
    const response = new Response("already read", { status: 502 })
    await response.text()
    const err = new HTTPError("upstream error", response)
    expect(await readHttpErrorBody(err)).toBe(
      "Copilot request failed (HTTP 502)",
    )
  })
})
